"""MpiRifeVideo: RIFE frame interpolation from a video file to a video file, one second of frames
in RAM at a time (MPI-1397).

The Load Video -> MpiRifeVFI -> Save Video chain holds the whole clip in RAM as float32 three
times over: a minute of 1080p at x2 asked a user's PC for 72 GB in one allocation. A ComfyUI
IMAGE is one tensor, so no node in that chain can stream. This node does the decode, the
interpolation and the encode itself: ffmpeg decodes a second of frames, RIFE fills them in, the
result is piped to the encoder, and the last frame carries over as the first of the next second,
so the output is frame-for-frame what MpiRifeVFI makes. RAM stays flat whatever the clip length.

Kept out of rife/__init__.py on purpose: it needs the pack's ffmpeg helpers (..video,
..help_funcs), and tests/test_rife.py loads rife/ on its own.
"""
import os
import subprocess
import tempfile
import threading

import folder_paths  # type: ignore
import numpy as np
import torch
from comfy.model_management import get_torch_device, soft_empty_cache, throw_exception_if_processing_interrupted

from ..help_funcs import (
    find_ffmpeg,
    video_has_audio_stream,
    resolve_input_file,
    picker_choices,
    picked_names,
    PICKER_STRING_INPUT,
)
from ..video import VIDEO_EXTS, _decode_cmd, _probe_video_meta
from .nodes import CKPT_NAME_VER_DICT, DTYPE_MAP, DTYPE_OPTIONS, MpiRifeVFI, build_scale_list, interpolate_chunk


def _drain(stream, sink):
    """Read a pipe to the end on a thread, so a chatty ffmpeg never blocks on a full stderr."""
    t = threading.Thread(target=lambda: sink.append(stream.read()), daemon=True)
    t.start()
    return t


def _tail(chunks):
    return (chunks[0] if chunks else b"").decode("utf-8", "replace")[-2000:]


class MpiRifeVideo:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (list(CKPT_NAME_VER_DICT.keys()), {"default": "rife47.pth"}),
                "video": (
                    picker_choices(VIDEO_EXTS),
                    {
                        "video_upload": True,
                        "tooltip": "Pick a video, or use the upload button; the file goes into ComfyUI's input/ folder. Used when string is empty or does not load; None picks nothing.",
                    },
                ),
                "multiplier": ("INT", {"default": 2, "min": 1}),
                "fast_mode": ("BOOLEAN", {"default": False}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                "dtype": (DTYPE_OPTIONS, {"default": "float32"}),
                "filename_prefix": ("STRING", {"default": "MpiVideo"}),
            },
            "optional": {
                "string": PICKER_STRING_INPUT,
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                                       "tooltip": "Interpolation tasks per GPU call. Higher is faster and uses more VRAM."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "MpiNodes/Video"
    SEARCH_ALIASES = ["rife video", "interpolate video", "frame interpolation"]
    DESCRIPTION = (
        "RIFE 4.7 frame interpolation from a video file straight to an .mp4: inserts multiplier-1 "
        "frames between each pair and plays at fps x multiplier, audio kept. Decodes, interpolates "
        "and encodes one second at a time, so RAM stays flat on any clip length (Load Video + "
        "RIFE VFI + Save Video hold the whole clip in RAM). Same frames as Mpi RIFE VFI. `string` "
        "(a file inside input/, output/ or temp/) is tried first, then the picker. Outputs the "
        "written mp4's path."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, video=None):
        return True  # same reason as MpiLoadVideoUpload: injected names are not in the list

    def run(self, ckpt_name, video, multiplier=2, fast_mode=False, ensemble=True, scale_factor=1.0,
            dtype="float32", filename_prefix="MpiVideo", string="", batch_size=1):
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("[MpiRifeVideo] no ffmpeg found (imageio-ffmpeg bundle / VHS_FORCE_FFMPEG_PATH / PATH all failed).")
        path = next((p for p in map(resolve_input_file, picked_names(video, string)) if p and os.path.isfile(p)), None)
        if not path:
            raise RuntimeError(f"[MpiRifeVideo] no video to interpolate (string={string!r}, video={video!r}).")
        w, h, fps = _probe_video_meta(ffmpeg, path)
        if not w or not h:
            raise RuntimeError(f"[MpiRifeVideo] could not read the video size of {os.path.basename(path)}.")

        multiplier = max(int(multiplier), 1)
        out_fps = fps * multiplier
        chunk_frames = max(round(fps), 1)  # one second
        frame_bytes = w * h * 3
        torch_dtype = DTYPE_MAP[dtype]
        device = get_torch_device()
        model = MpiRifeVFI()._load_model(ckpt_name, dtype, False, device)
        scale_list = build_scale_list(scale_factor)

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h)
        name = f"{filename}_{counter:05}.mp4"
        out_path = os.path.join(full_output_folder, name)
        has_audio = video_has_audio_stream(path)
        # With audio the frames go to a temp mp4 first: its length is only known at the end, and
        # the mux below cuts or pads the audio to it (as MpiSaveVideo does).
        video_path = tempfile.mktemp(suffix=".mp4", dir=full_output_folder) if has_audio else out_path

        dec = subprocess.Popen(_decode_cmd(ffmpeg, path), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        enc = subprocess.Popen([
            ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{out_fps:g}", "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",  # libx264/yuv420p need even dimensions
            "-preset", "fast", "-crf", "18", "-movflags", "+faststart", video_path,
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        dec_err, enc_err = [], []
        drains = [_drain(dec.stderr, dec_err), _drain(enc.stderr, enc_err)]

        def write(frames):
            data = frames.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
            enc.stdin.write(data.tobytes())

        written = 0
        try:
            prev = None
            while True:
                throw_exception_if_processing_interrupted()
                buf = dec.stdout.read(frame_bytes * chunk_frames)
                n = len(buf) // frame_bytes
                if n == 0:
                    break
                chunk = torch.from_numpy(np.frombuffer(buf[: n * frame_bytes], dtype=np.uint8).reshape(n, h, w, 3).copy())
                chunk = chunk.permute(0, 3, 1, 2).to(torch.float32).div(255)
                frames = chunk if prev is None else torch.cat([prev, chunk])
                out = interpolate_chunk(model, frames, multiplier, scale_list, fast_mode, ensemble, torch_dtype, device, batch_size)
                write(out)
                written += len(out)
                prev = frames[-1:]
                soft_empty_cache()
            if prev is None:
                raise RuntimeError(f"[MpiRifeVideo] ffmpeg decoded no frames from {os.path.basename(path)}:\n{_tail(dec_err)}")
            write(prev)
            written += 1
            enc.stdin.close()
            if dec.wait() != 0:
                raise RuntimeError(f"[MpiRifeVideo] ffmpeg decode failed:\n{_tail(dec_err)}")
            if enc.wait() != 0:
                raise RuntimeError(f"[MpiRifeVideo] ffmpeg encode failed:\n{_tail(enc_err)}")

            if has_audio:
                duration = written / out_fps
                mux = subprocess.run([
                    ffmpeg, "-y", "-i", video_path, "-i", path, "-map", "0:v", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-af", "apad", "-t", f"{duration:.6f}",
                    "-movflags", "+faststart", out_path,
                ], capture_output=True)
                if mux.returncode != 0:
                    raise RuntimeError(f"[MpiRifeVideo] ffmpeg audio mux failed:\n{mux.stderr.decode('utf-8', 'replace')[-2000:]}")
        except BrokenPipeError:
            # The encoder died mid-write; its own stderr says why.
            enc.wait()
            raise RuntimeError(f"[MpiRifeVideo] ffmpeg encode failed:\n{_tail(enc_err)}")
        finally:
            for proc in (dec, enc):
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            for t in drains:
                t.join()
            if has_audio and os.path.exists(video_path):
                os.remove(video_path)
            soft_empty_cache()

        print(f"[MpiNodes/RIFE] {ckpt_name} x{multiplier}: {written} frames at {out_fps:g} fps, streamed one second at a time")
        return {
            "ui": {"videos": [{"filename": name, "subfolder": subfolder, "type": "output", "format": "video/mp4"}]},
            "result": (out_path,),
        }


NODE_CLASS_MAPPINGS = {
    "MpiRifeVideo": MpiRifeVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiRifeVideo": "Mpi RIFE Video",
}
