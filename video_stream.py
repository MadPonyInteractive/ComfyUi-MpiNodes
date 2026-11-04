"""Video file in, video file out, a chunk of frames in RAM at a time (MPI-1397, MPI-1418).

A ComfyUI IMAGE is one tensor holding the whole clip, so Load Video -> <op> -> Save Video holds
every frame in RAM, several times over once an op makes it bigger (RIFE x2 asked a user's PC for
72 GB, a 4x model upscale for 99 GB). The streaming nodes do the decode and the encode themselves
through this class: ffmpeg decodes rgb24 frames to a pipe, the node reads a chunk, works on it,
and writes the result to the encoder's pipe. The source's audio is muxed back at the end, cut or
padded to the new video's length the way MpiSaveVideo does it.

    with VideoStream("MpiRifeVideo", path, out_w, out_h, out_fps, filename_prefix) as vs:
        while (chunk := vs.read(n)) is not None:
            vs.write(work(chunk))
        result = vs.finish()

Leaving the `with` early (Stop, an error) kills both ffmpeg children and removes the temp file.
"""
import os
import subprocess
import tempfile
import threading

import folder_paths  # type: ignore
import numpy as np
import torch

from .help_funcs import find_ffmpeg, video_has_audio_stream
from .video import _decode_cmd, _probe_video_meta


def _drain(stream, sink):
    """Read a pipe to the end on a thread, so a chatty ffmpeg never blocks on a full stderr."""
    t = threading.Thread(target=lambda: sink.append(stream.read()), daemon=True)
    t.start()
    return t


def _tail(chunks):
    return (chunks[0] if chunks else b"").decode("utf-8", "replace")[-2000:]


def probe(tag, path):
    """(ffmpeg, w, h, fps) of a source video, or a RuntimeError naming the node."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(f"[{tag}] no ffmpeg found (imageio-ffmpeg bundle / VHS_FORCE_FFMPEG_PATH / PATH all failed).")
    w, h, fps = _probe_video_meta(ffmpeg, path)
    if not w or not h:
        raise RuntimeError(f"[{tag}] could not read the video size of {os.path.basename(path)}.")
    return ffmpeg, w, h, fps


class VideoStream:
    def __init__(self, tag, path, out_w, out_h, out_fps, filename_prefix):
        self.tag = tag
        self.path = path
        self.ffmpeg, self.w, self.h, self.fps = probe(tag, path)
        self.out_w, self.out_h, self.out_fps = int(out_w), int(out_h), out_fps
        self.filename_prefix = filename_prefix
        self.written = 0

    def __enter__(self):
        full_output_folder, filename, counter, self.subfolder, _ = folder_paths.get_save_image_path(
            self.filename_prefix, folder_paths.get_output_directory(), self.out_w, self.out_h)
        self.name = f"{filename}_{counter:05}.mp4"
        self.out_path = os.path.join(full_output_folder, self.name)
        self.has_audio = video_has_audio_stream(self.path)
        # With audio the frames go to a temp mp4 first: its length is only known at the end, and
        # finish() cuts or pads the audio to it.
        self.video_path = tempfile.mktemp(suffix=".mp4", dir=full_output_folder) if self.has_audio else self.out_path

        self.dec = subprocess.Popen(_decode_cmd(self.ffmpeg, self.path), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.enc = subprocess.Popen([
            self.ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.out_w}x{self.out_h}", "-r", f"{self.out_fps:g}", "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",  # libx264/yuv420p need even dimensions
            "-preset", "fast", "-crf", "18", "-movflags", "+faststart", self.video_path,
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.dec_err, self.enc_err = [], []
        self.drains = [_drain(self.dec.stderr, self.dec_err), _drain(self.enc.stderr, self.enc_err)]
        return self

    def read(self, n):
        """Up to n source frames as float32 [n, 3, h, w] in 0..1, or None at the end."""
        frame_bytes = self.w * self.h * 3
        buf = self.dec.stdout.read(frame_bytes * n)
        n = len(buf) // frame_bytes
        if n == 0:
            return None
        chunk = torch.from_numpy(np.frombuffer(buf[: n * frame_bytes], dtype=np.uint8).reshape(n, self.h, self.w, 3).copy())
        return chunk.permute(0, 3, 1, 2).to(torch.float32).div(255)

    def write(self, frames):
        """Encode float [n, 3, out_h, out_w] frames in 0..1."""
        data = frames.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        try:
            self.enc.stdin.write(data.tobytes())
        except BrokenPipeError:
            # The encoder died mid-write; its own stderr says why.
            self.enc.wait()
            raise RuntimeError(f"[{self.tag}] ffmpeg encode failed:\n{_tail(self.enc_err)}")
        self.written += len(frames)

    def finish(self):
        """Close the encode, mux the audio back, and return the node's {"ui", "result"}."""
        if self.written == 0:
            raise RuntimeError(f"[{self.tag}] ffmpeg decoded no frames from {os.path.basename(self.path)}:\n{_tail(self.dec_err)}")
        self.enc.stdin.close()
        if self.dec.wait() != 0:
            raise RuntimeError(f"[{self.tag}] ffmpeg decode failed:\n{_tail(self.dec_err)}")
        if self.enc.wait() != 0:
            raise RuntimeError(f"[{self.tag}] ffmpeg encode failed:\n{_tail(self.enc_err)}")
        if self.has_audio:
            duration = self.written / self.out_fps
            mux = subprocess.run([
                self.ffmpeg, "-y", "-i", self.video_path, "-i", self.path, "-map", "0:v", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-af", "apad", "-t", f"{duration:.6f}",
                "-movflags", "+faststart", self.out_path,
            ], capture_output=True)
            if mux.returncode != 0:
                raise RuntimeError(f"[{self.tag}] ffmpeg audio mux failed:\n{mux.stderr.decode('utf-8', 'replace')[-2000:]}")
        return {
            "ui": {"videos": [{"filename": self.name, "subfolder": self.subfolder, "type": "output", "format": "video/mp4"}]},
            "result": (self.out_path,),
        }

    def __exit__(self, *exc):
        for proc in (self.dec, self.enc):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for t in self.drains:
            t.join()
        if self.has_audio and os.path.exists(self.video_path):
            os.remove(self.video_path)
        return False
