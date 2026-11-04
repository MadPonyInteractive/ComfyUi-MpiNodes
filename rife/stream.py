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

import torch
from comfy.model_management import get_torch_device, soft_empty_cache, throw_exception_if_processing_interrupted

from ..help_funcs import (
    resolve_input_file,
    picker_choices,
    picked_names,
    PICKER_STRING_INPUT,
)
from ..video import VIDEO_EXTS
from ..video_stream import VideoStream, probe
from .nodes import CKPT_NAME_VER_DICT, DTYPE_MAP, DTYPE_OPTIONS, MpiRifeVFI, build_scale_list, interpolate_chunk


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
        path = next((p for p in map(resolve_input_file, picked_names(video, string)) if p and os.path.isfile(p)), None)
        if not path:
            raise RuntimeError(f"[MpiRifeVideo] no video to interpolate (string={string!r}, video={video!r}).")
        _, w, h, fps = probe("MpiRifeVideo", path)

        multiplier = max(int(multiplier), 1)
        out_fps = fps * multiplier
        chunk_frames = max(round(fps), 1)  # one second
        torch_dtype = DTYPE_MAP[dtype]
        device = get_torch_device()
        model = MpiRifeVFI()._load_model(ckpt_name, dtype, False, device)
        scale_list = build_scale_list(scale_factor)

        try:
            with VideoStream("MpiRifeVideo", path, w, h, out_fps, filename_prefix) as vs:
                prev = None
                while True:
                    throw_exception_if_processing_interrupted()
                    chunk = vs.read(chunk_frames)
                    if chunk is None:
                        break
                    frames = chunk if prev is None else torch.cat([prev, chunk])
                    vs.write(interpolate_chunk(model, frames, multiplier, scale_list, fast_mode, ensemble, torch_dtype, device, batch_size))
                    prev = frames[-1:]
                    soft_empty_cache()
                if prev is not None:
                    vs.write(prev)
                result = vs.finish()
        finally:
            soft_empty_cache()

        print(f"[MpiNodes/RIFE] {ckpt_name} x{multiplier}: {vs.written} frames at {out_fps:g} fps, streamed one second at a time")
        return result


NODE_CLASS_MAPPINGS = {
    "MpiRifeVideo": MpiRifeVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiRifeVideo": "Mpi RIFE Video",
}
