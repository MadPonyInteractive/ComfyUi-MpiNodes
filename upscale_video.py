"""MpiUpscaleVideo: upscale a video file to an mp4, a frame at a time (MPI-1418).

Load Video -> Upscale Image (using Model) -> Upscale Image By -> Save Video holds the whole clip in
RAM, and the model's 4x output for every frame is one CPU tensor: a ~10 s 1080p clip asked a 32 GB
PC for 99 GB. This node decodes a second of frames, runs each frame through the upscale model on
its own (ComfyUI's tiled upscale loops over frames anyway, so no speed is lost), scales it to the
target with lanczos at once, and pipes it to the encoder. RAM stays at about one 4x frame
whatever the clip length. Same pixels as the chain it replaces: same tiles, same OOM fallback,
same lanczos, same rounding.
"""
import os
import re
import subprocess

import comfy.utils
from comfy import model_management
from comfy.model_management import soft_empty_cache, throw_exception_if_processing_interrupted

from .help_funcs import resolve_input_file, picker_choices, picked_names, PICKER_STRING_INPUT
from .video import VIDEO_EXTS
from .video_stream import VideoStream, probe

TILE = 512
OVERLAP = 32


def _frame_estimate(ffmpeg, path, fps):
    """Frames in the clip from the container's Duration, for the progress bar only."""
    err = subprocess.run([ffmpeg, "-i", path], capture_output=True).stderr.decode("utf-8", "replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if not m:
        return 0
    seconds = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return max(round(seconds * fps), 1)


class _ModelUpscaler:
    """ImageUpscaleWithModel's body (comfy_extras/nodes_upscale_model.py), for one frame at a time.

    The tile size it falls back to on an out-of-memory error is kept for the frames after."""

    def __init__(self, upscale_model, frame_shape):
        self.model = upscale_model
        self.device = upscale_model.patcher.load_device
        self.tile = TILE
        n, c, h, w = frame_shape
        self.memory_required = (512 * 512 * 3) * 4 * max(upscale_model.scale, 1.0) * 384.0 + n * c * h * w * 4

    def __call__(self, frames):
        model_management.load_models_gpu([self.model.patcher], memory_required=self.memory_required, force_full_load=True)
        in_img = frames.to(self.device)
        while True:
            try:
                s = comfy.utils.tiled_scale(in_img, lambda a: self.model(a.float()), tile_x=self.tile, tile_y=self.tile,
                                            overlap=OVERLAP, upscale_amount=self.model.scale,
                                            output_device=model_management.intermediate_device())
                break
            except Exception as e:
                model_management.raise_non_oom(e)
                self.tile //= 2
                if self.tile < 128:
                    raise e
        return s.clamp(0, 1.0).to(model_management.intermediate_dtype())


class MpiUpscaleVideo:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": (
                    picker_choices(VIDEO_EXTS),
                    {
                        "video_upload": True,
                        "tooltip": "Pick a video, or use the upload button; the file goes into ComfyUI's input/ folder. Used when string is empty or does not load; None picks nothing.",
                    },
                ),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01,
                                       "tooltip": "Final size = source size x scale_by, whatever the model's own scale."}),
                "use_model": ("BOOLEAN", {"default": False,
                                          "tooltip": "On: each frame goes through upscale_model, then lanczos to the final size. Off: lanczos only, the model is never loaded."}),
                "filename_prefix": ("STRING", {"default": "MpiVideo"}),
            },
            "optional": {
                "string": PICKER_STRING_INPUT,
                "upscale_model": ("UPSCALE_MODEL", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "MpiNodes/Video"
    SEARCH_ALIASES = ["upscale video", "video upscale", "video super resolution"]
    DESCRIPTION = (
        "Upscale a video file straight to an .mp4 by scale_by, audio kept: through an upscale "
        "model then lanczos, or lanczos alone. Decodes, upscales and encodes a frame at a time, so "
        "RAM stays flat on any clip length (Load Video + Upscale Image (using Model) + Save Video "
        "hold every 4x frame in RAM at once). Same pixels as that chain. `string` (a file inside "
        "input/, output/ or temp/) is tried first, then the picker. Outputs the written mp4's path."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, video=None):
        return True  # same reason as MpiLoadVideoUpload: injected names are not in the list

    def check_lazy_status(self, video, scale_by, use_model, filename_prefix, string="", upscale_model=None):
        return ["upscale_model"] if use_model and upscale_model is None else []

    def run(self, video, scale_by=2.0, use_model=False, filename_prefix="MpiVideo", string="", upscale_model=None):
        path = next((p for p in map(resolve_input_file, picked_names(video, string)) if p and os.path.isfile(p)), None)
        if not path:
            raise RuntimeError(f"[MpiUpscaleVideo] no video to upscale (string={string!r}, video={video!r}).")
        if use_model and upscale_model is None:
            raise RuntimeError("[MpiUpscaleVideo] use_model is on but no upscale_model is connected.")
        ffmpeg, w, h, fps = probe("MpiUpscaleVideo", path)

        # The rounding of the graph this replaces: the model's output, then ImageScaleBy by
        # scale_by / model scale (MpiMath a/b), each side round()ed.
        model_scale = upscale_model.scale if use_model else 1
        out_w = round(w * model_scale * (scale_by / model_scale))
        out_h = round(h * model_scale * (scale_by / model_scale))
        upscaler = _ModelUpscaler(upscale_model, (1, 3, h, w)) if use_model else None
        total = _frame_estimate(ffmpeg, path, fps)
        pbar = comfy.utils.ProgressBar(total) if total else None
        chunk_frames = max(round(fps), 1)  # one second decoded at a time; the model sees one frame

        try:
            with VideoStream("MpiUpscaleVideo", path, out_w, out_h, fps, filename_prefix) as vs:
                while (chunk := vs.read(chunk_frames)) is not None:
                    for i in range(len(chunk)):
                        throw_exception_if_processing_interrupted()
                        frame = chunk[i:i + 1]
                        if upscaler:
                            frame = upscaler(frame)
                        vs.write(comfy.utils.common_upscale(frame, out_w, out_h, "lanczos", "disabled"))
                        if pbar:
                            pbar.update_absolute(min(vs.written, total), total)
                result = vs.finish()
        finally:
            soft_empty_cache()

        how = f"model x{model_scale:g} + lanczos" if use_model else "lanczos"
        print(f"[MpiNodes/UpscaleVideo] {w}x{h} -> {out_w}x{out_h} ({how}): {vs.written} frames at {fps:g} fps, one frame at a time")
        return result


NODE_CLASS_MAPPINGS = {
    "MpiUpscaleVideo": MpiUpscaleVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiUpscaleVideo": "Mpi Upscale Video",
}
