"""Depth nodes (Cubric Studio MPI-1605): Video Depth Anything (vendored in depth3d/vda, weights from
models/depth/) as a grey depth picture or a grey depth video, near = white.

MpiDepthVideo runs upstream's offline windows as a stream (depth3d/stream.py) in two passes over one
VideoStream: pass 1 decodes, infers and spills the raw depth (float16, network size) to a temp file
while it tracks the clip's min/max; pass 2 maps that to grey with the WHOLE clip's range (a per-chunk
range would flicker), scales it to the source size and encodes. RAM stays at two windows of frames
whatever the clip length; same frame count and fps as the source, its sound copied back.
"""
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm  # type: ignore
import comfy.utils  # type: ignore
import folder_paths  # type: ignore
from comfy.model_management import throw_exception_if_processing_interrupted  # type: ignore

from .depth3d.stream import DepthWindows, VideoDepthAnything, encoder_of, net_size, prepare, to_grey
from .help_funcs import PICKER_STRING_INPUT, picked_names, picker_choices, resolve_input_file
from .scene import _OnDevice
from .upscale_video import _frame_estimate
from .video import VIDEO_EXTS
from .video_stream import VideoStream, probe

folder_paths.add_model_folder_path("depth", os.path.join(folder_paths.models_dir, "depth"))
_MODEL = {}  # one model kept on the CPU between runs (~110 MB)
CHUNK = 8  # source frames in RAM at once: 8 float frames of 1080p are ~200 MB, a second of them 600


def _depth_files():
    return [f for f in folder_paths.get_filename_list("depth") if f.endswith((".pth", ".safetensors"))]


def _load(name):
    path = folder_paths.get_full_path_or_raise("depth", name)
    if _MODEL.get("path") != path:
        _MODEL.clear()
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        model = VideoDepthAnything(encoder_of(sd))
        model.load_state_dict(sd, strict=True)
        model.train(False)
        _MODEL.update(path=path, model=model)
    return _MODEL["model"]


def _runner(model, device):
    """[1, T, 3, h, w] window -> [T, h, w] float32 numpy, fp16 autocast on CUDA as upstream."""
    def run(x):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            depth = model(x.to(device))
        return depth[0].float().cpu().numpy()
    return run


def _grey_frames(depth, lo, hi, h, w, device="cpu"):
    """[N, h', w'] relative depth -> [N, 3, h, w] grey frames in 0..1 at the source size."""
    g = torch.from_numpy(to_grey(np.asarray(depth, dtype=np.float32), lo, hi)).unsqueeze(1).to(device)
    g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
    return g.expand(-1, 3, -1, -1)


class MpiDepthImage:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (_depth_files(), {"tooltip": "A Video Depth Anything checkpoint from models/depth/."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "run"
    CATEGORY = "MpiNodes/Depth"
    SEARCH_ALIASES = ["depth", "depth map", "depth anything", "extract depth"]
    DESCRIPTION = (
        "A grey depth map of each picture, near = white, far = black, at the picture's size. "
        "Video Depth Anything on one frame; each picture is scaled to its own nearest and farthest point."
    )

    def run(self, image, model):
        vda = _load(model)
        h, w = image.shape[1:3]
        size = net_size(h, w)
        out = []
        with _OnDevice(vda) as device:
            infer = _runner(vda, device)
            for i in range(image.shape[0]):
                throw_exception_if_processing_interrupted()
                d = infer(prepare(image[i:i + 1].to(device), size).unsqueeze(0))
                out.append(_grey_frames(d, float(d.min()), float(d.max()), h, w, device).cpu())
        return (torch.cat(out).permute(0, 2, 3, 1).contiguous(),)


class MpiDepthVideo:

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
                "model": (_depth_files(), {"tooltip": "A Video Depth Anything checkpoint from models/depth/."}),
                "filename_prefix": ("STRING", {"default": "MpiDepth"}),
            },
            "optional": {
                "string": PICKER_STRING_INPUT,
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "MpiNodes/Depth"
    SEARCH_ALIASES = ["video depth", "depth video", "depth anything", "extract depth"]
    DESCRIPTION = (
        "A grey depth video of a video file, near = white, same size, frame count and fps, sound kept. "
        "Video Depth Anything's own 32-frame windows, so the depth does not flicker, run as a stream: "
        "RAM stays flat on any clip length. `string` (a file inside input/, output/ or temp/) is tried "
        "first, then the picker. Outputs the written mp4's path."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, video=None):
        return True  # same reason as MpiLoadVideoUpload: injected names are not in the list

    def run(self, video, model, filename_prefix="MpiDepth", string=""):
        path = next((p for p in map(resolve_input_file, picked_names(video, string)) if p and os.path.isfile(p)), None)
        if not path:
            raise RuntimeError(f"[MpiDepthVideo] no video to read (string={string!r}, video={video!r}).")
        vda = _load(model)
        ffmpeg, w, h, fps = probe("MpiDepthVideo", path)
        size = net_size(h, w)
        total = _frame_estimate(ffmpeg, path, fps)
        pbar = comfy.utils.ProgressBar(2 * total) if total else None
        frame_bytes = size[0] * size[1] * 2  # float16
        device = mm.get_torch_device()  # resizes run there too: on the CPU they were most of the time
        lo, hi, spilled = np.inf, -np.inf, 0

        with tempfile.TemporaryFile(dir=folder_paths.get_temp_directory()) as spill, \
                VideoStream("MpiDepthVideo", path, w, h, fps, filename_prefix) as vs:

            def keep(frames):
                nonlocal lo, hi, spilled
                for d in frames:
                    lo, hi = min(lo, float(d.min())), max(hi, float(d.max()))
                    spill.write(d.astype(np.float16).tobytes())
                spilled += len(frames)
                if pbar:
                    pbar.update_absolute(min(spilled, total), 2 * total)

            # Pass 1: depth for every frame, in order, onto disk.
            with _OnDevice(vda) as device:
                windows = DepthWindows(_runner(vda, device), size)
                while (chunk := vs.read(CHUNK)) is not None:
                    throw_exception_if_processing_interrupted()
                    keep(windows.push(chunk.to(device).permute(0, 2, 3, 1)))
                keep(windows.finish())

            # Pass 2: grey with the clip's own range, at the source size.
            spill.seek(0)
            while buf := spill.read(frame_bytes * CHUNK):
                throw_exception_if_processing_interrupted()
                depth = np.frombuffer(buf, dtype=np.float16).reshape(-1, *size)
                vs.write(_grey_frames(depth, lo, hi, h, w, device))
                if pbar:
                    pbar.update_absolute(min(total + vs.written, 2 * total), 2 * total)
            result = vs.finish()

        print(f"[MpiNodes/DepthVideo] {w}x{h} at {fps:g} fps: {vs.written} depth frames (network {size[1]}x{size[0]})")
        return result


NODE_CLASS_MAPPINGS = {
    "MpiDepthImage": MpiDepthImage,
    "MpiDepthVideo": MpiDepthVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiDepthImage": "Mpi Depth Image",
    "MpiDepthVideo": "Mpi Depth Video",
}
