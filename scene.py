"""3D scene nodes (Cubric Studio MPI-623): equirect depth and fill lifting with MoGe v1 (vendored
in scene3d/moge, weights from models/moge/), and the 360 wrap helpers its graphs share.

Depth crosses to the host app as a raw little-endian float32 file (H rows of W values, no header)
under output/scenes/; each depth node returns that file's path, like MpiBrushTrain's .ply."""
import hashlib
import os

import cv2
import numpy as np
import torch

import comfy.model_management as mm  # type: ignore
import comfy.utils  # type: ignore
import folder_paths  # type: ignore

from .help_funcs import resolve_input_file
from .scene3d.lift import lift_depth
from .scene3d.moge.model import MoGeModel
from .scene3d.moge.panorama import panorama_depth
from .scene3d.wrap import pad_at_scale, wrap_crop, wrap_cut_merge, wrap_pad, wrap_soften

folder_paths.add_model_folder_path("moge", os.path.join(folder_paths.models_dir, "moge"))

# ponytail: the config of the one checkpoint we ship (Ruicheng/moge-vitl, re-saved as safetensors);
# a second MoGe checkpoint would carry its own config in the file's metadata.
MOGE_VITL_CONFIG = {
    "encoder": "dinov2_vitl14", "remap_output": "exp", "output_mask": True, "split_head": True,
    "intermediate_layers": 4, "dim_upsample": [256, 128, 64], "dim_times_res_block_hidden": 2,
    "num_res_blocks": 2, "trained_area_range": [250000, 500000], "last_conv_channels": 32,
    "last_conv_size": 1,
}
_MOGE = {}  # ponytail: one model kept on the CPU between runs (~1.3 GB RAM); drop it if RAM bites


def _moge_files():
    return [f for f in folder_paths.get_filename_list("moge") if f.endswith(".safetensors")]


def _load_moge(name):
    path = folder_paths.get_full_path_or_raise("moge", name)
    if _MOGE.get("path") != path:
        _MOGE.clear()
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        model = MoGeModel.from_state_dict(sd, MOGE_VITL_CONFIG)
        model.train(False)
        _MOGE.update(path=path, model=model)
    return _MOGE["model"]


class _OnDevice:
    """MoGe on the torch device for one run, back on the CPU after - the core upscaler's pattern,
    so ComfyUI evicts other models only when VRAM is short."""

    def __init__(self, model):
        self.model, self.device = model, mm.get_torch_device()

    def __enter__(self):
        mm.free_memory(mm.module_size(self.model) * 2, self.device)  # ponytail: weights + activations guess
        self.model.to(self.device)
        return self.device

    def __exit__(self, *exc):
        self.model.to("cpu")
        mm.soft_empty_cache()


def _save_f32(arr, name):
    out_dir = folder_paths.get_output_directory()
    full_dir, filename, counter, _sub, _prefix = folder_paths.get_save_image_path(
        f"scenes/{name}", out_dir, arr.shape[1], arr.shape[0])
    os.makedirs(full_dir, exist_ok=True)
    path = os.path.join(full_dir, f"{filename}_{counter:05}_.f32")
    np.ascontiguousarray(arr, dtype="<f4").tofile(path)
    return path


def _to_u8(image):
    return (image[0].cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)


class MpiPanoDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "A 2:1 equirectangular 360 panorama."}),
                "model": (_moge_files(), {"tooltip": "MoGe v1 weights in models/moge/."}),
                "resolution_level": ("INT", {"default": 9, "min": 0, "max": 9,
                                             "tooltip": "MoGe detail per view, 0-9."}),
                "depth_width": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 64,
                                        "tooltip": "Width of the depth grid; its height is half."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("depth_path",)
    FUNCTION = "run"
    CATEGORY = "MpiNodes/Scene"
    DESCRIPTION = ("Equirect depth of a 360 pano with MoGe v1 (12 views merged). Writes a raw float32 "
                   "file, depth_width/2 rows x depth_width, sky pushed to twice the farthest depth, "
                   "under output/scenes/ and returns its path.")

    def run(self, image, model, resolution_level, depth_width):
        w, h = depth_width, depth_width // 2
        src = cv2.resize(_to_u8(image), (w, h), interpolation=cv2.INTER_AREA)
        moge = _load_moge(model)
        with _OnDevice(moge) as device:
            depth, valid = panorama_depth(moge, src, device, resolution_level=resolution_level)
        if not valid.any():
            raise ValueError("MpiPanoDepth: MoGe found no valid depth in this panorama")
        depth = depth.copy()
        depth[~valid] = 2.0 * float(depth[valid].max())  # sky -> a far dome
        return (_save_f32(depth, "pano_depth"),)


class MpiLiftDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The fill: a render whose holes a model painted."}),
                "known_depth": ("STRING", {"default": "", "tooltip": (
                    "Raw float32 file in input/ (same size as the image, row-major): the camera z the "
                    "scene already has at each pixel, 0 where nothing was rendered.")}),
                "fov_x": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 179.0, "step": 0.01,
                                    "tooltip": "Horizontal field of view of the render, degrees."}),
                "model": (_moge_files(), {"tooltip": "MoGe v1 weights in models/moge/."}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("depth_path", "fit_error")
    FUNCTION = "run"
    CATEGORY = "MpiNodes/Scene"
    DESCRIPTION = ("Lift a fill into the scene: MoGe depth of the image, scaled and shifted by least "
                   "squares to the known camera z, kept on the holes. Writes a raw float32 file (0 = not "
                   "kept) under output/scenes/; fit_error is the median relative error on known pixels.")

    @classmethod
    def IS_CHANGED(cls, known_depth, **_):
        """Re-run when the file's bytes change under the same name, as core LoadImage does."""
        path = resolve_input_file(known_depth)
        if not path or not os.path.isfile(path):
            return ""
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def run(self, image, known_depth, fov_x, model):
        h, w = image.shape[1], image.shape[2]
        path = resolve_input_file(known_depth)
        if not path or not os.path.isfile(path):
            raise ValueError(f"MpiLiftDepth: known_depth file not found in input/: {known_depth!r}")
        raw = np.fromfile(path, dtype="<f4")
        if raw.size != h * w:
            raise ValueError(f"MpiLiftDepth: known_depth holds {raw.size} values, the image is {w}x{h}")
        moge = _load_moge(model)
        with _OnDevice(moge) as device:
            zm = moge.infer(image[0].permute(2, 0, 1).to(device), fov_x=fov_x)["depth"]
            zc = torch.from_numpy(raw.reshape(h, w)).to(device)
            depth, _a, _b, rel = lift_depth(zm.float(), zc)
            depth = depth.cpu().numpy()
        return (_save_f32(depth, "lift_depth"), rel)


class MpiWrapPad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad": ("INT", {"default": 32, "min": 0, "max": 2048,
                                "tooltip": "Columns copied from the opposite edge onto each side."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "MpiNodes/Scene"
    DESCRIPTION = ("Pad a 360 pano with its own opposite edges, so an upscaler or refiner sees the wrap "
                   "like any other column. Undo with Mpi Wrap Crop or Mpi Wrap Cut Merge.")

    def run(self, image, pad):
        return (wrap_pad(image, pad),)


class MpiWrapCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "A wrap-padded image, resized or not since."}),
                "reference": ("IMAGE", {"tooltip": "The image as it was BEFORE Mpi Wrap Pad."}),
                "pad": ("INT", {"default": 32, "min": 0, "max": 2048, "tooltip": "Mpi Wrap Pad's pad."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "MpiNodes/Scene"
    DESCRIPTION = "Crop Mpi Wrap Pad's columns off again, at whatever scale the image is now."

    def run(self, image, reference, pad):
        return (wrap_crop(image, reference.shape[2], pad),)


class MpiWrapSoften:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "A 360 pano whose left and right edges meet."}),
                "sigma": ("FLOAT", {"default": 12.0, "min": 0.5, "max": 64.0, "step": 0.5,
                                    "tooltip": "Horizontal blur across the wrap, px."}),
                "radius": ("INT", {"default": 48, "min": 1, "max": 512,
                                   "tooltip": "Blend width each side of the wrap, px."}),
                "threshold": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 64.0, "step": 0.5, "tooltip": (
                    "Soften only where the blur changes less than this many levels (flat sky); 0 = "
                    "soften everywhere along the wrap.")}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "MpiNodes/Scene"
    DESCRIPTION = ("Soften the hard 1 px line where a 360 pano's edges meet, in flat areas only, so a "
                   "following seam pass joins it instead of turning it into a light ridge.")

    def run(self, image, sigma, radius, threshold):
        out = [torch.from_numpy(wrap_soften(im.cpu().numpy().astype(np.float32) * 255.0, sigma, radius,
                                            threshold) / 255.0).float().clamp(0, 1) for im in image]
        return (torch.stack(out),)


class MpiWrapCutMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "A wrap-padded image, refined or upscaled since."}),
                "reference": ("IMAGE", {"tooltip": "The image as it was BEFORE Mpi Wrap Pad."}),
                "pad": ("INT", {"default": 128, "min": 8, "max": 2048, "tooltip": "Mpi Wrap Pad's pad."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "MpiNodes/Scene"
    DESCRIPTION = ("Crop Mpi Wrap Pad's columns off a refined image, joining the two renders of the "
                   "wrap strip along a hidden cut instead of a cross-fade (no ghost outlines).")

    def run(self, image, reference, pad):
        overlap = pad_at_scale(image.shape[2], reference.shape[2], pad)
        out = [torch.from_numpy(wrap_cut_merge(im.cpu().numpy().astype(np.float32) * 255.0, overlap) / 255.0)
               .float().clamp(0, 1) for im in image]
        return (torch.stack(out),)
