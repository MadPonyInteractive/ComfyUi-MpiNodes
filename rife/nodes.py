"""RIFE frame interpolation for Cubric Studio's Interpolate workflow (MPI-1342).

Ported from ComfyUI-Frame-Interpolation (vfi_models/rife/__init__.py, vfi_utils.py), MIT licence,
Copyright (c) 2023 Fannovel16. See rife/LICENSE-Frame-Interpolation. Only RIFE 4.7 (rife47.pth)
is ported, with the inputs the Flow sets; the other 14 VFI models, the interpolation-state
list and the cupy/taichi ops are not needed and not carried.

Differences from upstream, both forced by the Comfy Registry rules (.claude/rules/registry-safety.md):
  * No runtime download. Upstream fetched the weight from GitHub/Hugging Face mirrors on first
    use. Here the weight is looked up by NAME in two fixed folders and a missing file raises an
    error naming them. Cubric Studio installs rife47.pth through its own download manager.
  * torch.load(weights_only=True) instead of an unrestricted load. rife47.pth is a plain state dict.

Needs only torch (engine-owned). Upstream's einops / kornia / scipy / opencv / cupy / taichi /
torchvision are not used by this path.
"""
import gc
import os

import folder_paths
import torch
from comfy.model_management import get_torch_device, soft_empty_cache

from .arch import IFNet

CKPT_NAME_VER_DICT = {
    "rife47.pth": "4.7",
}

DTYPE_OPTIONS = ["float32", "float16", "bfloat16"]
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Module-level model cache: avoids reloading weights on every node execution.
# Key: (ckpt_name, dtype, torch_compile)
_model_cache = {}


def rife_weight_dirs():
    """Where rife47.pth may live: models/rife (the new home) and the folder the old
    ComfyUI-Frame-Interpolation pack scanned (<ComfyUI>/custom_nodes/<pack>/ckpts/rife), so the
    weight users already downloaded keeps working."""
    return [
        os.path.join(folder_paths.models_dir, "rife"),
        os.path.join(folder_paths.base_path, "custom_nodes", "comfyui-frame-interpolation", "ckpts", "rife"),
    ]


def find_rife_weight(ckpt_name):
    if ckpt_name not in CKPT_NAME_VER_DICT:  # the dropdown already guarantees this
        raise ValueError(f"Unknown RIFE checkpoint {ckpt_name!r}; expected one of {list(CKPT_NAME_VER_DICT)}")
    dirs = rife_weight_dirs()
    for d in dirs:
        p = os.path.join(d, ckpt_name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"RIFE weight {ckpt_name} not found. Put it in one of: " + "; ".join(dirs)
    )


def preprocess_frames(frames):
    """[n, h, w, c] -> [n, 3, h, w]"""
    return frames[..., :3].permute(0, 3, 1, 2)


def postprocess_frames(frames):
    """[n, c, h, w] -> [n, h, w, 3] on cpu"""
    return frames.permute(0, 2, 3, 1)[..., :3].cpu()


def build_scale_list(scale_factor):
    return [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]


def plan_tasks(n_frames, multiplier):
    """One (pair_idx, timestep) task per intermediate frame: pair i gets timesteps 1/m .. (m-1)/m."""
    tasks = []
    for pair_idx in range(n_frames - 1):
        for step in range(1, multiplier):
            tasks.append((pair_idx, step / multiplier))
    return tasks


class MpiRifeVFI:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (list(CKPT_NAME_VER_DICT.keys()), {"default": "rife47.pth"}),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 1}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                "dtype": (DTYPE_OPTIONS, {"default": "float32"}),
                "torch_compile": ("BOOLEAN", {"default": False,
                                              "tooltip": "Compile the model with torch.compile() for 10-30% faster inference "
                                                         "after the first (warm-up) run. Requires PyTorch 2.0+."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                                       "tooltip": "Number of interpolation tasks per GPU call. "
                                                  "Higher values improve throughput but use more VRAM. "
                                                  "Set to 1 for the most conservative behaviour."}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "MpiNodes/Video"
    DESCRIPTION = "RIFE 4.7 frame interpolation: inserts multiplier-1 frames between each pair of frames."

    def _load_model(self, ckpt_name, dtype, torch_compile, device):
        cache_key = (ckpt_name, dtype, torch_compile)
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        torch_dtype = DTYPE_MAP[dtype]
        model = IFNet()
        model.load_state_dict(torch.load(find_rife_weight(ckpt_name), map_location="cpu", weights_only=True))
        if torch_dtype != torch.float32:
            model = model.to(torch_dtype)
        model.eval().to(device)
        if torch_compile:
            model = torch.compile(model)
        _model_cache[cache_key] = model
        return model

    def vfi(
        self,
        ckpt_name,
        frames,
        clear_cache_after_n_frames=10,
        multiplier=2,
        fast_mode=False,
        ensemble=False,
        scale_factor=1.0,
        dtype="float32",
        torch_compile=False,
        batch_size=1,
    ):
        torch_dtype = DTYPE_MAP[dtype]
        device = get_torch_device()
        interpolation_model = self._load_model(ckpt_name, dtype, torch_compile, device)

        frames = preprocess_frames(frames)
        n_pairs = len(frames) - 1
        multiplier = int(multiplier)
        scale_list = build_scale_list(scale_factor)

        tasks = plan_tasks(len(frames), multiplier)
        tasks_remaining_per_pair = {i: max(multiplier - 1, 0) for i in range(n_pairs)}
        results = {i: [] for i in range(n_pairs)}

        frames_processed_since_cache_clear = 0
        pos = 0

        with torch.inference_mode():
            while pos < len(tasks):
                batch_tasks = tasks[pos: pos + batch_size]

                frame0_list, frame1_list, timestep_list = [], [], []
                for pair_idx, dt in batch_tasks:
                    frame0_list.append(frames[pair_idx: pair_idx + 1])
                    frame1_list.append(frames[pair_idx + 1: pair_idx + 2])
                    timestep_list.append(dt)

                frame0_batch = torch.cat(frame0_list, dim=0).to(device, dtype=torch_dtype)
                frame1_batch = torch.cat(frame1_list, dim=0).to(device, dtype=torch_dtype)
                timestep_tensor = torch.tensor(timestep_list, dtype=torch_dtype, device=device).view(-1, 1, 1, 1)

                # Positional on purpose: identical to upstream, so fast_mode lands on IFNet's
                # `training` and ensemble on `fastmode` (see IFNet.forward).
                middle_frames = interpolation_model(
                    frame0_batch,
                    frame1_batch,
                    timestep_tensor,
                    scale_list,
                    fast_mode,
                    ensemble,
                ).clamp(0, 1).detach().cpu()

                for idx, (pair_idx, _) in enumerate(batch_tasks):
                    results[pair_idx].append(middle_frames[idx: idx + 1].to(dtype=torch_dtype))
                    tasks_remaining_per_pair[pair_idx] -= 1
                    if tasks_remaining_per_pair[pair_idx] == 0:
                        frames_processed_since_cache_clear += 1
                        if frames_processed_since_cache_clear >= clear_cache_after_n_frames:
                            soft_empty_cache()
                            gc.collect()
                            frames_processed_since_cache_clear = 0

                pos += len(batch_tasks)

        output_frames = []
        for pair_idx in range(n_pairs):
            output_frames.append(frames[pair_idx: pair_idx + 1].to(dtype=torch_dtype))
            for mid in results[pair_idx]:
                output_frames.append(mid)
        output_frames.append(frames[-1:].to(dtype=torch_dtype))

        soft_empty_cache()
        print(f"[MpiNodes/RIFE] {ckpt_name} x{multiplier}: {sum(len(v) for v in results.values()) + len(frames)} frames")

        # Always return float32: downstream ComfyUI nodes require it
        out_tensor = torch.cat(output_frames, dim=0).to(torch.float32)
        return (postprocess_frames(out_tensor),)


NODE_CLASS_MAPPINGS = {
    "MpiRifeVFI": MpiRifeVFI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiRifeVFI": "Mpi RIFE VFI",
}
