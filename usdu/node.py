"""MpiUltimateSDUpscale, ported from ComfyUI_UltimateSDUpscale for Cubric Studio (MPI-1343).

Ported from ssitu/ComfyUI_UltimateSDUpscale (usdu_nodes.py, usdu_patch.py, usdu_utils.py,
crop_model_patch.py, modules/), pinned at bebd5696, GPL-3.0, wrapping Coyote-A's
ultimate-upscale-for-automatic1111 script, GPL-3.0. See LICENSE-ComfyUI_UltimateSDUpscale and
LICENSE-ultimate-upscale-for-automatic1111. The node keeps the original's input names, defaults and
choices, so a workflow only has to change its class_type from UltimateSDUpscale. Only the code paths
that node reaches are carried:

  * Only UltimateSDUpscale; the NoUpscale and CustomSample variants are not carried, and with them
    the custom_sampler / custom_sigmas path (SamplerCustom) inside the tile pipeline.
  * The A1111 script is vendored as upscale_script.py (no runtime download, no Gradio UI); the
    original's import-path and module-table juggling is gone, every import is relative.
  * No config.json read (the pack ships none): per-tile progress stays at its default of True.
  * No Pillow compat shim (it patched Image.Resampling globally).

Needs only torch, numpy, pillow, torchvision, tqdm and comfy (all engine-owned).
"""
import logging
from contextlib import contextmanager

import torch
import comfy.samplers  # type: ignore

from .utils import tensor_to_pil, pil_to_tensor
from .modules.processing import StableDiffusionProcessing
from .modules import shared
from .modules.upscaler import UpscalerData
from . import usdu_patch  # noqa: F401  (applies the batch patches to the vendored script)
from . import upscale_script as usdu

logger = logging.getLogger(__name__)

CATEGORY = "MpiNodes/Upscale"
MAX_RESOLUTION = 8192

MODES = {
    "Linear": usdu.USDUMode.LINEAR,
    "Chess": usdu.USDUMode.CHESS,
    "None": usdu.USDUMode.NONE,
}
SEAM_FIX_MODES = {
    "None": usdu.USDUSFMode.NONE,
    "Band Pass": usdu.USDUSFMode.BAND_PASS,
    "Half Tile": usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


@contextmanager
def suppress_logging(level=logging.CRITICAL + 1):
    """Temporarily suppress logging output (duplicate tqdm progress bars)."""
    root_logger = logging.getLogger()
    old_level = root_logger.getEffectiveLevel()
    root_logger.setLevel(level)
    try:
        yield
    finally:
        root_logger.setLevel(old_level)


class MpiUltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "The image to upscale."}),
            # Sampling Params
            "model": ("MODEL", {"tooltip": "The model to use for image-to-image."}),
            "positive": ("CONDITIONING", {"tooltip": "The positive conditioning for each tile."}),
            "negative": ("CONDITIONING", {"tooltip": "The negative conditioning for each tile."}),
            "vae": ("VAE", {"tooltip": "The VAE model to use for tiles."}),
            "upscale_by": ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "tooltip": "The factor to upscale the image by."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The seed to use for image-to-image."}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "The number of steps to use for each tile."}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "The CFG scale to use for each tile."}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler to use for each tile."}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler to use for each tile."}),
            "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for each tile."}),
            # Upscale Params
            "upscale_model": ("UPSCALE_MODEL", {"tooltip": "The upscaler model for upscaling the image."}),
            "mode_type": (list(MODES.keys()), {"tooltip": "The tiling order to use for the redraw step."}),
            "tile_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."}),
            "tile_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."}),
            "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."}),
            "tile_padding": ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."}),
            # Seam fix params
            "seam_fix_mode": (list(SEAM_FIX_MODES.keys()), {"tooltip": "The seam fix mode to use."}),
            "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for the seam fix."}),
            "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."}),
            "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."}),
            "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."}),
            # Misc
            "force_uniform_tiles": ("BOOLEAN", {"default": True, "tooltip": "Force all tiles to be the same as the set tile size, even when tiles could be smaller. This can help prevent the model from working with irregular tile sizes."}),
            "tiled_decode": ("BOOLEAN", {"default": False, "tooltip": "Whether to use tiled decoding when decoding tiles."}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "tooltip": "The number of tiles to process in a batch. Higher values can reduce processing time but use more VRAM. Yields different results than individual tiles. Only affects the main redraw step, not the seam fix step."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = CATEGORY
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Upscales an image and runs image-to-image on tiles from the input image."

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1):
        redraw_mode = MODES[mode_type]
        seam_fix_mode = SEAM_FIX_MODES[seam_fix_mode]

        # The A1111 script works through module-level state (as in the original pack).
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        assert batch_size == 1 or force_uniform_tiles, "batch_size greater than 1 requires force_uniform_tiles to be True; all tiles in the batch must be the same size."

        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles, tiled_decode,
            tile_width, tile_height, redraw_mode, seam_fix_mode, batch_size,
        )

        with suppress_logging():
            script = usdu.Script()
            script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                       mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                       seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                       upscaler_index=0, save_upscaled_image=False, redraw_mode=redraw_mode,
                       save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                       seams_fix_type=seam_fix_mode, target_size_type=2,
                       custom_width=None, custom_height=None, custom_scale=upscale_by)

        images = [pil_to_tensor(img) for img in shared.batch]
        return (torch.cat(images, dim=0),)


NODE_CLASS_MAPPINGS = {"MpiUltimateSDUpscale": MpiUltimateSDUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"MpiUltimateSDUpscale": "Mpi Ultimate SD Upscale"}
