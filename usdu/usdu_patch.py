"""Batch/rounding patches over the vendored A1111 script, from ComfyUI_UltimateSDUpscale usdu_patch.py
(bebd5696), GPL-3.0, see LICENSE-ComfyUI_UltimateSDUpscale. Same patches, applied once at import to
the vendored classes in upscale_script.py. Only the debug logging differs (no handler is added to
the logger and the batch chatter is at debug level).
"""
from __future__ import annotations
from functools import wraps
import logging
import math
from typing import Tuple, List

from PIL import Image
from .modules import shared
from .modules.processing import process_batch_tiles
from . import upscale_script as usdu
from .modules import processing, images, devices

logger = logging.getLogger(__name__)


def round_length(length: int, multiple: int = 8) -> int:
    """Round length to nearest multiple (default 8)."""
    return round(length / multiple) * multiple


def patch_usdu_upscaler_init():
    """Patch USDUpscaler.__init__ to round upscaler p.width/p.height to multiples."""
    old_init = usdu.USDUpscaler.__init__

    @wraps(old_init)
    def new_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height):
        p.width = round_length(image.width * p.upscale_by)
        p.height = round_length(image.height * p.upscale_by)
        return old_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height)

    usdu.USDUpscaler.__init__ = new_init


def patch_usdu_redraw_init():
    """Patch USDURedraw.init_draw to round tile size used for redraw."""
    old_init_draw = usdu.USDURedraw.init_draw

    @wraps(old_init_draw)
    def new_init_draw(self, p, width, height):
        mask, draw = old_init_draw(self, p, width, height)
        p.width = round_length(self.tile_width + self.padding)
        p.height = round_length(self.tile_height + self.padding)
        return mask, draw

    usdu.USDURedraw.init_draw = new_init_draw


def patch_usdu_seams_fix_init():
    old_init = usdu.USDUSeamsFix.init_draw

    @wraps(old_init)
    def new_init(self, p):
        old_init(self, p)
        p.width = round_length(self.tile_width + self.padding)
        p.height = round_length(self.tile_height + self.padding)

    usdu.USDUSeamsFix.init_draw = new_init


def patch_usdu_upscale_method():
    """Patch USDUpscaler.upscale to keep shared.batch resized to p.width/p.height."""
    old_upscale = usdu.USDUpscaler.upscale

    @wraps(old_upscale)
    def new_upscale(self):
        old_upscale(self)
        # Keep shared.batch consistent with the upscaling width/height for subsequent processing.
        shared.batch = [self.image] + [
            img.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
            for img in shared.batch[1:]
        ]

    usdu.USDUpscaler.upscale = new_upscale


def patched_script_run(self, p, _, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding,
                      upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
                      seams_fix_type, target_size_type, custom_width, custom_height, custom_scale):
    """Script.run that preserves the original batch_size and delegates to the patched pipeline."""
    preserved_batch_size = getattr(p, 'batch_size', 1)
    logger.debug("USDU run, preserving batch_size=%s", preserved_batch_size)

    processing.fix_seed(p)
    devices.torch_gc()

    p.do_not_save_grid = True
    p.do_not_save_samples = True
    p.inpaint_full_res = False

    p.inpainting_fill = 1
    p.n_iter = 1
    p.batch_size = preserved_batch_size

    seed = p.seed

    init_img = p.init_images[0]
    if init_img is None:
        return processing.Processed(p, [], seed, "Empty image")
    init_img = images.flatten(init_img, shared.opts.img2img_background_color)

    # Override size by user choice
    if target_size_type == 1:
        p.width = custom_width
        p.height = custom_height
    elif target_size_type == 2:
        p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
        p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

    upscaler = usdu.USDUpscaler(p, init_img, upscaler_index, save_upscaled_image, save_seams_fix_image, tile_width, tile_height)
    upscaler.upscale()

    upscaler.setup_redraw(redraw_mode, padding, mask_blur)
    upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width, seams_fix_type)
    upscaler.print_info()
    upscaler.add_extra_info()
    upscaler.process()
    result_images = upscaler.result_images

    logger.debug("USDU run complete, batch_size=%s", p.batch_size)
    return processing.Processed(p, result_images, seed, upscaler.initial_info or "")


def patch_usdu_linear_and_chess_process():
    old_linear = usdu.USDURedraw.linear_process
    old_chess = usdu.USDURedraw.chess_process

    @wraps(old_linear)
    def new_linear_process(self, p, image, rows, cols):
        batch_size = getattr(p, 'batch_size', 1)

        if batch_size <= 1:
            return old_linear(self, p, image, rows, cols)

        # Batch mode
        mask_template, draw_template = self.init_draw(p, image.width, image.height)
        tiles_to_process: List[Tuple[int, int]] = []

        for yi in range(rows):
            for xi in range(cols):
                if shared.state.interrupted:
                    break

                tiles_to_process.append((xi, yi))

                if len(tiles_to_process) >= batch_size or (yi == rows - 1 and xi == cols - 1):
                    shared.batch = process_batch_tiles(p, tiles_to_process, shared.batch, self.calc_rectangle)
                    tiles_to_process = []

        p.width = image.width
        p.height = image.height
        return image

    @wraps(old_chess)
    def new_chess_process(self, p, image, rows, cols):
        batch_size = getattr(p, 'batch_size', 1)
        if batch_size <= 1:
            return old_chess(self, p, image, rows, cols)

        mask_template, draw_template = self.init_draw(p, image.width, image.height)

        # Determine tile "white/black" order
        tile_colors = []
        for yi in range(rows):
            row_colors = []
            for xi in range(cols):
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                row_colors.append(color)
            tile_colors.append(row_colors)

        # Helper to iterate tiles in chess order: white first, then black
        def chess_order_iter(white: bool):
            for yi in range(rows):
                for xi in range(cols):
                    if tile_colors[yi][xi] == white:
                        yield (xi, yi)

        # Process white tiles then black tiles
        for color in (True, False):
            tiles_to_process: List[Tuple[int, int]] = []
            for tx, ty in chess_order_iter(color):
                if shared.state.interrupted:
                    break
                tiles_to_process.append((tx, ty))
                if len(tiles_to_process) >= batch_size:
                    shared.batch = process_batch_tiles(p, tiles_to_process, shared.batch, self.calc_rectangle)
                    tiles_to_process = []
            if tiles_to_process:
                shared.batch = process_batch_tiles(p, tiles_to_process, shared.batch, self.calc_rectangle)

        p.width = image.width
        p.height = image.height
        return image

    usdu.USDURedraw.linear_process = new_linear_process
    usdu.USDURedraw.chess_process = new_chess_process


patch_usdu_upscaler_init()
patch_usdu_redraw_init()
patch_usdu_seams_fix_init()
patch_usdu_upscale_method()
usdu.Script.run = patched_script_run
patch_usdu_linear_and_chess_process()
