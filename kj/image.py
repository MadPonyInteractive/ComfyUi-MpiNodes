"""Image, batch and mask nodes ported from ComfyUI-KJNodes for Cubric Studio (MPI-1341).

Ported from comfyui-kjnodes (nodes/image_nodes.py, nodes/mask_nodes.py), GPL-3.0, Copyright (c)
kijai. See LICENSE-KJNodes. Each node keeps the original's input names and accepted values so a
workflow only has to change its class_type. Only the code paths Cubric Studio reaches are carried:

  * MpiColorMatch      - method "mkl" only (color-matcher is not needed; see color_match.py).
  * MpiImageResize     - every keep_proportion mode, the CPU/GPU switch and the five torch
                         upscale methods; NVIDIA RTX VSR (needs the nvvfx library) and the
                         sub-batch progress text are not carried.
  * MpiImageConcat     - IMAGE only (the original also took MASK); all four directions.
  * MpiImageBatchExtend - overlap modes cut / linear_blend / ease_in_out / filmic_crossfade;
                         perceptual_crossfade (needs kornia) is not carried.
  * MpiGrowMaskBlur    - the 3x3 erosion/dilation is plain torch, kornia is not needed.
  * MpiImageRange, MpiImageBatchMulti - as the originals.

Needs only torch, numpy, scipy and pillow (all engine-owned).
"""
import math
from concurrent.futures import ThreadPoolExecutor
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage
from PIL import ImageColor, ImageFilter, Image

from comfy import model_management  # type: ignore
from comfy.utils import ProgressBar, common_upscale  # type: ignore
from nodes import MAX_RESOLUTION  # type: ignore

from .color_match import mkl_transfer

CATEGORY = "MpiNodes/KJ"


# ---------------------------------------------------------------- colour match

class MpiColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (["mkl"], {"default": "mkl"}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "multithread": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = ("Matches the colours of image_target to image_ref with the Monge-Kantorovich linear mapping. "
                   "A one-image reference is used for every target frame; a longer reference pairs frame by frame.")

    def colormatch(self, image_ref, image_target, method, strength=1.0, multithread=True):
        if strength == 0:
            return (image_target,)

        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        ref_np = image_ref.numpy()
        target_np = image_target.numpy()

        def process(i):
            tgt = target_np[i]
            ref = ref_np[0] if ref_np.shape[0] == 1 else ref_np[i]
            result = mkl_transfer(tgt, ref)
            if strength != 1:
                # the target without its alpha when the reference had none, like the transfer itself
                base = tgt[..., :result.shape[2]]
                result = base + strength * (result - base)
            return torch.from_numpy(result)

        if multithread and batch_size > 1:
            with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, batch_size)) as executor:
                out = list(executor.map(process, range(batch_size)))
        else:
            out = [process(i) for i in range(batch_size)]

        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)


# -------------------------------------------------------------------- batches

class MpiImageRange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_index": ("INT", {"default": 0, "min": -1, "max": 4096, "step": 1}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            },
        }

    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "imagesfrombatch"
    DESCRIPTION = "Returns a range of images (and masks) from a batch. start_index -1 takes the last num_frames."

    def imagesfrombatch(self, start_index, num_frames, images=None, masks=None):
        chosen_images = None
        chosen_masks = None
        if images is not None:
            if start_index == -1:
                start_index = max(0, len(images) - num_frames)
            if start_index < 0 or start_index >= len(images):
                raise ValueError("Start index is out of range")
            chosen_images = images[start_index:min(start_index + num_frames, len(images))]
        if masks is not None:
            if start_index == -1:
                start_index = max(0, len(masks) - num_frames)
            if start_index < 0 or start_index >= len(masks):
                raise ValueError("Start index is out of range for masks")
            chosen_masks = masks[start_index:min(start_index + num_frames, len(masks))]
        return (chosen_images, chosen_masks)


class MpiImageBatchExtend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE", {"tooltip": "The source images to extend"}),
                "overlap": ("INT", {"default": 13, "min": 1, "max": 4096, "step": 1,
                                    "tooltip": "Number of overlapping frames between source and new images"}),
                "overlap_side": (["source", "new_images"], {"default": "source", "tooltip": "Which side to overlap on"}),
                "overlap_mode": (["cut", "linear_blend", "ease_in_out", "filmic_crossfade"],
                                 {"default": "linear_blend", "tooltip": "Method to use for overlapping frames"}),
            },
            "optional": {
                "new_images": ("IMAGE", {"tooltip": "The new images to extend with"}),
            },
        }

    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("source_images", "start_images", "extended_images")
    OUTPUT_TOOLTIPS = (
        "The original source images (passthrough)",
        "The input images used as the starting point for extension",
        "The extended images with overlap, if no new images are provided this will be empty",
    )
    FUNCTION = "imagesfrombatch"
    DESCRIPTION = ("Helper for video extension. First give the source and the overlap to get the starting frames; "
                   "on a second copy give the newly generated frames and choose how to overlap them.")

    def imagesfrombatch(self, source_images, overlap, overlap_side, overlap_mode, new_images=None):
        if overlap > len(source_images):
            return source_images, source_images, source_images

        if new_images is not None:
            if source_images.shape[1:3] != new_images.shape[1:3]:
                raise ValueError(f"Source and new images must have the same shape: "
                                 f"{source_images.shape[1:3]} vs {new_images.shape[1:3]}")
            prefix = source_images[:-overlap]
            if overlap_side == "source":
                blend_src = source_images[-overlap:]
                blend_dst = new_images[:overlap]
            else:  # new_images
                blend_src = new_images[:overlap]
                blend_dst = source_images[-overlap:]
            suffix = new_images[overlap:]

            def ramp():
                a = torch.linspace(0, 1, overlap + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
                return a.view(-1, 1, 1, 1)

            if overlap_mode == "linear_blend":
                alpha = ramp()
                blended = (1 - alpha) * blend_src + alpha * blend_dst
                extended_images = torch.cat((prefix, blended, suffix), dim=0)
            elif overlap_mode == "filmic_crossfade":
                gamma = 2.2
                alpha = ramp()
                linear = (1 - alpha) * torch.pow(blend_src, gamma) + alpha * torch.pow(blend_dst, gamma)
                extended_images = torch.cat((prefix, torch.pow(linear, 1.0 / gamma), suffix), dim=0)
            elif overlap_mode == "ease_in_out":
                t = ramp()
                eased = 3 * t * t - 2 * t * t * t
                blended = (1 - eased) * blend_src + eased * blend_dst
                extended_images = torch.cat((prefix, blended, suffix), dim=0)
            elif overlap_mode == "cut":
                if overlap_side == "new_images":
                    extended_images = torch.cat((source_images, new_images[overlap:]), dim=0)
                else:
                    extended_images = torch.cat((source_images[:-overlap], new_images), dim=0)
            else:
                raise ValueError(f"Unknown overlap_mode {overlap_mode!r}")
        else:
            extended_images = torch.zeros((1, 64, 64, 3), device="cpu")

        start_images = source_images[-overlap:]
        return (source_images, start_images, extended_images)


class MpiImageBatchMulti:
    """image_1 and image_2 are declared; further image_N inputs are read from kwargs, as the
    original did (its dynamic-input script is a frontend nicety this port does not ship)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "image_1": ("IMAGE",),
            },
            "optional": {
                "image_2": ("IMAGE",),
            },
        }

    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    DESCRIPTION = "Creates an image batch from multiple images. A missing input is a black stand-in of the first image's length."

    def combine(self, inputcount, **kwargs):
        first = kwargs["image_1"]
        h, w = first.shape[1], first.shape[2]

        max_ch = first.shape[-1]
        total_frames = first.shape[0]
        for c in range(1, inputcount):
            img = kwargs.get(f"image_{c + 1}")
            if img is not None:
                max_ch = max(max_ch, img.shape[-1])
                total_frames += img.shape[0]
            else:
                total_frames += first.shape[0]

        out = torch.empty((total_frames, h, w, max_ch), dtype=first.dtype)
        offset = 0
        for c in range(inputcount):
            img = kwargs.get(f"image_{c + 1}")
            if img is None:
                img = torch.zeros((first.shape[0], h, w, max_ch), dtype=first.dtype)
            if img.shape[1:3] != (h, w):
                img = common_upscale(img.movedim(-1, 1), w, h, "bilinear", "center").movedim(1, -1)
            if img.shape[-1] < max_ch:
                img = F.pad(img, (0, max_ch - img.shape[-1]), mode="constant", value=1.0)
            n = img.shape[0]
            out[offset:offset + n].copy_(img)
            offset += n
        return (out.cpu(),)


# --------------------------------------------------------------- concatenate

class MpiImageConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "direction": (["right", "down", "left", "up"], {"default": "right"}),
                "match_image_size": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    DESCRIPTION = ("Concatenates image2 to image1 in the given direction. With match_image_size image2 is resized "
                   "(bicubic) to image1's height (left/right) or width (up/down); without it the smaller image "
                   "is centred and the rest filled with zeros.")

    def execute(self, image1, image2, direction, match_image_size):
        return (self.concatenate(image1, image2, direction, match_image_size),)

    @staticmethod
    def concatenate(image1, image2, direction, match_image_size):
        bs1, bs2 = image1.shape[0], image2.shape[0]
        B = max(bs1, bs2)
        H1, W1 = image1.shape[1], image1.shape[2]
        C1, C2 = image1.shape[-1], image2.shape[-1]
        out_C = max(C1, C2)

        if match_image_size:
            orig_aspect = image2.shape[2] / image2.shape[1]
            if direction in ("left", "right"):
                H2 = image1.shape[1]
                W2 = int(H2 * orig_aspect)
            else:
                W2 = image1.shape[2]
                H2 = int(W2 / orig_aspect)
        else:
            H2, W2 = image2.shape[1], image2.shape[2]

        if direction in ("right", "left"):
            out_H, out_W = max(H1, H2), W1 + W2
        else:
            out_H, out_W = H1 + H2, max(W1, W2)

        if direction == "right":
            i1_y, i1_x, i2_y, i2_x = (out_H - H1) // 2, 0, (out_H - H2) // 2, W1
        elif direction == "left":
            i1_y, i1_x, i2_y, i2_x = (out_H - H1) // 2, W2, (out_H - H2) // 2, 0
        elif direction == "down":
            i1_y, i1_x, i2_y, i2_x = 0, (out_W - W1) // 2, H1, (out_W - W2) // 2
        else:  # up
            i1_y, i1_x, i2_y, i2_x = H2, (out_W - W1) // 2, 0, (out_W - W2) // 2

        output = torch.zeros(
            (B, out_H, out_W, out_C),
            dtype=model_management.intermediate_dtype(),
            device=model_management.intermediate_device(),
        )

        def write(dst, src, src_C):
            if dst.shape[-1] == src_C:
                dst.copy_(src)
            else:
                dst[..., :src_C].copy_(src)
                dst[..., src_C:].fill_(1.0)

        slot1 = output[:, i1_y:i1_y + H1, i1_x:i1_x + W1, :]
        if bs1 == B:
            write(slot1, image1, C1)
        else:
            write(slot1[:bs1], image1, C1)
            write(slot1[bs1:], image1[-1:].expand(B - bs1, -1, -1, -1), C1)

        slot2 = output[:, i2_y:i2_y + H2, i2_x:i2_x + W2, :]
        if match_image_size:
            pbar = ProgressBar(B)
            device = model_management.get_torch_device()
            for i in range(B):
                src_i = min(i, bs2 - 1)
                frame = image2[src_i:src_i + 1].to(device).permute(0, 3, 1, 2)
                resized = F.interpolate(frame, size=(H2, W2), mode="bicubic", antialias=True).permute(0, 2, 3, 1)
                write(slot2[i:i + 1], resized, C2)
                pbar.update(1)
        else:
            if bs2 == B:
                write(slot2, image2, C2)
            else:
                write(slot2[:bs2], image2, C2)
                write(slot2[bs2:], image2[-1:].expand(B - bs2, -1, -1, -1), C2)
        return output


# --------------------------------------------------------------------- resize

def string_to_color(color_string):
    """'0, 0, 0' / '1.0, 0.5, 0.0' / '#RRGGBB' / 'red' / '128' -> [r, g, b(, a)] in 0..255.
    An unreadable string is black, with a warning (the original's behaviour)."""
    color_list = [0, 0, 0]
    if "," in color_string:
        try:
            values = [float(channel.strip()) for channel in color_string.split(",")]
            if all(0 <= v <= 1 for v in values):
                color_list = [int(v * 255) for v in values]
            else:
                color_list = [int(v) for v in values]
        except ValueError:
            logging.warning(f"Invalid color format: {color_string}. Using default black.")
    elif color_string.startswith("#") or (color_string.lstrip("#").isalnum()
                                          and not color_string.lstrip("#").replace(".", "", 1).isdigit()):
        stripped = color_string.lstrip("#")
        if len(stripped) in (6, 8) and all(c in "0123456789ABCDEFabcdef" for c in stripped):
            color_list = [int(stripped[i:i + 2], 16) for i in range(0, len(stripped), 2)]
        else:
            try:
                color_list = list(ImageColor.getrgb(color_string))
            except ValueError:
                logging.warning(f"Invalid color name or hex format: {color_string}. Using default black.")
    else:
        try:
            value = float(color_string.strip())
            value = int(value * 255) if 0 <= value <= 1 else int(value)
            color_list = [value, value, value]
        except ValueError:
            logging.warning(f"Invalid color format: {color_string}. Using default black.")
    return np.clip(color_list, 0, 255).tolist()


def _gaussian_blur_nchw(img, sigma_px):
    if sigma_px <= 0:
        return img
    radius = max(1, int(3.0 * float(sigma_px)))
    k = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    k1 = torch.exp(-(x * x) / (2.0 * float(sigma_px) * float(sigma_px)))
    k1 = k1 / k1.sum()
    c = img.shape[1]
    kx = k1.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    ky = k1.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    img = F.conv2d(img, kx, padding=(0, radius), groups=c)
    return F.conv2d(img, ky, padding=(radius, 0), groups=c)


def pad_image(image, pad_left, pad_right, pad_top, pad_bottom, color, pad_mode, mask=None):
    """The original ImagePadKJ.pad for the arguments ImageResizeKJv2 gives it (no extra padding, no
    target size). pad_mode: edge | edge_pixel | color | pillarbox_blur. Returns (image, mask)."""
    B, H, W, C = image.shape
    if mask is not None:
        BM, HM, WM = mask.shape
        if HM != H or WM != W:
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode="nearest-exact").squeeze(1)

    bg_color = [x / 255.0 for x in string_to_color(color)]
    if len(bg_color) == 1:
        bg_color = bg_color * 3
    bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)

    padded_width = W + pad_left + pad_right
    padded_height = H + pad_top + pad_bottom

    if pad_mode == "pillarbox_blur":
        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
            scale_fill = max(padded_width / float(W), padded_height / float(H)) if (W > 0 and H > 0) else 1.0
            bg_w = max(1, int(round(W * scale_fill)))
            bg_h = max(1, int(round(H * scale_fill)))
            src_b = image[b].movedim(-1, 0).unsqueeze(0)
            bg = common_upscale(src_b, bg_w, bg_h, "bilinear", crop="disabled")
            y0 = max(0, (bg_h - padded_height) // 2)
            x0 = max(0, (bg_w - padded_width) // 2)
            bg = bg[:, :, y0:min(bg_h, y0 + padded_height), x0:min(bg_w, x0 + padded_width)]
            if bg.shape[2] != padded_height or bg.shape[3] != padded_width:
                pad_h = padded_height - bg.shape[2]
                pad_w = padded_width - bg.shape[3]
                top_fix = max(0, pad_h // 2)
                left_fix = max(0, pad_w // 2)
                bg = F.pad(bg, (left_fix, max(0, pad_w - left_fix), top_fix, max(0, pad_h - top_fix)), mode="replicate")
            sigma = max(1.0, 0.006 * float(min(padded_height, padded_width)))
            bg = _gaussian_blur_nchw(bg, sigma_px=sigma)
            if C >= 3:
                r, g, bch = bg[:, 0:1], bg[:, 1:2], bg[:, 2:3]
                luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                gray = torch.cat([luma, luma, luma], dim=1)
                desat = 0.20
                bg[:, 0:3, :, :] = torch.cat([r, g, bch], dim=1) * (1.0 - desat) + gray * desat
            bg = torch.clamp(bg * 0.35, 0.0, 1.0)
            out_image[b] = bg.squeeze(0).movedim(0, -1)
        out_image[:, pad_top:pad_top + H, pad_left:pad_left + W, :] = image
        out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
        out_masks[:, pad_top:pad_top + H, pad_left:pad_left + W] = mask if mask is not None else 0.0
        return out_image, out_masks

    if pad_mode == "edge_pixel":
        # replicating the border pixel into the pad, corners included
        out_image = F.pad(image.movedim(-1, 1), (pad_left, pad_right, pad_top, pad_bottom), mode="replicate").movedim(1, -1)
    else:
        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
            if pad_mode == "edge":
                # the MEAN colour of each edge
                out_image[b, :pad_top, :, :] = image[b, 0, :, :].mean(dim=0)
                out_image[b, pad_top + H:, :, :] = image[b, H - 1, :, :].mean(dim=0)
                out_image[b, :, :pad_left, :] = image[b, :, 0, :].mean(dim=0)
                out_image[b, :, pad_left + W:, :] = image[b, :, W - 1, :].mean(dim=0)
                out_image[b, pad_top:pad_top + H, pad_left:pad_left + W, :] = image[b]
            else:  # color
                out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                out_image[b, pad_top:pad_top + H, pad_left:pad_left + W, :] = image[b]

    if mask is not None:
        out_masks = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    else:
        out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
        out_masks[:, pad_top:pad_top + H, pad_left:pad_left + W] = 0.0
    return out_image, out_masks


class MpiImageResize:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (cls.upscale_methods,),
                "keep_proportion": (["stretch", "resize", "pad", "pad_edge", "pad_edge_pixel", "crop",
                                     "pillarbox_blur", "total_pixels"], {"default": "stretch"}),
                "pad_color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color to use for padding."}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
                "device": (["cpu", "gpu"],),
            },
        }

    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("IMAGE", "width", "height", "mask")
    FUNCTION = "resize"
    DESCRIPTION = ("Resizes the image to width x height. keep_proportion keeps the aspect ratio by the highest "
                   "dimension (resize), pads the rest (pad*), crops to fill (crop) or targets a pixel count.")

    def resize(self, image, width, height, upscale_method, keep_proportion, pad_color, crop_position,
               divisible_by, device="cpu", mask=None):
        B, H, W, C = image.shape

        # ComfyUI's 64x64 placeholder mask means "no mask"
        if mask is not None and mask.shape[-2:] == (64, 64) and (H != 64 or W != 64):
            mask = None
        if mask is not None and mask.shape[-2:] != (H, W):
            mask = common_upscale(mask.unsqueeze(1), W, H, "bilinear", crop="disabled").squeeze(1)

        if device == "gpu":
            if upscale_method == "lanczos":
                raise ValueError("Lanczos is not supported on the GPU")
            device = model_management.get_torch_device()
        else:
            device = torch.device("cpu")

        pillarbox_blur = keep_proportion == "pillarbox_blur"
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion in ("resize", "total_pixels") or keep_proportion.startswith("pad") or pillarbox_blur:
            if keep_proportion == "total_pixels":
                total_pixels = width * height
                aspect_ratio = W / H
                new_height = int(math.sqrt(total_pixels / aspect_ratio))
                new_width = int(math.sqrt(total_pixels * aspect_ratio))
            elif width == 0 and height == 0:
                new_width, new_height = W, H
            elif width == 0:
                new_width = round(W * (height / H))
                new_height = height
            elif height == 0:
                new_width = width
                new_height = round(H * (width / W))
            else:
                ratio = min(width / W, height / H)
                new_width = round(W * ratio)
                new_height = round(H * ratio)

            if keep_proportion.startswith("pad") or pillarbox_blur:
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

            width, height = new_width, new_height
        else:
            if width == 0:
                width = W
            if height == 0:
                height = H

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        out_image = image if image.device == device else image.to(device)
        out_mask = None if mask is None else (mask if mask.device == device else mask.to(device))

        if keep_proportion == "crop":
            old_height = out_image.shape[-3]
            old_width = out_image.shape[-2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            if old_aspect > new_aspect:
                crop_w = round(old_height * new_aspect)
                crop_h = old_height
            else:
                crop_w = old_width
                crop_h = round(old_width / new_aspect)
            if crop_position == "center":
                x = (old_width - crop_w) // 2
                y = (old_height - crop_h) // 2
            elif crop_position == "top":
                x = (old_width - crop_w) // 2
                y = 0
            elif crop_position == "bottom":
                x = (old_width - crop_w) // 2
                y = old_height - crop_h
            elif crop_position == "left":
                x = 0
                y = (old_height - crop_h) // 2
            else:  # right
                x = old_width - crop_w
                y = (old_height - crop_h) // 2
            out_image = out_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)
            if out_mask is not None:
                out_mask = out_mask.narrow(-1, x, crop_w).narrow(-2, y, crop_h)

        out_image = common_upscale(out_image.movedim(-1, 1), width, height, upscale_method, crop="disabled").movedim(1, -1)
        if out_mask is not None:
            if upscale_method == "lanczos":
                out_mask = common_upscale(out_mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method,
                                          crop="disabled").movedim(1, -1)[:, :, :, 0]
            else:
                out_mask = common_upscale(out_mask.unsqueeze(1), width, height, upscale_method, crop="disabled").squeeze(1)

        if (keep_proportion.startswith("pad") or pillarbox_blur) and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
            padded_width = width + pad_left + pad_right
            padded_height = height + pad_top + pad_bottom
            if divisible_by > 1:
                if padded_width % divisible_by > 0:
                    pad_right += divisible_by - padded_width % divisible_by
                if padded_height % divisible_by > 0:
                    pad_bottom += divisible_by - padded_height % divisible_by
            pad_mode = ("pillarbox_blur" if pillarbox_blur else
                        "edge" if keep_proportion == "pad_edge" else
                        "edge_pixel" if keep_proportion == "pad_edge_pixel" else
                        "color")
            out_image, out_mask = pad_image(out_image, pad_left, pad_right, pad_top, pad_bottom, pad_color, pad_mode, mask=out_mask)

        out_image = out_image.cpu()
        out_mask = out_mask.cpu() if out_mask is not None else torch.zeros(1, 64, 64, device=torch.device("cpu"), dtype=torch.float32)
        return (out_image, out_image.shape[2], out_image.shape[1], out_mask)


# ------------------------------------------------------------------- mask grow

def _dilate3x3(x, tapered):
    """Grey dilation of an (N, 1, H, W) tensor with a 3x3 plus (tapered) or square footprint. Outside
    the image counts as -inf, the same as kornia's geodesic border."""
    if not tapered:
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    p = F.pad(x, (1, 1, 1, 1), mode="constant", value=float("-inf"))
    h, w = x.shape[-2:]
    out = x
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        out = torch.maximum(out, p[..., dy:dy + h, dx:dx + w])
    return out


class MpiGrowMaskBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "incremental_expandrate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1}),
                "lerp_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = CATEGORY
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask", "mask_inverted")
    FUNCTION = "expand_mask"
    DESCRIPTION = """# Grow Mask With Blur
- mask: input mask or mask batch
- expand: grow (positive) or shrink (negative) the mask by this many pixels
- incremental_expandrate: raise the expand amount by this much per frame
- tapered_corners: plus-shaped instead of square growth
- flip_input: invert the input first
- blur_radius: above 0 blurs the mask (8-bit result)
- lerp_alpha / decay_factor: blend each frame with the previous one
- fill_holes: fill holes in the mask (slow)"""

    def expand_mask(self, mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate,
                    lerp_alpha, decay_factor, fill_holes=False):
        main_device = model_management.get_torch_device()
        alpha = lerp_alpha
        decay = decay_factor
        if flip_input:
            mask = 1.0 - mask

        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        previous_output = None
        current_expand = expand
        for m in growmask:
            output = m.unsqueeze(0).unsqueeze(0).to(main_device)
            if abs(round(current_expand)) > 0 and output.max() > 0:
                for _ in range(abs(round(current_expand))):
                    if current_expand < 0:
                        output = -_dilate3x3(-output, tapered_corners)  # erosion
                    else:
                        output = _dilate3x3(output, tapered_corners)

            output = output.squeeze(0).squeeze(0)

            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)

            if fill_holes:
                filled = scipy.ndimage.binary_fill_holes((output > 0).cpu().numpy())
                output = torch.from_numpy(filled.astype(np.float32)).to(output.device)

            if alpha < 1.0 and previous_output is not None:
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output.cpu())

        if blur_radius != 0:
            for idx, tensor in enumerate(out):
                arr = np.clip(255.0 * tensor.detach().numpy().squeeze(), 0, 255).astype(np.uint8)
                blurred = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(blur_radius))
                out[idx] = torch.from_numpy(np.array(blurred).astype(np.float32) / 255.0).unsqueeze(0)
            blurred = torch.cat(out, dim=0)
            return (blurred, 1.0 - blurred)
        stacked = torch.stack(out, dim=0)
        return (stacked, 1.0 - stacked)


NODE_CLASS_MAPPINGS = {
    "MpiColorMatch": MpiColorMatch,
    "MpiImageRange": MpiImageRange,
    "MpiImageBatchExtend": MpiImageBatchExtend,
    "MpiImageBatchMulti": MpiImageBatchMulti,
    "MpiImageConcat": MpiImageConcat,
    "MpiImageResize": MpiImageResize,
    "MpiGrowMaskBlur": MpiGrowMaskBlur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiColorMatch": "Mpi Color Match",
    "MpiImageRange": "Mpi Image Range From Batch",
    "MpiImageBatchExtend": "Mpi Image Batch Extend With Overlap",
    "MpiImageBatchMulti": "Mpi Image Batch Multi",
    "MpiImageConcat": "Mpi Image Concatenate",
    "MpiImageResize": "Mpi Image Resize",
    "MpiGrowMaskBlur": "Mpi Grow Mask With Blur",
}
