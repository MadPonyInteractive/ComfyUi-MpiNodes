import torch  # type:ignore
from .help_funcs import aspect_ratio, create_mask_from_bbox, round_to_multiple, crop_offset
import math
import os
import numpy as np  # type: ignore
from PIL import Image, ImageOps  # type: ignore
from comfy_execution.graph import ExecutionBlocker  # type: ignore
from nodes import PreviewImage  # type: ignore


def get_corrected_source_dimensions(
    width, height, horizontal_split, vertical_split, upscale_factor
):
    """
    Calculate source dimensions that will produce an upscaled image
    exactly divisible by the split count.
    """
    if horizontal_split <= 0 or vertical_split <= 0:
        raise ValueError("Splits must be greater than 0")

    upscaled_w = width * upscale_factor
    upscaled_h = height * upscale_factor

    # Calculate tile size (round up)
    tile_w = math.ceil(upscaled_w / horizontal_split)
    tile_h = math.ceil(upscaled_h / vertical_split)

    # Ensure tiles are divisible by upscale_factor for clean source dimensions
    tile_w = math.ceil(tile_w / upscale_factor) * upscale_factor
    tile_h = math.ceil(tile_h / upscale_factor) * upscale_factor

    return int(tile_w), int(tile_h)


class MpiGridDimensions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "horizontal_split": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                    },
                ),
                "vertical_split": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                    },
                ),
                "upscale_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "step": 0.1,
                        "min": 0.1,
                        "max": 10.0,
                    },
                ),
                "auto": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "tile_width",
        "tile_height",
        "horizontal_split",
        "vertical_split",
    )
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = "Calculate grid dimensions and corrected source size for perfect tiling"
    FUNCTION = "compute"

    def compute(
        self, image, horizontal_split, vertical_split, upscale_factor, auto
    ):
        _, H, W, _ = image.shape

        if auto:
            longer = max(W * upscale_factor, H * upscale_factor)
            if longer < 1024:
                max_tiles = 4
            elif longer < 2048:
                max_tiles = 6
            else:
                max_tiles = 9
            best_h, best_v = 1, 1
            best_score = float("inf")
            for h in range(1, 9):
                for v in range(1, 9):
                    if h * v > max_tiles:
                        continue
                    tile_w = W / h
                    tile_h = H / v
                    if tile_w > 768 or tile_h > 768:
                        continue
                    score = abs((tile_w / tile_h) - 1.0)
                    if score < best_score:
                        best_score = score
                        best_h, best_v = h, v
            horizontal_split, vertical_split = best_h, best_v

        tile_w, tile_h = get_corrected_source_dimensions(
            W, H, horizontal_split, vertical_split, upscale_factor
        )

        return (tile_w, tile_h, horizontal_split, vertical_split)


class MpiMaskDebugInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    CATEGORY = "MpiNodes/Debug"
    DESCRIPTION = "Print mask shape, data type, and device information"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("shape", "dtype", "device")
    FUNCTION = "debug"

    def debug(self, mask):
        shape_str = str(mask.shape)
        dtype_str = str(mask.dtype)
        device_str = str(mask.device)
        print("Mask Debug Info:")
        print("  Shape:", shape_str)
        print("  Dtype:", dtype_str)
        print("  Device:", device_str)
        return (shape_str, dtype_str, device_str)


class MpiBboxToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "bbox": ("BBOX", {}),
                "bbox_format": (["xyxy", "xywh"],),
            }
        }

    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = "Convert bounding boxes to mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "check"

    def check(self, images, bbox, bbox_format):
        B = images.shape[0]
        if len(bbox) != B:
            raise ValueError(f"Expected {B} bboxes, got {len(bbox)}")

        masks = []
        for i in range(B):
            mask = create_mask_from_bbox(
                images[i].unsqueeze(0),
                bbox[i],
                normalized=False,
                bbox_format=bbox_format,
            )
            masks.append(mask)

        masks = torch.cat(masks, dim=0)  # (B, H, W)
        # print("Final mask batch shape:", masks.shape)
        return (masks,)


class MpiAspectRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 4096,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 4096,
                    },
                ),
            }
        }

    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = "Calculate aspect ratio from width and height"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ratio",)
    FUNCTION = "check"

    def check(self, width: int, height: int):
        return (aspect_ratio(width, height),)


class MpiScaledDimensions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": (
                    "INT",
                    {
                        "default": 720,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "If 0, it will output the original dimensions",
                    },
                ),
                "side": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "use_max",
                        "label_off": "use_min",
                    },
                ),
                "upscale_method": (
                    ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"],
                    {
                        "default": "lanczos",
                        "tooltip": "Interpolation used to resize the scaled_image output.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "BOOLEAN", "IMAGE")
    RETURN_NAMES = ("scaled_width", "scaled_height", "is_portrait", "scaled_image")
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = "Scale image dimensions proportionally to target size, and output the image resized to those dimensions."
    FUNCTION = "compute"

    def compute(self, image, size, side, upscale_method="lanczos"):
        B, H, W, C = image.shape
        is_portrait = H > W

        if not size:
            return (W, H, is_portrait, image)

        target_dim = max(H, W) if side else min(H, W)
        scale = size / target_dim
        scaled_width = int(W * scale)
        scaled_height = int(H * scale)

        import comfy.utils  # type: ignore

        # common_upscale wants NCHW; IMAGE tensors are NHWC.
        samples = image.movedim(-1, 1)
        samples = comfy.utils.common_upscale(
            samples, scaled_width, scaled_height, upscale_method, "disabled"
        )
        scaled_image = samples.movedim(1, -1)

        return (scaled_width, scaled_height, is_portrait, scaled_image)


class MpiGetImageAtIndex:
    RETURN_TYPES = ("IMAGE", "INT")
    FUNCTION = "indexedimagesfrombatch"
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = """Returns the image at the specified index as an image batch.
Negative values start counting from the last image.
-1 would mean that you get the last image!"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "index": (
                    "INT",
                    {
                        "default": "-1",
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
            },
        }

    def indexedimagesfrombatch(self, images, index):
        # Empty batch: nothing to index. Block downstream instead of IndexError.
        if images.shape[0] == 0:
            return (ExecutionBlocker(None), index)

        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor([index], dtype=torch.long)

        # Select the images at the specified indices
        chosen_images = images[indices_tensor]

        return (chosen_images, index)


def load_image_from_path(path, channel="alpha"):
    """Load an image file into a ComfyUI IMAGE tensor + MASK, the same way
    ComfyUI's built-in LoadImage does (EXIF-transpose, RGB, /255, alpha->mask).
    channel selects the mask source: alpha (inverted, like LoadImage), or one of
    red/green/blue taken straight from the RGB image."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode == "I":
        img = img.point(lambda i: i * (1 / 255))
    rgb = img.convert("RGB")
    arr = np.array(rgb).astype(np.float32) / 255.0
    image = torch.from_numpy(arr)[None,]
    if channel == "alpha":
        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((rgb.height, rgb.width), dtype=torch.float32)
    else:
        idx = {"red": 0, "green": 1, "blue": 2}[channel]
        mask = torch.from_numpy(arr[..., idx])
    return image, mask.unsqueeze(0)


class MpiLoadImageFromPath(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Image file path. Named 'string' so it matches MpiString / MpiAnyChecker outputs.",
                    },
                ),
                "channel": (
                    ["alpha", "red", "green", "blue"],
                    {
                        "default": "alpha",
                        "tooltip": "Which channel to output as the mask. 'alpha' inverts the alpha channel like LoadImage; red/green/blue take that color channel directly.",
                    },
                ),
                "block_if_empty": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "ON: empty/missing path blocks downstream execution. OFF: outputs a blank 1x1 image so the graph continues.",
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    OUTPUT_NODE = True
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "Load an image from a file path and preview it in-graph. Also outputs "
        "width and height. If the path is empty/missing, downstream execution is "
        "blocked (no need for a separate blocker node) unless block_if_empty is "
        "off, in which case it outputs a blank 1x1 image so the graph continues."
    )
    FUNCTION = "load"

    def load(self, string, channel="alpha", block_if_empty=True, prompt=None, extra_pnginfo=None):
        path = (string or "").strip()
        if not path or not os.path.isfile(path):
            if block_if_empty:
                blocked = ExecutionBlocker(None)
                return {"ui": {"images": []}, "result": (blocked, blocked, 0, 0)}
            image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            mask = torch.zeros((1, 1, 1), dtype=torch.float32)
            return {"ui": {"images": []}, "result": (image, mask, 0, 0)}

        image, mask = load_image_from_path(path, channel)
        _, h, w, _ = image.shape
        # Reuse PreviewImage.save_images for the in-graph thumbnail.
        preview = self.save_images(image, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return {"ui": preview["ui"], "result": (image, mask, w, h)}


def read_upscale_model_scale(upscale_model, fallback):
    """Read native scale from a spandrel ImageModelDescriptor (the UPSCALE_MODEL
    object from ComfyUI's UpscaleModelLoader). Uses fallback when metadata is
    absent. Raises on a non-positive result to keep callers safe from
    division-by-zero downstream."""
    scale = getattr(upscale_model, "scale", None)
    if scale is None:
        scale = fallback
    scale = int(scale)
    if scale <= 0:
        raise ValueError(
            f"Upscale model scale must be positive, got {scale}. "
            "Set a valid fallback_scale (>= 1)."
        )
    return scale


class MpiUpscaleModelScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "fallback_scale": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Used only if the model exposes no .scale metadata",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("scale_int", "scale_float")
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "Read an upscale model's native scale (1x/2x/4x/8x) from its descriptor "
        "metadata instead of parsing the filename. Outputs INT and FLOAT."
    )
    FUNCTION = "get_scale"

    def get_scale(self, upscale_model, fallback_scale):
        scale = read_upscale_model_scale(upscale_model, fallback_scale)
        return (scale, float(scale))


class MpiCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "tooltip": "Target crop width. 0 = keep full width (only divisible-by trim applies).",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "tooltip": "Target crop height. 0 = keep full height (only divisible-by trim applies).",
                    },
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 256,
                        "tooltip": "Final crop size is floored to a multiple of this.",
                    },
                ),
                "position": (["center", "left", "right", "top", "bottom"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "Crop an image to width/height at a chosen anchor. width/height of 0 "
        "keep that dimension full. The final crop is floored to a multiple of "
        "divisible_by, so with width=height=0 it just trims to a divisible size."
    )
    FUNCTION = "crop"

    def crop(self, image, width, height, divisible_by, position):
        _, H, W, _ = image.shape

        target_w = W if width <= 0 else min(width, W)
        target_h = H if height <= 0 else min(height, H)

        target_w = round_to_multiple(target_w, divisible_by, round=False)
        target_h = round_to_multiple(target_h, divisible_by, round=False)

        x = crop_offset(W, target_w, position, "x")
        y = crop_offset(H, target_h, position, "y")

        cropped = image[:, y:y + target_h, x:x + target_w, :]
        return (cropped, target_w, target_h)


def clamp_box(box, W, H):
    """Intersect an (x, y, w, h) MPI_BOX with a WxH image.
    Returns (x, y, w, h) clamped inside the image; w/h may come back 0 if the
    box lies fully outside."""
    x, y, w, h = box
    x0 = max(0, min(int(x), W))
    y0 = max(0, min(int(y), H))
    x1 = max(0, min(int(x) + int(w), W))
    y1 = max(0, min(int(y) + int(h), H))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


class MpiBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 0, "max": 0xFFFFFFFF}),
                "height": ("INT", {"default": 512, "min": 0, "max": 0xFFFFFFFF}),
                # x/y accept NEGATIVE values: a box may start left of / above the
                # image so it can sit tight on a subject at the frame edge without
                # growing until it swallows the neighbour. Widening a widget range
                # cannot break a saved workflow - every stored value is still legal.
                "x": ("INT", {"default": 0, "min": -0xFFFFFFFF, "max": 0xFFFFFFFF,
                              "tooltip": "Left edge of the box, in pixels. May be "
                                         "negative to start outside the image."}),
                "y": ("INT", {"default": 0, "min": -0xFFFFFFFF, "max": 0xFFFFFFFF,
                              "tooltip": "Top edge of the box, in pixels. May be "
                                         "negative to start outside the image."}),
            }
        }

    RETURN_TYPES = ("MPI_BOX",)
    RETURN_NAMES = ("mpi_box",)
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "Build an MPI_BOX rectangle from width/height/x/y. x/y are the top-left "
        "corner. Consumers clamp the box to the image, so out-of-bounds is safe."
    )
    FUNCTION = "doit"

    def doit(self, width, height, x, y):
        return ((x, y, width, height),)


class MpiFromBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mpi_box": ("MPI_BOX", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "x", "y")
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = "Unpack an MPI_BOX back into width, height, x, y integers."
    FUNCTION = "doit"

    def doit(self, mpi_box):
        x, y, w, h = mpi_box
        return (w, h, x, y)


class MpiBoxCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mpi_box": ("MPI_BOX", {"forceInput": True}),
            },
            "optional": {
                "pad": ("BOOLEAN", {"default": False,
                                    "tooltip": "Pad the crop back out to the "
                                               "requested box size when the box "
                                               "overhangs the image, replicating "
                                               "the edge pixels. Off, the crop is "
                                               "just the intersection."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MPI_BOX")
    RETURN_NAMES = ("image", "mpi_box")
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "Crop an image to an MPI_BOX region. The box is clamped to the image; "
        "a box fully outside the image passes the image through unchanged. "
        "Also outputs the clamped box, so downstream nodes see the region "
        "actually used. With pad on, an overhanging box is padded back out to "
        "its requested size by edge replication, so the crop keeps the aspect "
        "that was asked for - which is what a square reference crop needs."
    )
    FUNCTION = "crop"

    def crop(self, image, mpi_box, pad=False):
        _, H, W, _ = image.shape
        x, y, w, h = clamp_box(mpi_box, W, H)
        # ponytail: empty intersection passes through rather than erroring mid-graph
        if w == 0 or h == 0:
            return (image, (x, y, 0, 0))
        cropped = image[:, y:y + h, x:x + w, :]
        if not pad:
            return (cropped, (x, y, w, h))

        # Pad the intersection back out to the REQUESTED box. Only the sides that
        # were actually trimmed grow, so a box fully inside the image is untouched
        # and this stays a no-op for every existing graph.
        rx, ry, rw, rh = (int(v) for v in mpi_box)
        left, top = x - rx, y - ry
        right, bottom = (rx + rw) - (x + w), (ry + rh) - (y + h)
        left, top = max(0, left), max(0, top)
        right, bottom = max(0, right), max(0, bottom)
        if not (left or top or right or bottom):
            return (cropped, (x, y, w, h))
        # F.pad wants [N, C, H, W] and pads the last two dims; the tensor here is
        # [B, H, W, C]. 'replicate' extends the edge pixels, which reads as the
        # subject continuing rather than a hard border the model will try to draw.
        chw = cropped.permute(0, 3, 1, 2)
        padded = torch.nn.functional.pad(chw, (left, right, top, bottom), mode="replicate")
        return (padded.permute(0, 2, 3, 1), (rx, ry, rw, rh))


class MpiBoxMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mpi_box": ("MPI_BOX", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("MASK", "MPI_BOX")
    RETURN_NAMES = ("mask", "mpi_box")
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "Build a mask the size of the image, black except for a white rectangle "
        "at the MPI_BOX region. The box is clamped to the image. Also outputs "
        "the clamped box, so downstream nodes see the region actually drawn."
    )
    FUNCTION = "make_mask"

    def make_mask(self, image, mpi_box):
        B, H, W, _ = image.shape
        x, y, w, h = clamp_box(mpi_box, W, H)
        mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)
        if w and h:
            mask[:, y:y + h, x:x + w] = 1.0
        return (mask, (x, y, w, h))


def square_bbox_from_mask(mask, padding=0):
    """Tight bbox of the mask's nonzero pixels, expanded to a centered square
    that is clamped (and shrunk if needed) to stay fully inside the image.
    mask: (H, W) tensor. Returns (x, y, size) ints, or None if mask is empty."""
    H, W = mask.shape
    ys, xs = torch.where(mask > 0)
    if ys.numel() == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max()) + 1  # exclusive
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(W, x_max + padding)
    y_max = min(H, y_max + padding)

    # Square side = longer edge of the tight box, but never bigger than the image.
    side = max(x_max - x_min, y_max - y_min)
    side = min(side, W, H)

    # Center the square on the tight box's center, then clamp into the image.
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    x = int(round(cx - side / 2))
    y = int(round(cy - side / 2))
    x = max(0, min(x, W - side))
    y = max(0, min(y, H - side))

    return x, y, side


class MpiMaskSquareBbox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "Pixels added around the tight mask box before squaring.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("square_mask", "x", "y", "size")
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "Square bounding box around a mask, centered on the mask and clamped "
        "(shrunk if needed) to stay fully inside the image. Outputs a filled "
        "square MASK plus its x, y, and side length."
    )
    FUNCTION = "compute"

    def compute(self, mask, padding):
        # MASK is (B, H, W); operate on the first item, keep the batch dim out.
        single = mask[0] if mask.dim() == 3 else mask
        H, W = single.shape

        box = square_bbox_from_mask(single, padding)
        out = torch.zeros((1, H, W), dtype=mask.dtype, device=mask.device)
        if box is None:
            return (out, 0, 0, 0)

        x, y, size = box
        out[:, y:y + size, x:x + size] = 1.0
        return (out, x, y, size)


def _box_blur(img, radius):
    """(H, W, C) box blur via avg_pool2d with reflect padding. radius 0 = no-op."""
    if radius < 1:
        return img
    k = 2 * radius + 1
    x = img.permute(2, 0, 1).unsqueeze(0)               # 1,C,H,W
    x = torch.nn.functional.pad(x, (radius,) * 4, mode="reflect")
    x = torch.nn.functional.avg_pool2d(x, k, stride=1)
    return x.squeeze(0).permute(1, 2, 0)


def _gaussian_feather(mask, radius):
    """Blur a (H, W) 0/1 mask into a soft blend weight. Separable box passes
    approximate a gaussian closely enough for feathering and stay cheap."""
    if radius < 1:
        return mask
    w = mask.unsqueeze(-1)
    for _ in range(3):                                   # 3 box passes ~ gaussian
        w = _box_blur(w, max(1, radius // 3))
    return w.squeeze(-1).clamp(0, 1)


def _ring_of(mask, ring_px):
    """Band of untouched pixels hugging the mask: dilate, then subtract the mask."""
    if ring_px < 1:
        return torch.zeros_like(mask, dtype=torch.bool)
    k = 2 * ring_px + 1
    x = mask.view(1, 1, *mask.shape)
    grown = torch.nn.functional.max_pool2d(x, k, stride=1, padding=ring_px)
    return (grown.view(*mask.shape) > 0.5) & (mask <= 0.5)


def heal_region(image, mask, color_strength, grain_strength, ring_px, feather_px, radius=1):
    """Match an inpainted region's colour and grain to the real pixels around it.

    Generative fills drift: their colour statistics wander from the surrounding
    pixels (invisible on foliage, obvious on skin) and their fine grain lands
    well under the real thing. Both are corrected against a RING of untouched
    pixels hugging the mask -- never a rectangular crop, which would still
    contain the object being removed and drag the correction toward it.

    image: (H, W, C) float 0-1. mask: (H, W) float 0-1. Returns (H, W, C).
    """
    m = mask > 0.5
    ring = _ring_of(mask, ring_px)
    if not m.any() or not ring.any():
        return image                                     # nothing to match against

    out = image.clone()
    fill, real = image[m], image[ring]

    if color_strength:
        delta = real.mean(dim=0) - fill.mean(dim=0)
        out[m] = out[m] + delta * color_strength

    if grain_strength:
        # Scale the fill's OWN high-frequency residual up to the ring's amplitude.
        # Amplifying real detail can't repeat or invent texture the way tiling
        # donor patches would.
        hf = out - _box_blur(out, radius)
        f_std = hf[m].std(dim=0)
        r_std = hf[ring].std(dim=0)
        gain = torch.where(f_std > 1e-6, r_std / f_std.clamp(min=1e-6),
                           torch.ones_like(f_std))
        gain = 1.0 + (gain.clamp(1.0, 4.0) - 1.0) * grain_strength  # only ever add
        out[m] = out[m] + hf[m] * (gain - 1.0)

    if feather_px:
        # Feather the correction itself, or its own edge becomes a new seam.
        w = _gaussian_feather(mask, feather_px).unsqueeze(-1)
        out = image * (1 - w) + out * w

    return out.clamp(0, 1)


class MpiInpaintHeal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "color_strength": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "tooltip": "Shift the filled region's colour toward the "
                                   "surrounding ring. 0 disables.",
                    },
                ),
                "grain_strength": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "tooltip": "Lift the filled region's fine grain toward the "
                                   "surrounding ring. 0 disables.",
                    },
                ),
                "ring_px": (
                    "INT",
                    {
                        "default": 31, "min": 1, "max": 512,
                        "tooltip": "Width of the untouched band sampled as the "
                                   "reference. Wide enough for a stable average, "
                                   "narrow enough to stay locally lit.",
                    },
                ),
                "feather_px": (
                    "INT",
                    {
                        "default": 15, "min": 0, "max": 256,
                        "tooltip": "Softens the correction's edge so it does not "
                                   "become a visible seam of its own.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "Heal an inpainted region against the real pixels around it. Corrects "
        "colour drift and restores fine grain by matching a ring of untouched "
        "pixels hugging the mask -- unlike a colour-match on a rectangular crop, "
        "which still contains the removed object and pulls the fill toward it. "
        "Run AFTER stitching, with the mask of the region that was regenerated."
    )
    FUNCTION = "heal"

    def heal(self, image, mask, color_strength, grain_strength, ring_px, feather_px):
        B = image.shape[0]
        H, W = image.shape[1:3]
        out = []
        for i in range(B):
            m = mask[i] if mask.dim() == 3 and mask.shape[0] > i else mask.reshape(-1, *mask.shape[-2:])[0]
            if m.shape != (H, W):
                m = torch.nn.functional.interpolate(
                    m.view(1, 1, *m.shape), size=(H, W), mode="bilinear", align_corners=False
                ).view(H, W)
            out.append(heal_region(image[i], m.to(image.device),
                                   color_strength, grain_strength, ring_px, feather_px))
        return (torch.stack(out),)


class MpiImageSplice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The full clip. Returned unchanged except where the patch is written."}),
                "patch": ("IMAGE", {"tooltip": "The frames to write in, e.g. an inpainted window."}),
                "start": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "tooltip": "The clip frame the patch's FIRST frame is written at - the same index the window was cut at."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "Write a run of frames back into a longer clip at a given frame. The return "
        "leg of windowed work: cut a window out, run the expensive thing on it alone, "
        "splice the result back. A mismatched canvas or a patch running past the end "
        "raises, because either one otherwise lands as a clip that plays but is wrong."
    )
    FUNCTION = "doit"

    def doit(self, images, patch, start):
        if patch.shape[1:] != images.shape[1:]:
            raise ValueError(
                f"The patch is {patch.shape[2]}x{patch.shape[1]} and the clip is "
                f"{images.shape[2]}x{images.shape[1]}; a splice needs one canvas."
            )
        if start + patch.shape[0] > images.shape[0]:
            raise ValueError(
                f"A {patch.shape[0]}-frame patch written at frame {start} runs past "
                f"the end of a {images.shape[0]}-frame clip."
            )
        out = images.clone()
        out[start:start + patch.shape[0]] = patch.to(out)
        return (out,)


if __name__ == "__main__":
    # square_bbox_from_mask: tight box -> centered square, clamped to image
    m = torch.zeros((100, 200))
    m[40:60, 90:110] = 1.0            # 20x20 mask centered at (100, 50)
    x, y, s = square_bbox_from_mask(m)
    assert s == 20, s
    assert (x, y) == (90, 40), (x, y)  # already square, unchanged

    m2 = torch.zeros((100, 200))
    m2[10:90, 95:105] = 1.0           # tall 80x10 -> square side 80, centered
    x, y, s = square_bbox_from_mask(m2)
    assert s == 80, s
    assert y == 10, y                  # fits vertically
    assert x == 60, x                  # centered on x=100, clamped inside 200

    m3 = torch.zeros((50, 200))
    m3[5:45, 90:180] = 1.0            # 40 tall, 90 wide, but image is 50 tall
    x, y, s = square_bbox_from_mask(m3)
    assert s == 50, s                  # shrunk to image height
    assert y == 0 and 0 <= x <= 150    # clamped fully inside

    assert square_bbox_from_mask(torch.zeros((10, 10))) is None
    print("square_bbox ok")

    # crop_offset anchoring
    assert crop_offset(100, 40, "left", "x") == 0
    assert crop_offset(100, 40, "right", "x") == 60
    assert crop_offset(100, 40, "center", "x") == 30
    assert crop_offset(100, 40, "top", "x") == 30      # top irrelevant on x -> center
    assert crop_offset(100, 40, "top", "y") == 0
    assert crop_offset(100, 40, "bottom", "y") == 60
    # divisible-by trim floors size
    assert round_to_multiple(101, 8, round=False) == 96

    class _M:
        def __init__(self, s):
            self.scale = s
    assert read_upscale_model_scale(_M(4), 1) == 4          # reads metadata
    assert read_upscale_model_scale(object(), 2) == 2       # fallback when missing
    try:
        read_upscale_model_scale(_M(0), 0)                  # non-positive -> raise
        assert False, "expected ValueError"
    except ValueError:
        pass

    # heal_region: a dull, colour-shifted patch inside grainy surroundings
    torch.manual_seed(0)
    base = torch.full((80, 80, 3), 0.5) + torch.randn(80, 80, 3) * 0.04
    hmask = torch.zeros((80, 80))
    hmask[30:50, 30:50] = 1.0
    hm = hmask > 0.5
    patch = torch.full((20, 20, 3), 0.5) + torch.randn(20, 20, 3) * 0.01   # 4x less grain
    patch[:, :, 2] -= 0.06                                                 # blue drift
    broken = base.clone()
    broken[30:50, 30:50] = patch

    hring = _ring_of(hmask, 15)
    before = (broken[hm].mean(dim=0) - broken[hring].mean(dim=0)).abs().mean()
    fixed = heal_region(broken, hmask, 1.0, 1.0, ring_px=15, feather_px=0)
    after = (fixed[hm].mean(dim=0) - fixed[hring].mean(dim=0)).abs().mean()
    assert after < before / 5, (before.item(), after.item())      # colour converges

    def _grain(img, sel):
        return (img - _box_blur(img, 1))[sel].std().item()
    assert _grain(fixed, hm) > _grain(broken, hm) * 1.5           # grain lifted
    assert _grain(fixed, hm) <= _grain(base, hring) * 1.3         # but not overshot
    # strengths at 0 and an empty mask must both be no-ops
    assert torch.equal(heal_region(broken, hmask, 0.0, 0.0, 15, 0), broken)
    assert torch.equal(heal_region(broken, torch.zeros((80, 80)), 1.0, 1.0, 15, 0), broken)
    # untouched pixels must never move
    assert torch.allclose(fixed[~hm], broken[~hm], atol=1e-6)
    print("heal_region ok")

    # --- splice: the window goes back exactly where it was cut from ------
    clip = torch.zeros(10, 4, 4, 3)
    win = torch.ones(3, 4, 4, 3)
    spliced = MpiImageSplice().doit(clip, win, 5)[0]
    assert spliced[5:8].min() == 1.0, "the patch must land at start"
    assert spliced[:5].max() == 0.0 and spliced[8:].max() == 0.0, "nothing else moves"
    assert clip.max() == 0.0, "the input clip must not be mutated in place"
    for _bad, _at in ((torch.ones(3, 2, 2, 3), 5), (torch.ones(6, 4, 4, 3), 5)):
        try:
            MpiImageSplice().doit(clip, _bad, _at)
        except ValueError:
            pass
        else:
            raise AssertionError("a mismatched splice must raise")
    print("splice ok")

    print("ok")
