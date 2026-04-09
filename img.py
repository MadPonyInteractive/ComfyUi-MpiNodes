import torch  # type:ignore
from .help_funcs import aspect_ratio, create_mask_from_bbox, round_to_multiple
import math


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
            }
        }

    RETURN_TYPES = ("INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("scaled_width", "scaled_height", "is_portrait")
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = "Scale image dimensions proportionally to target size"
    FUNCTION = "compute"

    def compute(self, image, size, side):
        B, H, W, C = image.shape
        is_portrait = H > W

        if not size:
            return (W, H, is_portrait)

        target_dim = max(H, W) if side else min(H, W)
        scale = size / target_dim
        scaled_width = int(W * scale)
        scaled_height = int(H * scale)

        return (scaled_width, scaled_height, is_portrait)


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
        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor([index], dtype=torch.long)

        # Select the images at the specified indices
        chosen_images = images[indices_tensor]

        return (chosen_images, index)
