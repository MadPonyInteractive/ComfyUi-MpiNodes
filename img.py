import torch  # type:ignore
from .help_funcs import aspect_ratio, create_mask_from_bbox


class MpiMaskDebugInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    CATEGORY = "MpiNodes/Debug"
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
                "max_dimension": (
                    "INT",
                    {
                        "default": 720,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "If 0, it will output the original dimmensions",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("scaled_width", "scaled_height", "is_portrait")
    CATEGORY = "MpiNodes/ImgOps"
    FUNCTION = "compute"

    def compute(self, image, max_dimension):
        B, H, W, C = image.shape
        is_portrait = H > W
        if not max_dimension:
            return (W, H, is_portrait)

        if is_portrait:
            scaled_height = max_dimension
            scaled_width = int((W / H) * max_dimension)
        else:
            scaled_width = max_dimension
            scaled_height = int((H / W) * max_dimension)

        return (scaled_width, scaled_height, is_portrait)


class MpiGetImageAtIndex:
    RETURN_TYPES = ("IMAGE", "INT")
    FUNCTION = "indexedimagesfrombatch"
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        """Returns the image at the specified index as an image batch."""
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "index": (
                    "INT",
                    {
                        "default": "-1",
                        "min": -99999999999,
                        "tooltip": "Negative values start counting from the last image.\n-1 would mean that you get the last image!",
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
