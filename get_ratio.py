from .help_funcs import aspect_ratio


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

    CATEGORY = "MpiNodes/ImgImgOps"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ratio",)
    FUNCTION = "check"

    def check(self, width: int, height: int):
        return (aspect_ratio(width, height),)
