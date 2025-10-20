class MpiDummyToggleNode:
    CATEGORY = "MpiNodes/Dev"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_extra": ("BOOLEAN", {"default": False}),
                "extra_value": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    def run(self, enable_extra, extra_value):
        if enable_extra:
            return (f"Extra value used: {extra_value}",)
        else:
            return ("Extra value disabled",)

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True
