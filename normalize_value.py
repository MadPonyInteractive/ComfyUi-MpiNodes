class MpiNormalizeValue:
    CATEGORY = "MpiNodes/Math"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("normalized",)
    FUNCTION = "normalize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "*",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0},
                ),
                "input_min": ("FLOAT", {"default": 0.0}),
                "input_max": ("FLOAT", {"default": 100.0}),
                "output_min": ("FLOAT", {"default": 0.0}),
                "output_max": ("FLOAT", {"default": 1.0}),
                "clamp": ("BOOLEAN", {"default": True}),
                "mode": (["linear", "inverse", "log", "exp"],),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        if isinstance(input_types["value"], bool):
            input_types["value"] = int(input_types["value"])
        return True

    def normalize(
        self, value, input_min, input_max, output_min, output_max, clamp, mode
    ):
        if input_max == input_min:
            raise ValueError("Input range cannot be zero.")

        # Convert INT to FLOAT if needed
        value = float(value)

        # Normalize to 0–1
        normalized = (value - input_min) / (input_max - input_min)

        # Apply mode
        if mode == "inverse":
            normalized = 1.0 - normalized
        elif mode == "log":
            import math

            normalized = math.log1p(normalized * 9) / math.log1p(9)
        elif mode == "exp":
            normalized = normalized**2

        # Scale to output range
        scaled = output_min + normalized * (output_max - output_min)

        # Clamp if needed
        if clamp:
            scaled = max(min(scaled, output_max), output_min)

        return (scaled,)
