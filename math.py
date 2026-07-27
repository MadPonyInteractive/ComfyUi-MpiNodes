from .help_funcs import AlwaysEqualProxy, safe_math

class MpiMath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("*", {"forceInput": True}),
                "math_expression": (
                    "STRING",
                    {"multiline": True, "default": "a + b + c"},
                ),
            },
            "optional": {
                "b": ("*", {"forceInput": True}),
                "c": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("result",)
    FUNCTION = "doit"
    CATEGORY = "MpiNodes/Math"
    DESCRIPTION = "Simple math node that takes inputs (a, b, c) and evaluates a python expression. b and c are optional."

    def doit(self, a, math_expression, b=0, c=0):
        try:
            result = safe_math(math_expression, {"a": a, "b": b, "c": c})
            return (result,)
        except Exception as e:
            print(f"[MpiMath] Error evaluating expression '{math_expression}': {e}")
            return (0.0,)


class MpiExpoFloat:
    CATEGORY = "MpiNodes/Math"
    DESCRIPTION = "Apply a power curve to a 0–1 float. Output = value ^ exponent"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            }
        }

    def apply(self, value, exponent):
        result = float(value) ** float(exponent)
        return (max(0.0, min(1.0, result)),)


class MpiClamp:
    CATEGORY = "MpiNodes/Math"
    DESCRIPTION = (
        "Clamp a number between a minimum and a maximum. With min 200 / max 1000: "
        "400 stays 400, 2000 becomes 1000, 100 becomes 200. INT in, INT out. "
        "Leave min at -1000000000 to only clamp the maximum."
    )
    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("value",)
    FUNCTION = "clamp"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*", {"forceInput": True}),
                "min_value": ("FLOAT", {"default": -1e9, "min": -1e9, "max": 1e9, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1000.0, "min": -1e9, "max": 1e9, "step": 0.01}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def clamp(self, value, min_value, max_value):
        # ponytail: min wins if the two are crossed over
        result = max(float(min_value), min(float(value), float(max_value)))
        # ponytail: mirror the input type so INT wiring stays INT
        return (int(result) if isinstance(value, int) else result,)
