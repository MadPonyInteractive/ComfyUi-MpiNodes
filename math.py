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
