import math
from .help_funcs import AlwaysEqualProxy

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
        # Prepare evaluation context with math module and input variables
        eval_dict = {
            "a": a,
            "b": b,
            "c": c,
            "math": math,
        }
        
        # Add all math functions directly to the context for convenience
        for name in dir(math):
            if not name.startswith("__"):
                eval_dict[name] = getattr(math, name)

        try:
            # Evaluate the expression
            result = eval(math_expression, {"__builtins__": {}}, eval_dict)
            return (result,)
        except Exception as e:
            print(f"[MpiMath] Error evaluating expression '{math_expression}': {e}")
            return (0.0,)
