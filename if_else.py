from .help_funcs import AlwaysEqualProxy


class MpiIfElse:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "true",
                        "label_off": "false",
                    },
                ),
                "true": ("*", {"forceInput": True, "lazy": True}),
                "false": ("*", {"forceInput": True, "lazy": True}),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("output",)
    CATEGORY = "MpiNodes/Logic"
    FUNCTION = "doit"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def check_lazy_status(self, boolean: bool, true, false):
        if boolean:
            return ["true"]
        return ["false"]

    def doit(self, boolean: bool, true, false):
        if boolean:
            return (true,)
        return (false,)
