from .help_funcs import AlwaysEqualProxy


class MpiLogger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": (
                    "STRING",
                    {"default": "[LOG]", "placeholder": "prefix text"},
                ),
            },
            "optional": {
                "any": (AlwaysEqualProxy("*"), {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Log any input value to the console with a prefix"
    FUNCTION = "log_value"
    OUTPUT_NODE = True

    def log_value(self, prefix: str, any=None):
        print(f"{prefix} {any}")
        return ()