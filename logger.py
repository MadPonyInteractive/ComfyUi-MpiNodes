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


class MpiExecLogger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (AlwaysEqualProxy("*"), {"forceInput": True}),
                "prefix": (
                    "STRING",
                    {"default": "[LOG]", "placeholder": "prefix text"},
                ),
                "message": (
                    "STRING",
                    {"default": "", "placeholder": "message to log"},
                ),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("any",)
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = (
        "Pass-through any input to output while logging a message to the console. "
        "Wire it inline to print progress as a workflow runs."
    )
    FUNCTION = "log_and_pass"

    def log_and_pass(self, any, prefix: str, message: str):
        print(f"{prefix} {message}")
        return (any,)