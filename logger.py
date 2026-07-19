from .help_funcs import AlwaysEqualProxy


class MpiLogger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["value", "message"], {"default": "value"}),
                "prefix": (
                    "STRING",
                    {"default": "[LOG]", "placeholder": "prefix text"},
                ),
                "message": (
                    "STRING",
                    {"default": "", "placeholder": "message to log"},
                ),
            },
            "optional": {
                "any": (AlwaysEqualProxy("*"), {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = (
        "End-of-chain logger. Prints when the input arrives. mode='value' prints "
        "the input value with a prefix; mode='message' prints only the message "
        "(logs that this point was reached without dumping the input). "
        "Has no output — wire it at the end of a chain."
    )
    FUNCTION = "log_value"
    OUTPUT_NODE = True

    def log_value(self, mode: str = "value", prefix: str = "[LOG]", message: str = "", any=None):
        print(f"{prefix} {message if mode == 'message' else any}")
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