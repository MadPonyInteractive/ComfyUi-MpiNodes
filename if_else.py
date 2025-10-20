from comfy_execution.graph import ExecutionBlocker  # type: ignore
from .help_funcs import AlwaysEqualProxy, get_random_bool


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


class MpiIfElseInverted:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (AlwaysEqualProxy("*"), {}),
                "boolean": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "out_b",
                        "label_off": "out_a",
                    },
                ),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"), AlwaysEqualProxy("*"))
    RETURN_NAMES = ("out_a", "out_b")
    CATEGORY = "MpiNodes/Logic"
    FUNCTION = "doit"

    def doit(self, input, boolean: bool):
        if boolean:
            return (ExecutionBlocker(None), input)
        return (input, ExecutionBlocker(None))


class MpiIfElseProbability:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "true": ("*", {"forceInput": True, "lazy": True}),
                "false": ("*", {"forceInput": True, "lazy": True}),
                "probability": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0.0 will always output 'false' and return False \n1.0 will always output 'true' and return True\n0.5 gives both a 50% chance",
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            },
        }

    CATEGORY = "MpiNodes/Logic"
    RETURN_TYPES = (AlwaysEqualProxy("*"), "BOOLEAN", "INT")
    RETURN_NAMES = ("output", "boolean", "seed")
    FUNCTION = "doit"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def check_lazy_status(
        self,
        probability,
        seed,
        true,
        false,
    ):
        if get_random_bool(probability, seed):
            return ["true", "seed"]
        return ["false", "seed"]

    def doit(
        self,
        probability,
        seed,
        true,
        false,
    ):
        picked = get_random_bool(probability, seed)
        return (true if picked else false, picked, seed)


class MpiIfElseProbabilityInverted:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (AlwaysEqualProxy("*"), {}),
                "probability": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0.0 will always output 'false' and return False \n1.0 will always output 'true' and return True\n0.5 gives both a 50% chance",
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            },
        }

    CATEGORY = "MpiNodes/Logic"
    RETURN_TYPES = (
        AlwaysEqualProxy("*"),
        AlwaysEqualProxy("*"),
        "BOOLEAN",
        "INT",
    )
    RETURN_NAMES = ("true", "false", "boolean", "seed")
    FUNCTION = "doit"

    def doit(
        self,
        input,
        probability,
        seed,
    ):
        picked = get_random_bool(probability, seed)
        if picked:
            return (input, ExecutionBlocker(None), picked, seed)
        return (ExecutionBlocker(None), input, picked, seed)
