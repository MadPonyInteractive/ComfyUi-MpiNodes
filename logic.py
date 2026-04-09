import math
from .help_funcs import round_to_multiple


class MpiRoundToMultiple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "INT",
                    {"default": 1024, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "multiple_of": ("INT", {"default": 64, "min": 1, "max": 1024}),
                "round": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "up",
                        "label_off": "down",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("rounded_value",)
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Round value to nearest multiple"
    FUNCTION = "round_to_multiple"

    def round_to_multiple(self, value, multiple_of, round):
        return (round_to_multiple(value, multiple_of, round),)


class MpiRoundToMultipleRes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 1024, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "multiple_of": ("INT", {"default": 64, "min": 1, "max": 1024}),
                "round": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "up",
                        "label_off": "down",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Round width and height to nearest multiple"
    FUNCTION = "round_to_multiple"

    def round_to_multiple(self, width, height, multiple_of, round):
        return (
            round_to_multiple(width, multiple_of, round),
            round_to_multiple(height, multiple_of, round),
        )


class MpiCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("*", {"forceInput": True}),
                "b": ("*", {"forceInput": True}),
                "operator": (
                    ["==", "!=", ">", "<", ">=", "<="],
                    {
                        "default": "==",
                        "tooltip": "Choose the comparison operator",
                    },
                ),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    CATEGORY = "MpiNodes/Logic"
    FUNCTION = "compare"
    DESCRIPTION = "General logic operator for comparing values"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def compare(self, a, b, operator):
        if operator == "==":
            result = a == b
        elif operator == "!=":
            result = a != b
        elif operator == ">":
            result = a > b
        elif operator == "<":
            result = a < b
        elif operator == ">=":
            result = a >= b
        elif operator == "<=":
            result = a <= b
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        return (result,)


class MpiConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*", {"forceInput": True}),
                "round": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "up",
                        "label_off": "down",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN")
    CATEGORY = "MpiNodes/Logic"
    FUNCTION = "doit"
    DESCRIPTION = "Converts any value to the available outputs!"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def doit(self, input, round):
        if isinstance(input, str):
            input = float(input)
        out = math.ceil((input)) if round else math.floor(input)
        return (str(out), out, float(out), bool(out))


class MpiString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Pass through a string value"
    FUNCTION = "doit"

    def doit(self, string: str):
        return (string,)


class MpiText(MpiString):
    DESCRIPTION = "Pass through multiline text value"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"multiline": True}),
            },
        }

    RETURN_NAMES = ("Text",)


class MpiInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": (
                    "INT",
                    {
                        "default": 0,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Pass through an integer value"
    FUNCTION = "doit"

    def doit(self, int: int):
        return (int,)


class MpiFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "step": 0.01,
                        "max": 1e308,
                        "min": -1e308,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Pass through a float value"
    FUNCTION = "doit"

    def doit(self, float: float):
        return (float,)


class MpiBool:
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
            },
        }

    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Pass through a boolean value"
    FUNCTION = "doit"


class MpiSimpleBoolean(MpiBool):
    DESCRIPTION = "Pass through a boolean value"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)

    def doit(
        self,
        boolean,
    ):
        return (boolean,)


class MpiBoolean(MpiBool):
    DESCRIPTION = "Convert boolean to boolean, integer, and float values"
    RETURN_TYPES = ("BOOLEAN", "INT", "FLOAT")
    RETURN_NAMES = ("boolean", "int", "float")

    def doit(
        self,
        boolean,
    ):
        return (boolean, int(boolean), float(boolean))


class MpiBooleanCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "true",
                        "label_off": "false",
                        "forceInput": True,
                    },
                ),
                "b": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "true",
                        "label_off": "false",
                        "forceInput": True,
                    },
                ),
                "mode": (
                    [
                        "equal",  # a == b
                        "not_equal",  # a != b
                        "both_false",  # not a and not b
                        "both_true",  # a and b
                        "a_true_b_false",  # a and not b
                        "a_false_b_true",  # not a and b
                        "one_is_true",  # a or b
                        "not_both_true",  # not (a and b)
                    ],
                ),
            }
        }

    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Compare two boolean values with multiple logic operations"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "compare"

    def compare(self, a, b, mode):
        if mode == "equal":
            return (a == b,)
        elif mode == "both_false":
            return (not a and not b,)
        elif mode == "a_true_b_false":
            return (a and not b,)
        elif mode == "a_false_b_true":
            return (not a and b,)
        elif mode == "one_is_true":
            return (a or b,)
        elif mode == "both_true":
            return (a and b,)
        elif mode == "not_equal":
            return (a != b,)
        elif mode == "not_both_true":
            return (not (a and b),)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
