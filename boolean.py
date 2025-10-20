class MpiBoolean:
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

    CATEGORY = "MpiNodes/logic"
    RETURN_TYPES = ("BOOLEAN", "INT", "FLOAT")
    RETURN_NAMES = ("boolean", "int", "float")
    FUNCTION = "doit"

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
