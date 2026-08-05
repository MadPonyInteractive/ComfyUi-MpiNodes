import math
import re
from comfy_execution.graph import ExecutionBlocker  # type: ignore
from .help_funcs import round_to_multiple, is_empty_value, AlwaysEqualProxy


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


def _coerce(value: float, like):
    """Convert the b_value float widget to the same type as `like`.

    bool -> 0 is False, anything else True. int -> truncated. float -> as is.
    Anything else raises: the widget can only express a number, so comparing
    against a string, dict, list or tensor has to go through the b input.
    """
    if isinstance(like, bool):
        return value != 0
    if isinstance(like, (int, float)):
        return type(like)(value)
    raise TypeError(
        f"MpiCompare: b_value is a number, so it cannot be compared against a "
        f"{type(like).__name__}. Connect the b input instead."
    )


class MpiCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("*", {"forceInput": True}),
                "operator": (
                    ["==", "!=", ">", "<", ">=", "<="],
                    {
                        "default": "==",
                        "tooltip": "Choose the comparison operator",
                    },
                ),
            },
            "optional": {
                "b": ("*", {"forceInput": True}),
                "b_value": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1e9,
                        "max": 1e9,
                        "step": 0.01,
                        "tooltip": "Compared against when b is not connected.\nConverted to the type of a: for a boolean 0 is false and anything else true, for an int it is truncated.\nTo compare against text, connect the b input instead.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    CATEGORY = "MpiNodes/Logic"
    FUNCTION = "compare"
    DESCRIPTION = "General logic operator for comparing values. Compares a against the b input, or against the b_value widget when b is not connected"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def compare(self, a, operator, b=None, b_value=0.0):
        if b is None:
            b = _coerce(b_value, a)

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


class MpiBooleanInvert(MpiBool):
    @classmethod
    def INPUT_TYPES(cls):
        # No widget: a NOT gate has nothing to set by hand, so the input is
        # wire-only. Same forceInput shape MpiBooleanCompare uses.
        return {
            "required": {
                "boolean": ("BOOLEAN", {"forceInput": True}),
            },
        }

    DESCRIPTION = "Invert a boolean: true in, false out. Pass-through NOT gate."
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("inverted",)

    def doit(
        self,
        boolean,
    ):
        return (not boolean,)


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


class MpiListCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("*", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT", "BOOLEAN")
    RETURN_NAMES = ("count", "has_items")
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Count entries in a list of any type. Outputs count and a boolean true if list is non-empty."
    FUNCTION = "count"
    INPUT_IS_LIST = True

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def count(self, list_input):
        n = len(list_input)
        return (n, n > 0)


class MpiReroute:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any",)
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Pass any value through unchanged. Rename the node title for a labelled reroute."
    FUNCTION = "doit"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def doit(self, any):
        return (any,)


class MpiAnyBlocker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any",)
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Pass any value through, but block execution downstream if it is empty (empty string/list/dict/None). 0, 0.0 and False are not empty and pass through."
    FUNCTION = "doit"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def doit(self, any):
        if is_empty_value(any):
            return (ExecutionBlocker(None),)
        return (any,)


class MpiIsListEmpty:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("*", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT")
    RETURN_NAMES = ("is_empty", "count")
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Check whether a list of any type is empty. Outputs is_empty (true if zero items) and the item count."
    FUNCTION = "doit"
    INPUT_IS_LIST = True

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def doit(self, list_input):
        n = len(list_input)
        return (n == 0, n)


class MpiBlocker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (AlwaysEqualProxy("*"), {"lazy": True}),
                "boolean": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "continue",
                        "label_off": "block",
                    },
                ),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("output",)
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Manual gate: pass input through when the switch is on (continue), block the branch when off (block). Blocking skips the work FEEDING this node too - the input is lazy, so when the gate is off nothing upstream of it runs either. A one-output if/else."
    FUNCTION = "doit"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def check_lazy_status(self, boolean: bool, input=None):
        # The gate decides from `boolean` alone, so when it is off the input is
        # never needed — don't ask for it, and everything upstream is skipped.
        # ExecutionBlocker only travels downstream; laziness is the only thing
        # that stops the work feeding a blocked branch from running anyway.
        return ["input"] if boolean else []

    def doit(self, input, boolean: bool):
        if boolean:
            return (input,)
        return (ExecutionBlocker(None),)


class MpiBlockIfEmptyList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("list_output",)
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Pass a list through, but block downstream execution if the list is empty (zero items). Place before nodes that index into a list so an empty list halts the branch instead of throwing IndexError. Unlike Mpi Any Blocker, this runs on the whole list, so a truly empty list still reaches it."
    FUNCTION = "doit"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def doit(self, list_input):
        # INPUT_IS_LIST means an empty upstream list arrives as [] and this node
        # still runs — a per-item node would just be skipped, blocking nothing.
        if len(list_input) == 0:
            return ([ExecutionBlocker(None)],)
        return (list_input,)


class MpiAnyChecker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("*", "BOOLEAN")
    RETURN_NAMES = ("any", "has_value")
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Pass any value through unchanged and output has_value: true if it holds something, false if empty (empty string/list/dict/None/zero-element tensor/empty audio). 0, 0.0 and False count as having a value."
    FUNCTION = "doit"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def doit(self, any):
        return (any, not is_empty_value(any))


class MpiSeedPassthrough:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Set control_after_generate to 'randomize' for a fresh seed each run.",
                    },
                ),
            },
            "optional": {
                "any": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("*", "INT")
    RETURN_NAMES = ("any", "seed")
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = (
        "Pass any value through and emit a seed. Forces the workflow to "
        "re-run every time (via IS_CHANGED) so seed-less workflows don't get "
        "stuck on cached outputs. Leave 'any' unconnected to use as a pure "
        "seed source."
    )
    FUNCTION = "doit"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    @classmethod
    def IS_CHANGED(cls, seed, any=None):
        # Return the seed so ComfyUI sees the node as dirty whenever the seed
        # changes; with control_after_generate=randomize that's every run.
        return seed

    def doit(self, seed, any=None):
        return (any, seed)


class MpiConditioningReroute:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"forceInput": True}),
                "negative": ("CONDITIONING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Pass positive and negative conditioning through unchanged. A labelled conditioning reroute."
    FUNCTION = "doit"

    def doit(self, positive, negative):
        return (positive, negative)


class MpiListRange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("*", {"forceInput": True}),
                "start": (
                    "INT",
                    {
                        "default": 0,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Start index (inclusive). Negative counts from end.",
                    },
                ),
                "end": (
                    "INT",
                    {
                        "default": -1,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "End index (inclusive). Negative counts from end. -1 = last item.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("*", "INT")
    RETURN_NAMES = ("list", "count")
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Output a sub-range of any list. Inclusive start/end, negative indices count from end."
    FUNCTION = "slice_range"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False)

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def slice_range(self, list_input, start, end):
        import torch  # type: ignore

        s = start[0] if isinstance(start, list) else start
        e = end[0] if isinstance(end, list) else end

        # Image batch arrives as a single-element list wrapping a tensor
        # of shape [B, H, W, C]. Slice along the batch dim instead of the
        # Python list, otherwise we'd always return the whole batch.
        if (
            len(list_input) == 1
            and isinstance(list_input[0], torch.Tensor)
            and list_input[0].ndim >= 3
        ):
            batch = list_input[0]
            n = batch.shape[0]
            if n == 0:
                return ([batch], 0)
            if s < 0:
                s = n + s
            if e < 0:
                e = n + e
            s = max(0, min(s, n - 1))
            e = max(0, min(e, n - 1))
            if e < s:
                return ([batch[0:0]], 0)
            sliced = batch[s : e + 1]
            return ([sliced], sliced.shape[0])

        n = len(list_input)
        if n == 0:
            return ([], 0)

        if s < 0:
            s = n + s
        if e < 0:
            e = n + e

        s = max(0, min(s, n - 1))
        e = max(0, min(e, n - 1))

        if e < s:
            return ([], 0)

        sliced = list_input[s : e + 1]
        return (sliced, len(sliced))


class MpiTextContains:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "words": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Words or phrases to look for inside text, separated by commas or new lines. Case-insensitive, whole words only ('cat' matches 'cat' and 'cats' but not 'catalogue'). Phrases ignore spacing, so 'bird in a tree' matches across line breaks. Irregular plurals (child/children) need their own entry.",
                    },
                ),
            }
        }

    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Check if any comma separated whole word is present in the input text. Use as a whitelist/blacklist check."
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "contains"

    def contains(self, text: str, words: str):
        # Multiline field, so a new line separates entries just like a comma.
        needles = [w.strip() for w in re.split(r"[,\n]", words) if w.strip()]
        if not needles:
            return (False,)

        # Any run of whitespace matches any other, so a wrapped or double-spaced
        # prompt still matches a multi-word entry like 'bird in a tree'.
        def esc(phrase: str):
            return r"\s+".join(re.escape(t) for t in phrase.split())

        parts = []
        for w in needles:
            # Also match the regular plural: cat/cats, box/boxes, baby/babies.
            # ponytail: rule-based, so irregulars (child/children, knife/knives)
            # still need their own entry in the list.
            if re.search(r"[b-df-hj-np-tv-z]y$", w, re.IGNORECASE):
                stem = esc(w[:-1]) + "(?:y|ies)"
            elif re.search(r"(?:s|x|z|ch|sh)$", w, re.IGNORECASE):
                stem = esc(w) + "(?:es)?"
            else:
                stem = esc(w) + "s?"
            # \b keeps 'cat' from matching 'catalogue' while still matching
            # ',cat' / 'cat,' / 'a cat sits'.
            parts.append(rf"\b{stem}\b")

        return (re.search("|".join(parts), text, re.IGNORECASE) is not None,)
