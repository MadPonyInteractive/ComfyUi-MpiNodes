from ..help_funcs import get_random_bool


class MpiPromptListSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_list_a": ("PROMPT_LST", {"forceInput": True}),
                "probability": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0.0 will always output 'list a' and return False \n1.0 will always output 'list b' and return True\n0.5 gives both a 50% chance",
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            },
            "optional": {
                "prompt_list_b": ("PROMPT_LST", {"forceInput": True}),
            },
        }

    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ("PROMPT_LST", "BOOLEAN", "INT")
    RETURN_NAMES = ("prompt_list", "boolean", "seed")
    FUNCTION = "join_prompt_lists"

    def join_prompt_lists(
        self,
        prompt_list_a,
        probability,
        seed,
        prompt_list_b=[],
    ):
        picked = get_random_bool(probability, seed)
        rsl = []
        if prompt_list_b:
            rsl = prompt_list_b if picked else prompt_list_a
        else:
            rsl = [] if picked else prompt_list_a

        return (rsl, picked, seed)


class MpiPromptListSelectorBool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_list_a": ("PROMPT_LST", {"forceInput": True}),
                "boolean": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "list_b",
                        "label_off": "list_a",
                    },
                ),
            },
            "optional": {
                "prompt_list_b": ("PROMPT_LST", {"forceInput": True}),
            },
        }

    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ("PROMPT_LST", "BOOLEAN")
    RETURN_NAMES = ("prompt_list", "boolean")
    FUNCTION = "join_prompt_lists"

    def join_prompt_lists(
        self,
        prompt_list_a,
        boolean,
        prompt_list_b=[],
    ):
        rsl = []
        if prompt_list_b:
            rsl = prompt_list_b if boolean else prompt_list_a
        else:
            rsl = [] if boolean else prompt_list_a

        return (rsl, boolean)


class MpiPromptListSelectorBoolInverted:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_list": ("PROMPT_LST", {"forceInput": True}),
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

    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ("PROMPT_LST", "PROMPT_LST")
    RETURN_NAMES = ("out_a", "out_b")
    FUNCTION = "join_prompt_lists"

    def join_prompt_lists(
        self,
        prompt_list,
        boolean,
    ):
        if boolean:
            return ([], prompt_list)
        else:
            return (prompt_list, [])
