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
    DESCRIPTION = "Probabilistically select between two prompt lists"
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
