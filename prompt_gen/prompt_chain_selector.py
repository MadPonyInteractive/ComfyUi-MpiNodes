from ..help_funcs import pick_seeded_item


class MpiPromptChainSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_chain": (
                    "PROMPT_LST",
                    {
                        "forceInput": True,
                        "tooltip": "Selects and outputs only 1 list from a chain",
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            },
        }

    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ("PROMPT_LST", "STRING", "INT")
    RETURN_NAMES = ("prompt_list", "selected_list", "seed")
    FUNCTION = "choose_list"

    def choose_list(
        self,
        prompt_chain,
        seed,
    ):
        picked = pick_seeded_item(prompt_chain, seed)
        return ([picked], picked["title"], seed)
