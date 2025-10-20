from ..help_funcs import merge_unique_by_key


class MpiPromptListJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_list": (
                    "PROMPT_LST",
                    {
                        "forceInput": True,
                        "tooltip": "Adds a chain or list to an existing chain",
                    },
                ),
                "added": (
                    "PROMPT_LST",
                    {
                        "forceInput": True,
                        "tooltip": "The prompt chain or list to add",
                    },
                ),
            },
        }

    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ("PROMPT_LST",)
    RETURN_NAMES = ("prompt_list",)
    FUNCTION = "join_lists"

    def join_lists(
        self,
        prompt_list,
        added,
    ):
        if added in prompt_list:
            return (prompt_list,)
        new_list = [i for i in prompt_list]
        new_list = merge_unique_by_key(new_list, added)
        return (new_list,)
