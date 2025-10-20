from ..help_funcs import get_blacklist


class MpiDisplayPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_list": (
                    "PROMPT_LST",
                    {
                        "forceInput": True,
                        "tooltip": "Converts available lists to a displayable format for debugging",
                    },
                ),
            }
        }

    CATEGORY = "MpiNodes/Debug"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("titles", "blocks", "processed_prompts")
    FUNCTION = "display"

    def display(self, prompt_list=[]):

        titles = ""
        processed_prompts = ""
        for prompt in prompt_list:
            if prompt.get("positive"):
                processed_prompts += prompt["title"] + ", "
            else:
                titles += prompt["title"] + ", "
        if titles.endswith(", "):
            titles = titles[:-2]
        if processed_prompts.endswith(", "):
            processed_prompts = processed_prompts[:-2]

        blacklist = get_blacklist(prompt_list)
        blocks = ",".join(blacklist)

        return (titles, blocks, processed_prompts)
