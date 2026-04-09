class MpiPromptOverride:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "options": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "When set, overrides the prompt list",
                    },
                ),
                "seed_override": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "If not -1 it overrides the current seed changing the resulting random selection \nDoes not affect the return seed of the prompt list!",
                    },
                ),
            },
        }

    CATEGORY = "MpiNodes/PromptGen"
    DESCRIPTION = "Override prompt list options with custom text and optional seed"
    RETURN_TYPES = ("PROMPT_OVR",)
    RETURN_NAMES = ("override",)
    FUNCTION = "doit"

    def doit(self, options: str, seed_override: int):
        return ((options, seed_override),)


class MpiPromptOverrideSeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_override": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "If not -1 it overrides the current seed changing the resulting random selection \nDoes not affect the return seed of the prompt list!",
                    },
                ),
            },
        }

    CATEGORY = "MpiNodes/PromptGen"
    DESCRIPTION = "Override seed for prompt list randomization"
    RETURN_TYPES = ("PROMPT_OVR",)
    RETURN_NAMES = ("override",)
    FUNCTION = "doit"

    def doit(self, seed_override: int):
        return (("", seed_override),)
