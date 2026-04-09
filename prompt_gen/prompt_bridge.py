class MpiPromptBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "prompt_list": ("PROMPT_LST", {"forceInput": True}),
                "seed": (
                    "INT",
                    {
                        "forceInput": True,
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
            },
        }

    CATEGORY = "MpiNodes/PromptGen"
    DESCRIPTION = "Pass through prompt list and seed values"
    RETURN_TYPES = ("PROMPT_LST", "INT")
    RETURN_NAMES = ("prompt_list", "seed")
    FUNCTION = "pass_it"

    def pass_it(self, prompt_list: list = [], seed: int = 0):
        return (prompt_list, seed)
