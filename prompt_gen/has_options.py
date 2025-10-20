class MpiHasOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "options_check": (
                    "STRING",
                    {
                        "multiline": False,
                        "tooltip": "The options to check if present in selected_options output from a 'Prompt List' (separated by commas)",
                    },
                ),
                "selected_options": (
                    "STRING",
                    {"default": "", "forceInput": True},
                ),
            }
        }

    CATEGORY = "MpiNodes/Logic"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "check"

    def check(self, selected_options: list, options_check: str):
        titles = [t.strip().lower() for t in options_check.split(",")]
        for option in selected_options:
            if option.lower() in titles:
                return (True,)

        return (False,)
