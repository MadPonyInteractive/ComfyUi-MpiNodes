class MpiDynamicCombo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "option_1": ("STRING", {}),
                "option_2": ("STRING", {}),
                "option_3": ("STRING", {}),
                "option_4": ("STRING", {}),
                "option_5": ("STRING", {}),
                "option_6": ("STRING", {}),
                "option_7": ("STRING", {}),
                "option_8": ("STRING", {}),
                "selected": (
                    "COMBO",
                    {
                        "choices": ["None"],  # placeholder, will be overridden
                    },
                ),
            }
        }

    CATEGORY = "MpiNodes/Logic"
    RETURN_TYPES = ("STRING", "INT")
    FUNCTION = "use_selected"

    def use_selected(
        self,
        option_1,
        option_2,
        option_3,
        option_4,
        option_5,
        option_6,
        option_7,
        option_8,
        selected,
    ):
        options = [
            option_1,
            option_2,
            option_3,
            option_4,
            option_5,
            option_6,
            option_7,
            option_8,
        ]
        sel_int = next(
            (i + 1 for i, val in enumerate(options) if val == selected), -1
        )
        return (selected, sel_int)
