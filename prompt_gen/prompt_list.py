from ..help_funcs import (
    probabilistic_shuffle,
    wrap_in_underscores,
    get_random_bool,
)


class MpiPromptList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "title": (
                    "STRING",
                    {
                        "multiline": False,
                        "tooltip": "The title used to trigger this list in a Prompt Processor",
                    },
                ),
                "prefix": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "A handy prefix to add to each option",
                    },
                ),
                "options": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "Each line is a selectable option to be added to the prompt",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "A handy suffix to add to each option\nBe careful as this will be placed at the end of each option, meaning after negative options.",
                    },
                ),
                "blocks": (
                    "STRING",
                    {
                        "multiline": False,
                        "tooltip": "Lists that this list should block (separated by commas)",
                    },
                ),
                "amount": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "tooltip": "If bigger than 0, it will output the selected 'amount' of items as a single option separated by commas. \nThis will stop _multipliers_x2_ and _pickers_3_ from working!",
                    },
                ),
                "specific_item": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 100,
                        "tooltip": "If bigger than -1: \n0 - Will bypass the list (Ignores 'Probability')\n1 or greater - It will output the selected item only. \nThis will stop _multipliers_x2_ and _pickers_3_ from working and overwrite the 'amount' option!",
                    },
                ),
                "random": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "enabled",
                        "label_off": "disabled",
                        "tooltip": "When disabled the list order will not change",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Affects 'probability' and 'options' order.",
                    },
                ),
                "probability": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Probability of this list moving on.\n0.0 will always bypass this list and return False \n1.0 will always let this list through and return True\n0.5 gives both a 50% chance",
                    },
                ),
            },
            "optional": {
                "opt_chained_list": ("PROMPT_LST", {"forceInput": True}),
                "opt_override": ("PROMPT_OVR", {"forceInput": True}),
            },
        }

    CATEGORY = "MpiNodes/PromptGen"
    DESCRIPTION = "Create a prompt list with options, probability, and blocking rules"
    RETURN_TYPES = ("PROMPT_LST", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("prompt_list", "seed", "boolean", "selected_options")
    FUNCTION = "create_list"

    def create_list(
        self,
        title: str,
        prefix: str,
        suffix: str,
        options: str,
        blocks: str,
        random: bool,
        seed: int,
        probability: float,
        amount: int,
        specific_item: int = -1,
        opt_chained_list: list = [],
        opt_override: tuple[str, int] = ("", -1),
    ):

        if not title:
            raise ValueError("Error: Prompt List needs a Title!")

        ovr_options, seed_override = opt_override
        seed_override = seed_override if seed_override != -1 else seed
        if ovr_options:
            options = ovr_options

        should_continue = (
            get_random_bool(probability, seed_override)
            if (random or not probability)
            else True
        )

        if not should_continue or specific_item == 0:
            return (opt_chained_list, seed, should_continue, [])

        options_lst: list[str] = options.splitlines()

        picked_options = (
            probabilistic_shuffle(options_lst, float(random), seed_override)
            if random
            else options_lst
        )

        if amount or specific_item != -1:
            picked_options = self.process_amount(
                picked_options, amount, specific_item
            )

        new_list = [i for i in opt_chained_list]
        title = title.lower()
        title = wrap_in_underscores(title)
        blocks = blocks.lower()
        listed_blocks = [item.strip() for item in blocks.split(",")]
        listed_blocks = [wrap_in_underscores(i) for i in listed_blocks]

        self_list = {
            "title": title,
            "options": [prefix + o + suffix for o in picked_options],
            "blocks": listed_blocks,
        }
        new_list.append(self_list)

        return (new_list, seed, should_continue, picked_options)

    def process_amount(self, options: list, amount: int, specific_item: int):

        if specific_item != -1:
            # if specific_item == 0 (it wont reach here as it as been handled before)
            return [options[specific_item - 1]]

        # turn pos and neg strings (pos | neg) into lists ["pos", "neg"]
        options = [[part.strip() for part in s.split("|")] for s in options]

        out_option_pos: str = ""
        out_option_neg: str = ""
        for option in options[:amount]:
            out_option_pos += option[0] + ", "
            if len(option) > 1:
                out_option_neg += option[1] + ", "
        out_option_pos = out_option_pos.rstrip(", ")
        out_option_neg = out_option_neg.rstrip(", ")
        single_option: str = out_option_pos
        if out_option_neg:
            single_option += " | " + out_option_neg

        return [single_option]
