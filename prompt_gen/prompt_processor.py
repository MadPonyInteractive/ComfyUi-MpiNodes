import re

from ..help_funcs import (
    wrap_in_underscores,
    get_blacklist,
    get_random_bool,
    merge_unique_by_key,
)


class MpiPromptProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_list": ("PROMPT_LST", {"forceInput": True}),
                "positive": (
                    "STRING",
                    {"multiline": True, "tooltip": "Positive prompt"},
                ),
                "negative": (
                    "STRING",
                    {"multiline": True, "tooltip": "Negative prompt"},
                ),
                "cleanup": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "enabled",
                        "label_off": "disabled",
                        "tooltip": "When enabled, it will remove any _title_ that couldn't be replaced",
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "probability": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0.0 will always bypass this prompt output and return False \n1.0 will always let this prompt through and return True\n0.5 gives both a 50% chance",
                    },
                ),
            },
            "optional": {
                "opt_title": (
                    "STRING",
                    {
                        "multiline": False,
                        "tooltip": "Unique ID, use to convert to a reusable prompt",
                    },
                ),
            },
        }

    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ("PROMPT_LST", "STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("prompt", "positive", "negative", "seed", "boolean")
    FUNCTION = "process"

    def process(
        self,
        prompt_list,
        positive,
        negative,
        cleanup,
        seed,
        probability,
        opt_title="",
    ):
        # Replace each _Word_ with its lowercase version
        positive = re.sub(
            r"_([^_]+)_", lambda m: f"_{m.group(1).lower()}_", positive
        )
        negative = re.sub(
            r"_([^_]+)_", lambda m: f"_{m.group(1).lower()}_", negative
        )

        blacklist = get_blacklist(prompt_list)

        prompts = [d for d in prompt_list if "positive" in d]
        lists = [d for d in prompt_list if "positive" not in d]

        # Process lists
        for lst in lists:
            if lst["title"] not in blacklist:
                multipliers = self.extract_x_tokens(positive)
                positive, negative = self.process_list(
                    positive, negative, lst, multipliers
                )

        # Process Prompts
        for prompt in prompts:
            if prompt["title"] not in blacklist:
                positive = positive.replace(
                    prompt["title"], prompt["positive"]
                )
                negative = negative.replace(
                    prompt["title"], prompt["negative"]
                )

        # Remove any _word_ followed by optional punctuation/whitespace before the next word
        if cleanup:
            positive = self.cleanup_prompt(positive)
            negative = self.cleanup_prompt(negative)

        self_prompt = []
        should_continue: bool
        if probability == 1.0:
            should_continue = True
        elif probability == 0.0:
            should_continue = False
        else:
            should_continue = get_random_bool(probability, seed)

        if should_continue and opt_title:
            title = opt_title.lower()
            title = wrap_in_underscores(title)
            self_prompt = [
                {
                    "title": title,
                    # "options": [],
                    "positive": positive,
                    "negative": negative,
                    "blocks": [],
                }
            ]

        out_prompt_list = merge_unique_by_key(
            [i for i in prompt_list], self_prompt
        )
        return (out_prompt_list, positive, negative, seed, should_continue)
        return (self_prompt, positive, negative, seed, should_continue)

    def cleanup_prompt(self, prompt: str) -> str:
        # Step 1: Remove underscored tokens and surrounding comma/space (preserve punctuation before them)
        # prompt = re.sub(
        #     r"(?:,\s*|^)?(_\w+_)(?:\s*,)?(?=\s|[.,!?]|$)", "", prompt
        # )
        prompt = re.sub(r"\s*_([a-zA-Z0-9_]+)_[.,]?", "", prompt)

        # Step 2: Remove space before punctuation
        prompt = re.sub(r"\s+([.,!?])", r"\1", prompt)

        # Step 3: Collapse double spaces and trim
        prompt = re.sub(r"\s{2,}", " ", prompt).strip()

        # Step 4: Collapse double periods into single
        prompt = re.sub(r"\.{2,}", ".", prompt)

        # Step 5: Replace trailing comma with a period if needed
        if prompt.endswith(","):
            prompt = prompt[:-1] + "."

        return prompt

    def process_list(
        self, positive: str, negative: str, lst: dict, multipliers: dict
    ):
        # turn pos and neg strings (pos | neg) into lists ["pos", "neg"]
        options = [
            [part.strip() for part in s.split("|")] for s in lst["options"]
        ]

        # Replace numbered triggers
        for i, option in enumerate(options):
            trigger = lst["title"] + str(i + 1) + "_"
            positive = positive.replace(trigger, option[0])
            if len(option) > 1:
                negative = negative.replace(trigger, option[1])

        # Replace multipliers
        amount = multipliers.get(lst["title"])
        if amount:
            trigger = lst["title"] + "x" + str(amount) + "_"
            pos_replace = ""
            neg_replace = ""
            for option in options[:amount]:
                pos_replace += option[0] + ", "
                if len(option) > 1:
                    neg_replace += option[1] + ", "
            pos_replace = pos_replace.rstrip(", ")
            neg_replace = neg_replace.rstrip(", ")
            positive = positive.replace(trigger, pos_replace)
            negative = negative.replace(trigger, neg_replace)

        # Replace un-numbered trigger
        for option in options:
            # positive = positive.replace(lst["title"], option[0])
            positive = self.replace_exact_token(
                positive, lst["title"], option[0]
            )

            if len(option) > 1:
                # negative = negative.replace(lst["title"], option[1])
                negative = self.replace_exact_token(
                    negative, lst["title"], option[1]
                )

        return positive, negative

    def replace_exact_token(self, text, token, replacement):
        # Escape the full token (e.g. "_env_") and match it only when not part of a longer word
        pattern = rf"(?<!\w){re.escape(token)}(?!\w)"
        return re.sub(pattern, replacement, text)

    def extract_x_tokens(self, text):
        # [
        #     {"_accessories_": 2},
        #     {"_makeup_": 1}
        # ]
        pattern = r"(_[^_]+)_x(\d+)_"
        matches = re.findall(pattern, text)
        return {f"{title}_": int(count) for title, count in matches}
