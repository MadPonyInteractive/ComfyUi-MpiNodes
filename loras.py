from .help_funcs import load_lora_cached, comfy_paths, AlwaysEqualProxy
import comfy.sd  # type: ignore


class MpiDoubleLora:
    CATEGORY = "MpiNodes/ModelOps"
    DESCRIPTION = "Apply LoRA to two models and clips simultaneously"
    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "CLIP")
    RETURN_NAMES = (
        "model_a_out",
        "model_b_out",
        "clip_a_out",
        "clip_b_out",
    )
    FUNCTION = "apply_lora"

    @classmethod
    def INPUT_TYPES(cls):
        file_list = comfy_paths.get_filename_list("loras")
        file_list.insert(0, "None")
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "lora_name": (file_list,),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
            "optional": {
                "clip_a": ("CLIP",),
                "clip_b": ("CLIP",),
            },
        }

    def apply_lora(
        self,
        model_a,
        model_b,
        lora_name,
        strength_model,
        strength_clip,
        clip_a=None,
        clip_b=None,
    ):
        if strength_model == 0 and strength_clip == 0 or lora_name == "None":
            return (model_a, model_b, clip_a, clip_b)

        lora = load_lora_cached(lora_name)

        model_a_out, clip_a_out = comfy.sd.load_lora_for_models(
            model_a, clip_a, lora, strength_model, strength_clip
        )
        model_b_out, clip_b_out = comfy.sd.load_lora_for_models(
            model_b, clip_b, lora, strength_model, strength_clip
        )

        return (model_a_out, model_b_out, clip_b_out, clip_a_out)
        # name_string = os.path.splitext(os.path.basename(lora_name))[0]
        # return (model_a_out, clip_a_out, model_b_out, clip_b_out, name_string)


class MpiLoraModel:
    CATEGORY = "MpiNodes/ModelOps"
    DESCRIPTION = "Apply LoRA to model with strength control"
    RETURN_TYPES = ("MODEL", AlwaysEqualProxy("*"), "FLOAT")
    RETURN_NAMES = (
        "model",
        "lora_name",
        "strength",
    )
    FUNCTION = "apply_lora"

    @classmethod
    def INPUT_TYPES(cls):
        file_list = comfy_paths.get_filename_list("loras")
        file_list.insert(0, "None")
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (file_list,),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
        }

    def apply_lora(
        self,
        model,
        lora_name,
        strength_model,
    ):
        if strength_model == 0 or lora_name == "None":
            return (model, lora_name, strength_model)

        lora = load_lora_cached(lora_name)

        model_out, clip_a_out = comfy.sd.load_lora_for_models(
            model, None, lora, strength_model, 0.0
        )

        return (model_out, lora_name, strength_model)


class MpiStyleSelector:
    CATEGORY = "MpiNodes/ModelOps"
    DESCRIPTION = (
        "Style router: one 'selector' integer picks a style, and every "
        "MpiStyleLoras bank chained after this node applies that style's LoRA "
        "and passes its trigger words along.\n\n"
        "'triggers' is one style per line. Line 1 pairs with lora_1 of the "
        "first bank, line 6 with lora_1 of the second bank, and so on. "
        "selector 0 = no style (model and clip pass through untouched, prompt "
        "is empty). A line whose lora slot is left at 'None' is a "
        "prompt-only style.\n\n"
        "strength_model / strength_clip apply to whichever LoRA is active. "
        "Leave strength_clip at 0 (or leave clip unconnected) for model-only "
        "LoRAs."
    )
    RETURN_TYPES = ("MPI_STYLE",)
    RETURN_NAMES = ("style",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "selector": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "Which style to use. 0 = none, 1 = first trigger line / lora_1 of the first bank.",
                    },
                ),
                "triggers": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "One style trigger per line. Line N is style N — keep the line order in sync with the lora slots.",
                    },
                ),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    def build(
        self,
        model,
        selector,
        triggers,
        strength_model,
        strength_clip,
        clip=None,
    ):
        lines = triggers.splitlines()
        prompt = lines[selector - 1].strip() if 1 <= selector <= len(lines) else ""

        return (
            {
                "model": model,
                "clip": clip,
                "selector": selector,
                "strength_model": strength_model,
                "strength_clip": strength_clip,
                "prompt": prompt,
                # selector value that the next bank's lora_1 answers to
                "first_slot": 1,
            },
        )


class MpiStyleLoras:
    CATEGORY = "MpiNodes/ModelOps"
    DESCRIPTION = (
        "A bank of 5 LoRA slots for an MpiStyleSelector. Applies only the LoRA "
        "whose slot matches the selector, then passes model, clip and the "
        "style's trigger words through.\n\n"
        "Each bank claims 5 consecutive selector values automatically: the "
        "first bank answers to selector 1-5, a second bank chained onto its "
        "'style' output answers to 6-10, and so on. Leave a slot at 'None' for "
        "a style that is trigger words only."
    )
    RETURN_TYPES = ("MPI_STYLE", "MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("style", "model", "clip", "prompt")
    FUNCTION = "apply_lora"

    @classmethod
    def INPUT_TYPES(cls):
        file_list = comfy_paths.get_filename_list("loras")
        file_list.insert(0, "None")
        return {
            "required": {
                "style": ("MPI_STYLE", {"forceInput": True}),
                "lora_1": (file_list,),
                "lora_2": (file_list,),
                "lora_3": (file_list,),
                "lora_4": (file_list,),
                "lora_5": (file_list,),
            },
        }

    def apply_lora(self, style, lora_1, lora_2, lora_3, lora_4, lora_5):
        names = (lora_1, lora_2, lora_3, lora_4, lora_5)
        out = dict(style)
        # ponytail: a bank always claims 5 slots, even if some are "None", so
        # the indexes of a chained bank never shift when a slot is emptied
        out["first_slot"] = style["first_slot"] + len(names)

        slot = style["selector"] - style["first_slot"]
        if 0 <= slot < len(names):
            lora_name = names[slot]
            strength_model = style["strength_model"]
            strength_clip = style["strength_clip"]
            if lora_name != "None" and (strength_model or strength_clip):
                lora = load_lora_cached(lora_name)
                out["model"], out["clip"] = comfy.sd.load_lora_for_models(
                    style["model"],
                    style["clip"],
                    lora,
                    strength_model,
                    strength_clip,
                )

        return (out, out["model"], out["clip"], out["prompt"])


class MpiLoraModelClip:
    CATEGORY = "MpiNodes/ModelOps"
    DESCRIPTION = "Apply LoRA to model and clip with independent strength controls"
    RETURN_TYPES = ("MODEL", "CLIP", AlwaysEqualProxy("*"), "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "model",
        "clip",
        "lora_name",
        "strength_model",
        "strength_clip",
    )
    FUNCTION = "apply_lora"

    @classmethod
    def INPUT_TYPES(cls):
        file_list = comfy_paths.get_filename_list("loras")
        file_list.insert(0, "None")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (file_list,),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
        }

    def apply_lora(
        self,
        model,
        clip,
        lora_name,
        strength_model,
        strength_clip,
    ):
        if (strength_model == 0 and strength_clip == 0) or lora_name == "None":
            return (model, clip, lora_name, strength_model, strength_clip)

        lora = load_lora_cached(lora_name)

        model_out, clip_out = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )

        return (model_out, clip_out, lora_name, strength_model, strength_clip)
