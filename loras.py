from .help_funcs import load_lora_cached, comfy_paths, AlwaysEqualProxy
import comfy.sd  # type: ignore


class MpiDoubleLora:
    CATEGORY = "MpiNodes/ModelOps"
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


class MpiLoraModelClip:
    CATEGORY = "MpiNodes/ModelOps"
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
