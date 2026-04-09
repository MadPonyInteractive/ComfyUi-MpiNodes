import gc
import torch  # type: ignore
import comfy.model_management  # type: ignore
from .help_funcs import AlwaysEqualProxy

_any = AlwaysEqualProxy("*")


class MpiClearVram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "passthrough": (_any, {"forceInput": True}),
            }
        }

    RETURN_TYPES = (_any,)
    RETURN_NAMES = ("passthrough",)
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "Offloads models from VRAM to RAM and clears the GPU cache. "
        "Models are reloaded automatically on the next run. "
        "Optionally passes through a value to chain in workflows."
    )
    FUNCTION = "doit"
    OUTPUT_NODE = True

    def doit(self, passthrough=None):
        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return (passthrough,)
