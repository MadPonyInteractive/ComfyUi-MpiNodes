import gc
import torch  # type: ignore
import comfy.model_management  # type: ignore
from .help_funcs import AlwaysEqualProxy

_any = AlwaysEqualProxy("*")


def _clear_vram():
    gc.collect()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
        _clear_vram()
        return (passthrough,)


class MpiClearVramEnd:
    """Terminal twin of MpiClearVram, for a branch you do not want to rewire.

    Clearing VRAM properly needs one of these per terminal branch, because the
    node only runs on branches that actually execute -- a single clear hung off
    the preview branch leaves the models resident for everything else. The
    pass-through version has to be spliced INTO a branch, which means moving a
    link; this one hangs off the end of a branch and changes nothing downstream,
    because there is no downstream.

    The input exists only to place the node in execution order. It is what ties
    the clear to "after this branch has finished", so it is required -- an
    unconnected sink would run at an arbitrary point.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (_any, {
                    "forceInput": True,
                    "tooltip": "Connect the last value on the branch. Its only job is "
                               "ordering: the clear runs once that branch is done.",
                }),
            }
        }

    RETURN_TYPES = ()
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "Offloads models from VRAM to RAM and clears the GPU cache, as a terminal "
        "node with no output. Hang one off the end of each branch that finishes on "
        "the GPU - a latent, a decoded image, a decoded audio - since a clear only "
        "runs on a branch that executes. Models are reloaded automatically on the "
        "next run."
    )
    FUNCTION = "doit"
    OUTPUT_NODE = True

    def doit(self, trigger=None):
        _clear_vram()
        return ()
