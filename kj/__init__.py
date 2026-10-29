"""Nodes ported from ComfyUI-KJNodes and color-matcher for Cubric Studio (MPI-1341).
"""
from . import image, ltx, lora_extract

NODE_CLASS_MAPPINGS = {**image.NODE_CLASS_MAPPINGS, **ltx.NODE_CLASS_MAPPINGS, **lora_extract.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**image.NODE_DISPLAY_NAME_MAPPINGS, **ltx.NODE_DISPLAY_NAME_MAPPINGS,
                              **lora_extract.NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
