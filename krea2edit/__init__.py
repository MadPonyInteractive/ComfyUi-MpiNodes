"""Krea 2 edit nodes ported from comfyui-krea2edit and ComfyUI-Krea2-Ostris-Edit for Cubric Studio
(MPI-1343).
"""
from . import edit, ostris

NODE_CLASS_MAPPINGS = {**edit.NODE_CLASS_MAPPINGS, **ostris.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**edit.NODE_DISPLAY_NAME_MAPPINGS, **ostris.NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
