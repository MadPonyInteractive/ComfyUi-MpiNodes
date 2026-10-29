"""Ultimate SD Upscale ported from ComfyUI_UltimateSDUpscale for Cubric Studio (MPI-1343).
"""
from . import node

NODE_CLASS_MAPPINGS = {**node.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**node.NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
