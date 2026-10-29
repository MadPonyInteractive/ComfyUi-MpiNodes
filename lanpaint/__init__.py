"""Nodes ported from LanPaint for Cubric Studio (MPI-1343).
"""
from . import nodes

NODE_CLASS_MAPPINGS = {**nodes.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**nodes.NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
