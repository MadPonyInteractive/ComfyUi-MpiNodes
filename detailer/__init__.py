# The face / person / tile detailer chain (MPI-1339), ported from ComfyUI-Impact-Pack and
# ComfyUI-Impact-Subpack so those two packs can leave the engine. NOT registered in the pack's
# __init__.py yet: the swap card does that. Importing this package needs no ComfyUI.
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: F401
