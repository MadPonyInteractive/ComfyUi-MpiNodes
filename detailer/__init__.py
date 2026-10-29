# The face / person / tile detailer chain (MPI-1339), ported from ComfyUI-Impact-Pack and
# ComfyUI-Impact-Subpack so those two packs can leave the engine. The pack's __init__.py merges
# these mappings into its own. Importing this package needs no ComfyUI.
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: F401
