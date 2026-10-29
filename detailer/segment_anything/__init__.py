"""Vendored subset of Meta's segment-anything (Apache-2.0, see LICENSE next to this file).

Copied unchanged from segment-anything 1.0 (modeling/*, predictor.py, utils/transforms.py) so the
detailer chain does not need the `segment-anything` pip package. Only change: predictor.py imports
`Sam` relatively, and `build.py` (below) is build_sam.py cut to the vit_b / vit_l / vit_h builders
with an explicit `weights_only=True` load.
"""
from .build import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l, sam_model_registry  # noqa: F401
from .predictor import SamPredictor  # noqa: F401
