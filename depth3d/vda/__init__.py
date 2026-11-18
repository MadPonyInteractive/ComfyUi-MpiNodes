# Video Depth Anything model code, vendored (Apache-2.0 - see LICENSE beside this file).
# Source: github.com/DepthAnything/Video-Depth-Anything @ 4f5ae23 (2025-10-07). Model files only:
# dinov2.py, dinov2_layers/, dpt.py, dpt_temporal.py, motion_module/, util/blocks.py. Changed:
# dpt_temporal.py drops the easydict import; the four "xFormers not available" boot prints are
# silenced; dinov2_layers/attention.py's Attention uses torch's fused scaled_dot_product_attention
# (same maths, no N x N matrix). The windowed inference lives in depth3d/stream.py.
