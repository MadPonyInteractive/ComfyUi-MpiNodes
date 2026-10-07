# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Modified for ComfyUi-MpiNodes (Mad Pony Interactive, 2026): the pretrained-weights DOWNLOAD is removed (MoGe's checkpoint carries the backbone weights and loads from a local file); only ViT-L/14, the backbone of the shipped MoGe checkpoint, is kept.

from ..models import vision_transformer as vits


def dinov2_vitl14(*, img_size: int = 518, patch_size: int = 14, init_values: float = 1.0,
                  ffn_layer: str = "mlp", block_chunks: int = 0, num_register_tokens: int = 0,
                  interpolate_antialias: bool = False, interpolate_offset: float = 0.1, **kwargs):
    """DINOv2 ViT-L/14, random init (the MoGe checkpoint supplies the weights)."""
    return vits.vit_large(img_size=img_size, patch_size=patch_size, init_values=init_values,
                          ffn_layer=ffn_layer, block_chunks=block_chunks,
                          num_register_tokens=num_register_tokens,
                          interpolate_antialias=interpolate_antialias,
                          interpolate_offset=interpolate_offset, **kwargs)
