"""CPU checks for wan_masked.py (MpiWanMaskedVideo) on tiny tensors. No ComfyUI, no GPU. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_wan_masked.py
"""
import os
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "comfy.utils" not in sys.modules:  # the one ComfyUI helper masked_guide uses, stood in for
    comfy = types.ModuleType("comfy")
    comfy.utils = types.ModuleType("comfy.utils")
    comfy.utils.common_upscale = lambda x, w, h, method, crop: F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    sys.modules.update({"comfy": comfy, "comfy.utils": comfy.utils})

from wan_masked import latent_mask, masked_guide  # noqa: E402


def test_all_known_generates_nothing():
    m = latent_mask(torch.ones(81, 16, 32), 2, 4)
    assert m.shape == (1, 1, 21, 2, 4), m.shape
    assert m.sum() == 0


def test_a_hole_lands_in_its_latent_frame():
    known = torch.ones(81, 16, 32)
    known[5:9, :8, :8] = 0  # pixel frames 5-8 = latent frame 2, top-left latent cell
    m = latent_mask(known, 2, 4)[0, 0]
    assert m[2, 0, 0] == 1 and m.sum() == 1, m.nonzero()
    known[0, :8, :8] = 0  # frame 0 is its own latent frame
    assert latent_mask(known, 2, 4)[0, 0, 0, 0, 0] == 1


def test_half_rule():
    known = torch.ones(81, 16, 32)
    known[1:5, :8, :3] = 0  # 3/8 of the cell unknown in latent frame 1: still known
    assert latent_mask(known, 2, 4).sum() == 0
    known[1:5, :8, :5] = 0  # 5/8 unknown: generate
    assert latent_mask(known, 2, 4)[0, 0, 1, 0, 0] == 1


def test_guide_holes_are_black_and_short_guides_end_in_holes():
    guide = torch.full((3, 8, 16, 3), 0.7)
    holes = torch.zeros(3, 8, 16, 3)
    holes[1, :4, :8] = 1.0  # white = hole
    image, known = masked_guide(guide, holes, 16, 8, 5)
    assert image.shape == (5, 8, 16, 3) and known.shape == (5, 8, 16)
    assert known[1, :4, :8].sum() == 0 and image[1, :4, :8].abs().sum() == 0, "a hole is unknown and black"
    assert known[1, 4:, 8:].min() == 1 and torch.allclose(image[1, 4:, 8:], torch.tensor(0.7)), "the rest is kept"
    assert known[3:].sum() == 0 and image[3:].abs().sum() == 0, "frames past the guide are holes"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
