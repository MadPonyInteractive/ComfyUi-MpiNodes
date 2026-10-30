"""CPU checks for MpiH3PinFirstFrame on tiny tensors. No GPU. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_h3_first_frame.py

Needs ComfyUI's comfy.nested_tensor: set COMFYUI_DIR, default G:/ComfyUi/ComfyUI.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("COMFYUI_DIR", "G:/ComfyUi/ComfyUI"))
import comfy.nested_tensor  # noqa: E402

from h3 import MpiH3MaskedPrefix, MpiH3PinFirstFrame  # noqa: E402


def av(steps, fill, audio=207):
    v = torch.full((1, 24, steps, 2, 3), float(fill))
    a = torch.full((1, 32, 2, audio), float(fill))
    return {"samples": comfy.nested_tensor.NestedTensor((v, a))}


def test_one_image_pins_step_zero_video_only():
    image = {"samples": torch.full((1, 24, 1, 2, 3), 5.0)}  # plain VAEEncode of one image
    out, = MpiH3PinFirstFrame().doit(av(37, 0), image)
    v, a = out["samples"].tensors
    mv, ma = out["noise_mask"].tensors
    assert (v[:, :, 0] == 5).all() and (v[:, :, 1:] == 0).all() and (a == 0).all()
    assert mv[0, 0, :, 0, 0].tolist() == [0.0] + [1.0] * 36
    assert ma.min() == 1  # audio stays fully generated


def test_av_clip_gives_its_first_step():
    clip = av(37, 1)
    clip["samples"].tensors[0][:, :, 0] = 9
    out, = MpiH3PinFirstFrame().doit(av(37, 0), clip)
    v = out["samples"].tensors[0]
    assert (v[:, :, 0] == 9).all() and (v[:, :, 1:] == 0).all()


def test_keeps_an_existing_prefix_mask():
    prefixed, *_ = MpiH3MaskedPrefix().doit(av(57, 0, audio=320), av(37, 1), 39, av(37, 2))
    out, = MpiH3PinFirstFrame().doit(prefixed, {"samples": torch.full((1, 24, 1, 2, 3), 5.0)})
    mv, ma = out["noise_mask"].tensors
    assert mv[0, 0, :, 0, 0].tolist() == [0.0] * 12 + [1.0] * 33 + [0.0] * 12
    assert ma[0, 0, 0].tolist() == [0.0] * 65 + [1.0] * 190 + [0.0] * 65


def test_canvas_mismatch_raises():
    try:
        MpiH3PinFirstFrame().doit(av(37, 0), {"samples": torch.zeros((1, 24, 1, 4, 3))})
    except ValueError as e:
        assert "canvas" in str(e)
    else:
        raise AssertionError("a frame on another canvas must raise")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
