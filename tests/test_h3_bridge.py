"""CPU checks for MpiH3MaskedPrefix's bridge tail on tiny tensors. No GPU. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_h3_bridge.py

Needs ComfyUI's comfy.nested_tensor: set COMFYUI_DIR, default G:/ComfyUi/ComfyUI.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("COMFYUI_DIR", "G:/ComfyUi/ComfyUI"))
import comfy.nested_tensor  # noqa: E402

from h3 import MpiH3MaskedPrefix, latent_frames  # noqa: E402

STEPS = {39: 12, 124: 37, 192: 57}  # 17k+5 frames -> 5k+2 latent steps


def av(frames, fill):
    v = torch.full((1, 24, STEPS[frames], 2, 3), float(fill))
    a = torch.full((1, 32, 2, round(frames * 40 / 24)), float(fill))
    return {"samples": comfy.nested_tensor.NestedTensor((v, a))}


def run(target, ctx, tail=None, **kw):
    out, ctx_frames, new_frames, report = MpiH3MaskedPrefix().doit(target, ctx, 39, tail, **kw)
    v, a = out["samples"].tensors
    mv, ma = out["noise_mask"].tensors
    return v, a, mv, ma, ctx_frames, new_frames


def test_steps_table():
    for frames, steps in STEPS.items():
        assert latent_frames(steps) == frames


def test_bridge_holds_both_ends():
    v, a, mv, ma, ctx_frames, new_frames = run(av(192, 0), av(124, 1), av(124, 2))
    assert (ctx_frames, new_frames) == (39, 114)
    assert (v[:, :, :12] == 1).all() and (v[:, :, 12:45] == 0).all() and (v[:, :, 45:] == 2).all()
    assert (a[..., :65] == 1).all() and (a[..., 65:255] == 0).all() and (a[..., 255:] == 2).all()
    assert mv[0, 0, :, 0, 0].tolist() == [0.0] * 12 + [1.0] * 33 + [0.0] * 12
    assert ma[0, 0, 0].tolist() == [0.0] * 65 + [1.0] * 190 + [0.0] * 65


def test_tail_takes_the_next_clips_opening_not_its_end():
    tail = av(124, 0)
    tail["samples"].tensors[0][:, :, :12] = 7
    tail["samples"].tensors[1][..., :65] = 7
    v, a, *_ = run(av(192, 0), av(124, 1), tail)
    assert (v[:, :, 45:] == 7).all() and (a[..., 255:] == 7).all()


def test_no_tail_is_unchanged():
    v, a, mv, ma, ctx_frames, new_frames = run(av(124, 0), av(124, 1))
    assert new_frames == 85 and mv[0, 0, 12:].min() == 1 and ma[0, 0, 0, 65:].min() == 1


def test_off_clock_target_raises():
    # 124 frames is on the video grid but 124 * 5/3 is not a whole audio step.
    try:
        run(av(124, 0), av(124, 1), av(124, 2))
    except ValueError as e:
        assert "divides by 3" in str(e)
    else:
        raise AssertionError("124-frame bridge must raise")


def test_no_room_raises():
    try:
        run(av(39, 0), av(124, 1), av(124, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("39-frame target cannot hold two 39-frame ends")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
