"""CPU checks for the H3 window join (Cubric Studio MPI-1608 Phase 3) on tiny tensors. No GPU, no ComfyUI.

    G:/ComfyUi/python_embeded/python.exe tests/test_h3_window_join.py

Optional real-data check: set MPI1608_W1 / MPI1608_W2 to two saved window latents (.latent) of one run.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from h3 import latent_frames  # noqa: E402
from h3_window_join import join_windows, lead_steps  # noqa: E402

STEPS = {124: 37, 141: 42}  # 17k+5 frames -> 5k+2 latent steps
AUDIO = {124: 207, 141: 235}  # frames * 40 / 24, as H3 sizes it


def window(frames, start, head=None):
    """A window whose video step t holds `start + t` and audio step s holds `1000 + start*... `; `head` = the
    previous window's (video, audio) to copy its tail into this window's front, as MpiH3MaskedPrefix does."""
    v = torch.arange(STEPS[frames], dtype=torch.float32).add(start).view(1, 1, -1, 1, 1).expand(1, 2, -1, 2, 3).clone()
    a = torch.arange(AUDIO[frames], dtype=torch.float32).add(10000 + start).view(1, 1, 1, -1).expand(1, 2, 2, -1).clone()
    if head is not None:
        hv, ha = head
        v[:, :, :12] = hv[:, :, -12:]
        a[..., :65] = ha[..., -65:]
    return v, a


def test_two_windows_join_to_209_frames():
    w1 = window(124, 0)
    w2 = window(124, 500, head=w1)
    v, a, report = join_windows([w1, w2], 39)
    assert v.shape[2] == 62 and latent_frames(62) == 209, report
    assert a.shape[-1] == 207 + 207 - 65
    assert torch.equal(v[:, :, :37], w1[0]) and torch.equal(v[:, :, 37:], w2[0][:, :, 12:])
    assert torch.equal(a[..., :207], w1[1])


def test_three_windows():
    w1 = window(124, 0)
    w2 = window(124, 500, head=w1)
    w3 = window(124, 900, head=w2)
    v, _, _ = join_windows([w1, w2, w3], 39)
    assert latent_frames(v.shape[2]) == 124 + 85 + 85


def test_a_window_from_another_run_raises():
    w1 = window(124, 0)
    stranger = window(124, 500)
    try:
        join_windows([w1, stranger], 39)
    except ValueError as e:
        assert "not the next window" in str(e)
    else:
        raise AssertionError("joined a window whose head is not window 1's tail")


def test_lead_in_dropped_from_the_first_window_only():
    assert lead_steps(42, 17) == 5 and lead_steps(42, 0) == 0
    w1 = window(141, 0)  # 17 lead-in frames + 124
    w2 = window(124, 500, head=w1)
    v, a, _ = join_windows([w1, w2], 39, lead_frames=17)
    assert latent_frames(v.shape[2]) == 124 + 85
    assert v[0, 0, 0, 0, 0].item() == 5  # step 5 of window 1 opens the film
    assert a.shape[-1] == (235 - 28) + (207 - 65)


def test_a_lead_in_off_the_grid_raises():
    try:
        lead_steps(42, 10)
    except ValueError:
        return
    raise AssertionError("a 10-frame lead-in was accepted")


def test_real_windows():
    p1, p2 = os.environ.get("MPI1608_W1"), os.environ.get("MPI1608_W2")
    if not (p1 and p2):
        return
    from safetensors.torch import load_file
    w = [load_file(p) for p in (p1, p2)]
    v, a, report = join_windows([(d["latent_tensor_0"], d["latent_tensor_1"]) for d in w], 39)
    print(report)
    assert latent_frames(v.shape[2]) == 209


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
