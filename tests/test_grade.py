"""CPU checks for grade.py (MpiGradeMatch) on tiny tensors. No ComfyUI, no GPU. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_grade.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F  # noqa: E402

from grade import MpiGradeMatch, _ring, grade_match  # noqa: E402

GAIN, OFFSET = torch.tensor([0.8, 0.9, 1.1]), torch.tensor([0.10, -0.05, 0.02])
RED = torch.tensor([0.9, 0.1, 0.1])


def _scene(frames=3, size=64):
    """A textured plate, a re-render of it with grading drift, and a red object added inside a box."""
    torch.manual_seed(0)
    plate = torch.rand(frames, size, size, 3) * 0.6 + 0.2
    mask = torch.zeros(frames, size, size)
    mask[:, 20:44, 20:44] = 1.0
    render = plate.clone()
    render[:, 26:38, 26:38] = RED  # the edit
    render = render * GAIN + OFFSET  # what a re-render does to the grade
    return plate, render, mask


def test_outside_is_the_plate_inside_is_graded_back():
    plate, render, mask = _scene()
    out = grade_match(plate, render, mask, band=8)
    assert torch.equal(out[:, :20], plate[:, :20])  # outside the mask: the original, untouched
    # the added object comes back in its OWN colour, the drift removed, not pulled toward the plate
    assert (out[:, 30, 30] - RED).abs().max() < 1e-3, out[0, 30, 30]
    # and the unchanged part of the box matches the plate it came from
    assert (out[:, 22, 22] - plate[:, 22, 22]).abs().max() < 1e-3


def test_motion_in_the_band_does_not_skew_the_fit():
    plate, render, mask = _scene()
    render[:, 44:50, 20:30] = 0.0  # a dark hand crossing the band in the re-render only
    out = grade_match(plate, render, mask, band=8)
    assert (out[:, 30, 30] - RED).abs().max() < 1e-2, out[0, 30, 30]


def test_one_mask_frame_holds_for_the_clip_and_sizes_are_resampled():
    plate, render, mask = _scene(frames=4)
    small = torch.nn.functional.interpolate(render.movedim(-1, 1), size=(32, 32), mode="nearest").movedim(1, -1)
    out = MpiGradeMatch().doit(plate, small, mask[:1], 8)[0]
    assert out.shape == plate.shape
    assert torch.equal(out[:, :20], plate[:, :20])


def test_empty_mask_returns_the_plate():
    plate, render, _ = _scene()
    out = grade_match(plate, render, torch.zeros(3, 64, 64), band=8)
    assert torch.allclose(out, plate)


def test_frame_count_mismatch_raises():
    plate, render, mask = _scene()
    try:
        grade_match(plate, render[:2], mask, band=8)
    except ValueError as e:
        assert "frame counts differ" in str(e)
    else:
        raise AssertionError("expected a ValueError")


def test_ring_is_the_full_square_dilation_and_a_moving_mask_still_gets_its_own():
    # The separable ring must match the (2*band+1)^2 max-pool it replaced pixel for pixel.
    inside = torch.zeros(64, 64, dtype=torch.bool)
    inside[10:30, 25:50] = True
    full = F.max_pool2d(inside.float()[None, None], 2 * 6 + 1, stride=1, padding=6)[0, 0] > 0.5
    assert torch.equal(_ring(inside, 6), full & ~inside)
    # The ring is built once for a still mask; a mask that moves must not reuse the first frame's.
    plate, render, mask = _scene(frames=2)
    moved = mask.clone()
    moved[1] = torch.roll(mask[1], shifts=(0, 12), dims=(0, 1))
    both = grade_match(plate, render, moved, 6)
    alone = grade_match(plate[1:], render[1:], moved[1:], 6)
    assert torch.allclose(both[1], alone[0])


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} passed")
