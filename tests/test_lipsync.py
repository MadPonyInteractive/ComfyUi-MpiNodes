"""CPU checks for lipsync.py (MpiFaceTrackCrop / MpiFaceTrackPaste) on small synthetic clips. No ComfyUI, no GPU.
Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_lipsync.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lipsync import lab_to_rgb, rgb_to_lab, track_crop, track_paste  # noqa: E402

N, H, W = 9, 300, 400


def _clip(face_px=(40, 60), grow=0):
    """A textured plate with a face box that drifts right (and grows by `grow` px a frame), lips in its lower part."""
    torch.manual_seed(0)
    img = torch.rand(N, H, W, 3) * 0.5 + 0.25
    face = torch.zeros(N, H, W)
    lips = torch.zeros(N, H, W)
    for i in range(N):
        fw, fh = face_px[0] + grow * i, face_px[1] + grow * i
        x0, y0 = 40 + 3 * i, 20
        face[i, y0:y0 + fh, x0:x0 + fw] = 1
        lips[i, y0 + int(fh * 0.7):y0 + int(fh * 0.8), x0 + fw // 4:x0 + 3 * fw // 4] = 1
    return img, face, lips


def test_crop_follows_the_face_and_masks_only_the_lower_face():
    img, face, lips = _clip()
    crop, mask, info = track_crop(img, face, lips, size=128, zoom=False)
    assert crop.shape == (N, 128, 128, 3) and mask.shape == (N, 128, 128)
    assert len(set(info["sides"])) == 1  # zoom off: one side
    xs = [o[0] for o in info["origins"]]
    assert xs == sorted(xs) and xs[-1] > xs[0]  # the square moves with the face
    for i in range(N):
        rows = torch.nonzero(mask[i].any(1)).flatten()
        assert len(rows) > 0
        # the face's top (crop row of y0=20) is above the mask: the eyes are never re-made
        top_c = (20 - info["origins"][i][1]) * 128 / info["sides"][i]
        assert rows[0] > top_c + 5, (i, int(rows[0]), top_c)


def test_zoom_keeps_a_growing_face_the_same_size_in_the_crop():
    img, face, lips = _clip(face_px=(20, 30), grow=10)
    _, mask, info = track_crop(img, face, lips, size=128, zoom=True)
    assert info["sides"][-1] > info["sides"][0] * 1.5
    # the masked area, in crop pixels, stays about the same through the clip
    a0, a1 = float(mask[4].sum()), float(mask[-2].sum())  # past the 96 px side floor
    assert 0.6 < a1 / a0 < 1.6, (a0, a1)


def test_hard_cut_lowers_the_mask_top():
    img, face, lips = _clip()
    _, soft, _ = track_crop(img, face, lips, size=128, zoom=False, top_cut=0.1)
    _, hard, _ = track_crop(img, face, lips, size=128, zoom=False, top_cut=0.1, hard_cut=True)
    top = lambda m: int(torch.nonzero(m.any(1)).flatten()[0])
    assert top(hard[4]) > top(soft[4])


def test_mouth_objects_outside_the_face_are_ignored():
    img, face, lips = _clip()
    car = lips.clone()
    car[:, 100:110, 150:190] = 1  # a second tracked "mouth" object far from the face
    _, a, _ = track_crop(img, face, lips, size=128, zoom=False)
    _, b, _ = track_crop(img, face, car, size=128, zoom=False)
    assert torch.equal(a, b)


def test_no_face_raises():
    img, face, lips = _clip()
    try:
        track_crop(img, face * 0, lips, size=128)
    except ValueError as e:
        assert "no face" in str(e)
    else:
        raise AssertionError("an empty face mask must fail loudly")


def test_lab_round_trip():
    x = torch.rand(1000, 3)
    assert (lab_to_rgb(rgb_to_lab(x)) - x).abs().max() < 1e-3


def test_paste_keeps_outside_and_grades_the_inside_back():
    img, face, lips = _clip()
    crop, mask, info = track_crop(img, face, lips, size=128, zoom=False)
    dark = (crop * 0.85).clamp(0, 1)  # what a video model hands back: the same face, darker
    out = track_paste(img, dark, mask, info, color_match=True)
    raw = track_paste(img, dark, mask, info, color_match=False)
    # outside the face's square nothing changes
    assert torch.equal(out[:, :, :20], img[:, :, :20])
    # inside the mask the matched paste sits with the source, the raw one is darker
    i = 4
    x, y, s = info["origins"][i][0], info["origins"][i][1], info["sides"][i]
    m = torch.nn.functional.interpolate(mask[i][None, None], size=(s, s))[0, 0] > 0.5
    src_l = rgb_to_lab(img[i, y:y + s, x:x + s])[m][:, 0].mean()
    assert abs(float(rgb_to_lab(out[i, y:y + s, x:x + s])[m][:, 0].mean() - src_l)) < 1.0
    assert float(src_l - rgb_to_lab(raw[i, y:y + s, x:x + s])[m][:, 0].mean()) > 3.0


def test_paste_keeps_the_shortest_count():
    img, face, lips = _clip()
    crop, mask, info = track_crop(img, face, lips, size=128, zoom=False)
    assert track_paste(img, crop[:7], mask, info).shape[0] == 7


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
