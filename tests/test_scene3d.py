"""CPU checks for scene3d/ (the maths behind scene.py's nodes) on tiny arrays. No ComfyUI, no GPU, no
weights. Run with the engine's python (it has torch, numpy, scipy, cv2):

    G:/ComfyUi/python_embeded/python.exe tests/test_scene3d.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene3d.lift import depth_edges, ground_depth, lift_depth  # noqa: E402
from scene3d.moge.panorama import (directions_to_spherical_uv, get_panorama_cameras,  # noqa: E402
                                   merge_panorama_depth, spherical_uv_to_directions,
                                   split_panorama_image)
from scene3d.wrap import pad_at_scale, wrap_crop, wrap_cut_merge, wrap_pad, wrap_soften  # noqa: E402


def test_wrap_pad_crop_round_trip():
    img = torch.rand(2, 8, 16, 3)
    p = wrap_pad(img, 3)
    assert p.shape == (2, 8, 22, 3)
    assert torch.equal(p[:, :, :3], img[:, :, -3:]) and torch.equal(p[:, :, -3:], img[:, :, :3])
    assert torch.equal(wrap_crop(p, 16, 3), img)
    up = p.repeat_interleave(4, dim=2)  # a 4x upscaler, nearest
    assert pad_at_scale(up.shape[2], 16, 3) == 12
    assert torch.equal(wrap_crop(up, 16, 3), img.repeat_interleave(4, dim=2))
    assert torch.equal(wrap_pad(img, 0), img)


def test_wrap_soften_touches_only_the_wrap():
    h, w, radius = 32, 256, 16
    ramp = np.tile(np.linspace(0, 255, w, dtype=np.float32)[None, :, None], (h, 1, 3))  # 255 jump at the wrap
    out = wrap_soften(ramp, sigma=4, radius=radius, threshold=0)
    assert np.array_equal(out[:, radius:w - radius], ramp[:, radius:w - radius])
    assert abs(out[0, 0, 0] - out[0, -1, 0]) < 0.25 * abs(ramp[0, 0, 0] - ramp[0, -1, 0])


def test_wrap_soften_threshold_keeps_hard_edges():
    h, w = 64, 256
    sky = np.full((h, w, 3), 180.0, np.float32)
    sky[:, :w // 2] += 2.0  # a faint step at the wrap: flat sky, softens
    soft = wrap_soften(sky, sigma=4, radius=16, threshold=6)
    assert abs(soft[h // 2, 0, 0] - soft[h // 2, -1, 0]) < 1.0
    cloud = np.full((h, w, 3), 180.0, np.float32)
    cloud[:, :w // 2] += 60.0  # a hard edge the wrap cuts: stays for the seam pass to join
    kept = wrap_soften(cloud, sigma=4, radius=16, threshold=6)
    assert np.abs(kept[h // 2] - cloud[h // 2]).max() < 0.5


def test_wrap_cut_merge_identity_and_tone():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (24, 128, 3)).astype(np.float32)
    O = 32
    padded = np.concatenate([img[:, -O:], img, img[:, :O]], axis=1)
    assert np.allclose(wrap_cut_merge(padded, O), img, atol=1e-3)  # two identical renders -> the core
    shifted = padded.copy()
    shifted[:, O + 128:] += 10.0  # the right pad's render came out 10 levels brighter
    out = wrap_cut_merge(shifted, O)
    assert out.shape == img.shape
    assert np.allclose(out[:, 0], img[:, 0] + 10.0, atol=1e-2)  # meets col W-1's render at the wrap
    assert np.allclose(out[:, O - 1], img[:, O - 1], atol=1e-2)  # and the core where the overlap ends
    assert np.array_equal(out[:, O:], img[:, O:])


def test_depth_edges():
    d = torch.ones(10, 10)
    d[:, 5:] = 2.0
    e = depth_edges(d, 0.05)
    assert e[:, 4:6].all() and not e[:, :3].any() and not e[:, 7:].any()


def test_lift_depth_recovers_scale_and_shift():
    h, w = 48, 64
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    truth = 3.0 + 0.02 * xx + 0.01 * yy  # a smooth tilted wall
    zm = (truth - 0.3) / 2.5             # MoGe's depth: right shape, wrong scale and shift
    zc = truth.clone()
    zc[:, w // 2:] = 0.0                 # the scene knows the left half only
    depth, a, b, rel = lift_depth(zm, zc)
    assert abs(a - 2.5) < 1e-3 and abs(b - 0.3) < 1e-3 and rel < 1e-5
    kept = depth > 0
    assert kept[:, w // 2:].all() and not kept[:, :w // 2 - 2].any()  # the holes, grown 2 px
    assert torch.allclose(depth[kept], truth[kept], rtol=1e-4)
    try:
        lift_depth(zm, torch.zeros(h, w))
        raise AssertionError("an empty known-depth map must raise")
    except ValueError:
        pass


def test_lift_depth_negative_z_fits_and_keeps():
    """Inside a house: the walls seen from behind are known for the fit but replaced by the fill."""
    h, w = 48, 96
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    truth = 3.0 + 0.02 * xx + 0.01 * yy
    zm = (truth - 0.3) / 2.5
    zc = truth.clone()
    zc[:, w // 3:2 * w // 3] *= -1.0     # back-faced walls: fit here, keep here
    zc[:, 2 * w // 3:] = 0.0             # holes: keep here
    depth, a, b, rel = lift_depth(zm, zc)
    assert abs(a - 2.5) < 1e-3 and abs(b - 0.3) < 1e-3 and rel < 1e-5
    kept = depth > 0
    assert kept[:, w // 3:].all() and not kept[:, :w // 3 - 2].any()
    assert torch.allclose(depth[kept], truth[kept], rtol=1e-4)
    # Only the walls known (no view out): the fit still lands, everything is kept.
    depth, a, b, _ = lift_depth(zm, -truth)
    assert abs(a - 2.5) < 1e-3 and abs(b - 0.3) < 1e-3 and (depth > 0).all()
    # The wrong scale on the walls must move the fit: they really are in it.
    zc2 = zc.clone()
    zc2[:, w // 3:2 * w // 3] *= 1.5
    assert abs(lift_depth(zm, zc2)[1] - 2.5) > 0.1


def test_lift_depth_one_depth_known_fits_scale_only():
    """MPI-623 window, build view 3: the only known pixels were one wall strip seen edge-on, all at one
    depth, so the slope was noise - it came out negative and the back wall, put behind the camera, was
    dropped. A known set with no depth spread fits scale only."""
    h, w = 48, 96
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    truth = torch.full((h, w), 3.0)                  # the back wall
    truth[:, :12] = 1.0                              # the near side wall's strip
    zm = truth / 2.5
    zm[:, :12] -= 0.002 * yy[:, :12] / h             # MoGe's noise on the strip, against the scene's
    zc = torch.zeros(h, w)
    zc[:, :12] = 1.0 + 0.004 * yy[:, :12] / h        # the scene knows the strip only, at ~one depth
    A = torch.stack([zm[:, :3].flatten(), torch.ones(h * 3)], 1).double()
    assert torch.linalg.lstsq(A, zc[:, :3].flatten()[:, None].double()).solution[0, 0] < 0  # the trap
    depth, a, b, _ = lift_depth(zm, zc)
    assert abs(a - 2.5) < 0.02 and b == 0.0
    kept = depth > 0
    assert kept[:, 13:].all()                        # the back wall is kept (col 12: the depth edge)...
    assert torch.allclose(depth[:, 13:], truth[:, 13:], rtol=0.01)  # ...where it stands


def test_lift_depth_scale_only_on_either_sign_of_noise():
    """Each half of the guard alone: one depth known with a steep POSITIVE noise slope (a = 100 threw
    the back wall 27x too far), and a known set WITH depth spread whose slope came out negative."""
    h, w = 48, 96
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    truth = torch.full((h, w), 3.0)
    truth[:, :12] = 1.0
    zm = truth / 2.5
    zm[:, :12] += 0.0004 * yy[:, :12] / h            # with the scene, but far too little
    zc = torch.zeros(h, w)
    zc[:, :12] = 1.0 + 0.04 * yy[:, :12] / h
    depth, a, b, _ = lift_depth(zm, zc)
    assert b == 0.0 and torch.allclose(depth[:, 13:], truth[:, 13:], rtol=0.03)
    zm2 = 1.0 - 0.005 * xx                           # spread (p90/p10 1.1) but against the scene
    zc2 = torch.where(xx < w // 3, 2.0 + 0.02 * xx, torch.zeros_like(xx))
    _, a, b, _ = lift_depth(zm2, zc2)
    assert a > 0 and b == 0.0


def test_ground_depth_level_camera():
    """A camera 0.5 above level ground, looking ahead: the bottom rows meet it, the top rows never do."""
    h, w = 40, 60
    g = ground_depth([0, 1, 0], 0.5, h, w, 90.0)  # f = 30 px
    v = h - 1
    assert abs(float(g[v, w // 2]) - 0.5 * 30 / (v + 0.5 - h / 2)) < 1e-4
    assert torch.isinf(g[: h // 2]).all()


def test_lift_depth_floor_lifts_what_sank_under_the_ground():
    """MPI-623 behind_well: the fit, made on far known pixels, sank the near floor 2.4-6x. With the
    ground plane, what lies under it moves onto it; what stands above it is untouched."""
    h, w = 200, 300                       # big enough that the floor is no depth edge row to row
    floor = ground_depth([0, 1, 0], 0.5, h, w, 90.0)
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    wall = 4.0 + 0.02 * xx                # a tilted far wall, so the fit has a slope to find
    truth = torch.minimum(floor, wall)
    zm = truth.clone()
    zm[yy >= h - 40] *= 3.0                # MoGe's near floor, as the live fit placed it: 3x too deep
    zc = truth.clone()
    zc[h // 2:] = 0.0                     # the floor is a hole: the fit sees the far wall only
    depth, a, b, _ = lift_depth(zm, zc, floor=floor)
    near = (yy >= h - 40) & (depth > 0)
    assert near.any() and torch.allclose(depth[near], floor[near], rtol=1e-5)  # on the ground, not under it
    above = (depth > 0) & (yy < h - 40)
    assert torch.allclose(depth[above], (a * zm + b)[above], rtol=1e-5)       # nothing above it moved
    sunk, _, _, _ = lift_depth(zm, zc)
    assert (sunk[near] > floor[near] * 2).all()                                # without it: under the floor


def test_panorama_directions_round_trip():
    uv = np.random.default_rng(1).uniform(0.01, 0.99, (100, 2))
    back = directions_to_spherical_uv(spherical_uv_to_directions(uv))
    assert np.allclose(back, uv, atol=1e-6)


def test_panorama_cameras_and_split():
    ext, intr = get_panorama_cameras(fov_x=100.0, fov_y=100.0)
    assert ext.shape == (12, 4, 4) and len(intr) == 12
    R = ext[:, :3, :3]
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-5)
    views = split_panorama_image(np.full((32, 64, 3), 77, np.uint8), ext, intr, 16)
    assert len(views) == 12 and all(v.shape == (16, 16, 3) for v in views)
    # upstream remaps with a black border, so a view across the wrap column dims a few pixels there
    assert all((v == 77).mean() > 0.9 for v in views)


def test_merge_panorama_depth_of_a_constant_room_is_flat():
    ext, intr = get_panorama_cameras(fov_x=100.0, fov_y=100.0)
    dist = [np.full((24, 24), 5.0, np.float32) for _ in range(12)]
    masks = [np.ones((24, 24), bool) for _ in range(12)]
    depth, valid = merge_panorama_depth(64, 32, dist, masks, ext, intr)
    assert depth.shape == (32, 64) and valid.all()
    assert depth.std() / depth.mean() < 1e-3  # zero gradients everywhere -> one depth


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} passed")
