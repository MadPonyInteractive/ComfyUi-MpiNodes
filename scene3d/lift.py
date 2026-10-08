"""Lift a fill into the scene: fit MoGe's relative depth of the fill to the camera z the scene already
knows, keep the hole pixels. Pure torch, ported from the MPI-623 bench (chain.py), so the tests run
without ComfyUI."""
import torch
import torch.nn.functional as F


def depth_edges(depth, rtol):
    """[H, W] bool: pixels whose 3x3 neighbourhood spans more than rtol of their depth."""
    d = depth[None, None]
    diff = F.max_pool2d(d, 3, 1, 1) + F.max_pool2d(-d, 3, 1, 1)
    return (diff[0, 0] / depth.clamp_min(1e-6)) > rtol


def _dilate(mask, k):
    return F.max_pool2d(mask[None, None].float(), k, 1, k // 2)[0, 0] > 0


def _erode(mask, k):  # the border never erodes, as cv2.erode's default
    return ~_dilate(~mask, k)


def ground_depth(normal, d, h, w, fov_x):
    """[H, W]: the camera z where each pixel's ray meets the plane n . X = d (camera frame, OpenCV: x
    right, y down, z ahead; a pinhole of horizontal field `fov_x` degrees centred on the image), inf
    where the ray never reaches it."""
    f = (w / 2) / torch.tan(torch.deg2rad(torch.tensor(float(fov_x))) / 2)
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    n = [float(v) for v in normal]
    nr = n[0] * (xx + 0.5 - w / 2) / f + n[1] * (yy + 0.5 - h / 2) / f + n[2]
    return torch.where(nr > 1e-6, float(d) / nr.clamp_min(1e-6), torch.full_like(nr, float("inf")))


def lift_depth(zm, zc, erode=9, grow=5, edge_rtol=0.05, floor=None):
    """zm: MoGe depth of the fill [H, W]; zc: the scene's camera z [H, W]. Positive = known, not kept;
    0 = unknown, kept; negative = known at -zc for the fit AND kept (a surface the fill replaces, e.g.
    walls seen from behind by a camera inside the house).

    Fits z = a * zm + b by least squares on the known pixels (eroded `erode` px off the unknown ones),
    then keeps the pixels that are not positive-known, grown by `grow` px, minus depth edges and
    invalid depth. `floor` ([H, W], `ground_depth`) is the ground: a kept pixel the fit puts past it,
    under the floor, moves onto it - the fit is made on mid/far known pixels, and extrapolated to the
    near floor it sank it 2.4-6x (MPI-623, behind_well). ponytail: a real pit under the ground (a
    well shaft) flattens too. Returns (depth [H, W] float32 with 0 = not kept, a, b, median relative
    fit error)."""
    known = torch.isfinite(zc) & (zc != 0)
    zk = zc.abs()
    fit = _erode(known & torch.isfinite(zm) & (zm > 0), erode)
    if int(fit.sum()) < 2:
        raise ValueError("lift: fewer than 2 known pixels to fit the fill's depth against - "
                         "the known-depth map is empty or does not match the fill")
    A = torch.stack([zm[fit], torch.ones_like(zm[fit])], 1).double()
    ab = torch.linalg.lstsq(A, zk[fit][:, None].double()).solution[:, 0].float()
    za = ab[0] * zm + ab[1]
    rel = float(((za[fit] - zk[fit]).abs() / zk[fit]).median())
    za_ok = torch.isfinite(za) & (za > 0)
    za_safe = torch.where(za_ok, za, torch.ones_like(za)).clamp_min(1e-4)
    keep = _dilate(~(known & (zc > 0)), grow) & za_ok & ~depth_edges(za_safe, edge_rtol)
    if floor is not None:
        za_safe = torch.where(za_safe > floor, floor, za_safe)
    return torch.where(keep, za_safe, torch.zeros_like(za_safe)), float(ab[0]), float(ab[1]), rel
