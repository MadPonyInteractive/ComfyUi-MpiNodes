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


def lift_depth(zm, zc, erode=9, grow=5, edge_rtol=0.05):
    """zm: MoGe depth of the fill [H, W]; zc: the scene's camera z [H, W], 0 where nothing is known.

    Fits z = a * zm + b by least squares on the known pixels (eroded `erode` px off the holes), then
    keeps the hole pixels grown by `grow` px, minus depth edges and invalid depth.
    Returns (depth [H, W] float32 with 0 = not kept, a, b, median relative fit error)."""
    known = torch.isfinite(zc) & (zc > 0)
    fit = _erode(known & torch.isfinite(zm) & (zm > 0), erode)
    if int(fit.sum()) < 2:
        raise ValueError("lift: fewer than 2 known pixels to fit the fill's depth against - "
                         "the known-depth map is empty or does not match the fill")
    A = torch.stack([zm[fit], torch.ones_like(zm[fit])], 1).double()
    ab = torch.linalg.lstsq(A, zc[fit][:, None].double()).solution[:, 0].float()
    za = ab[0] * zm + ab[1]
    rel = float(((za[fit] - zc[fit]).abs() / zc[fit]).median())
    za_ok = torch.isfinite(za) & (za > 0)
    za_safe = torch.where(za_ok, za, torch.ones_like(za)).clamp_min(1e-4)
    keep = _dilate(~known, grow) & za_ok & ~depth_edges(za_safe, edge_rtol)
    return torch.where(keep, za_safe, torch.zeros_like(za_safe)), float(ab[0]), float(ab[1]), rel
