# MoGe v1 geometry helpers (MIT, Copyright (c) Microsoft Corporation; notice in LICENSE beside
# this file), from moge/utils/geometry_torch.py and geometry_numpy.py. Changed for
# ComfyUi-MpiNodes (Mad Pony Interactive, 2026): only what MoGeModel.infer calls is kept, and
# solve_optimal_shift returns 0 for an empty tile (a near-empty mask tile of a flat sky or ground
# view left least_squares with fewer residuals than variables and crashed the whole depth pass).
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F


def normalized_view_plane_uv(width, height, aspect_ratio=None, dtype=None, device=None):
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=-1)


def solve_optimal_focal_shift(uv, xyz):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv, xy, z, shift):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return (f * xy_proj - uv).ravel()

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='trf')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
    return optim_shift, optim_focal


def solve_optimal_shift(uv, xyz, focal):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)
    if z.size == 0:
        return np.float32(0.0)

    def fn(uv, xy, z, shift):
        xy_proj = xy / (z + shift)[:, None]
        return (focal * xy_proj - uv).ravel()

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    return solution['x'].squeeze().astype(np.float32)


def recover_focal_shift(points, mask=None, focal=None, downsample_size=(64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.
    Assumes the optical center is at the map centre, no distortion, isometric x and y.

    points (..., H, W, 3), mask (..., H, W) optional, focal (...) optional, relative to the half
    diagonal. Returns (focal, shift), each of shape (...).
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0

    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])
    return optim_focal, optim_shift
