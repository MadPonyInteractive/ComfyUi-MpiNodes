# MoGe v1 panorama depth: split an equirect into 12 icosahedron views, run MoGe on each, merge the
# log-distance gradients back into one equirect by least squares. From MoGe's
# scripts/infer_panorama.py (MIT, Copyright (c) Microsoft Corporation) as modified by Matrix-3D
# (MIT, Copyright (c) 2025 Skywork); notices in LICENSE beside this file. Changed for
# ComfyUi-MpiNodes (Mad Pony Interactive, 2026): the CLI, file output and mesh export are removed,
# utils3d calls go to ./utils3d.py, and `panorama_depth` runs the whole pass in-process.
import cv2
import numpy as np
import torch
from scipy.ndimage import convolve
from scipy.sparse import csr_array, vstack
from scipy.sparse.linalg import lsmr

from .utils3d import (extrinsics_look_at_np, icosahedron_np, image_uv_np, intrinsics_from_fov_xy_np,
                      intrinsics_to_fov_np, project_cv_np, unproject_cv_np, uv_to_pixel_np)


def get_panorama_cameras(fov_x=90., fov_y=90.):
    vertices = icosahedron_np()
    intrinsics = intrinsics_from_fov_xy_np(np.deg2rad(fov_x), np.deg2rad(fov_y))
    extrinsics = extrinsics_look_at_np([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)


def spherical_uv_to_directions(uv):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)


def directions_to_spherical_uv(directions):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(directions[..., 2]) / np.pi
    return np.stack([u, v], axis=-1)


def split_panorama_image(image, extrinsics, intrinsics, resolution):
    height, width = image.shape[:2]
    uv = image_uv_np(width=resolution, height=resolution)
    views = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(unproject_cv_np(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = uv_to_pixel_np(spherical_uv, width=width, height=height).astype(np.float32)
        views.append(cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR))
    return views


def poisson_equation(width, height, wrap_x=False, wrap_y=False):
    grid_index = np.arange(height * width).reshape(height, width)
    grid_index = np.pad(grid_index, ((0, 0), (1, 1)), mode='wrap' if wrap_x else 'edge')
    grid_index = np.pad(grid_index, ((1, 1), (0, 0)), mode='wrap' if wrap_y else 'edge')

    data = np.array([[-4, 1, 1, 1, 1]], dtype=np.float32).repeat(height * width, axis=0).reshape(-1)
    indices = np.stack([
        grid_index[1:-1, 1:-1],
        grid_index[:-2, 1:-1],         # up
        grid_index[2:, 1:-1],          # down
        grid_index[1:-1, :-2],         # left
        grid_index[1:-1, 2:]           # right
    ], axis=-1).reshape(-1)
    indptr = np.arange(0, height * width * 5 + 1, 5)
    return csr_array((data, indices, indptr), shape=(height * width, height * width))


def grad_equation(width, height, wrap_x=False, wrap_y=False):
    grid_index = np.arange(width * height).reshape(height, width)
    if wrap_x:
        grid_index = np.pad(grid_index, ((0, 0), (0, 1)), mode='wrap')
    if wrap_y:
        grid_index = np.pad(grid_index, ((0, 1), (0, 0)), mode='wrap')

    data = np.concatenate([
        np.concatenate([
            np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),        # x[i,j]
            -np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),       # x[i,j-1]
        ], axis=1).reshape(-1),
        np.concatenate([
            np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),        # x[i,j]
            -np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),       # x[i-1,j]
        ], axis=1).reshape(-1),
    ])
    indices = np.concatenate([
        np.concatenate([
            grid_index[:, :-1].reshape(-1, 1),
            grid_index[:, 1:].reshape(-1, 1),
        ], axis=1).reshape(-1),
        np.concatenate([
            grid_index[:-1, :].reshape(-1, 1),
            grid_index[1:, :].reshape(-1, 1),
        ], axis=1).reshape(-1),
    ])
    indptr = np.arange(0, grid_index.shape[0] * (grid_index.shape[1] - 1) * 2 + (grid_index.shape[0] - 1) * grid_index.shape[1] * 2 + 1, 2)
    return csr_array((data, indices, indptr), shape=(grid_index.shape[0] * (grid_index.shape[1] - 1) + (grid_index.shape[0] - 1) * grid_index.shape[1], height * width))


def merge_panorama_depth(width, height, distance_maps, pred_masks, extrinsics, intrinsics):
    if max(width, height) > 256:
        panorama_depth_init, _ = merge_panorama_depth(width // 2, height // 2, distance_maps, pred_masks, extrinsics, intrinsics)
        panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), cv2.INTER_LINEAR)
    else:
        panorama_depth_init = None

    uv = image_uv_np(width=width, height=height)
    spherical_directions = spherical_uv_to_directions(uv)

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = project_cv_np(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)

        projected_pixels = uv_to_pixel_np(np.clip(projected_uv, 0, 1), width=distance_maps[i].shape[1], height=distance_maps[i].shape[0]).astype(np.float32)

        log_splitted_distance = np.log(distance_maps[i])
        panorama_log_distance_map = np.where(projection_valid_mask, cv2.remap(log_splitted_distance, projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        panorama_pred_mask = projection_valid_mask & (cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)

        # gradient map
        padded = np.pad(panorama_log_distance_map, ((0, 0), (0, 1)), mode='wrap')
        grad_x, grad_y = padded[:, :-1] - padded[:, 1:], padded[:-1, :] - padded[1:, :]

        padded = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')
        mask_x, mask_y = padded[:, :-1] & padded[:, 1:], padded[:-1, :] & padded[1:, :]

        panorama_log_distance_grad_maps.append((grad_x, grad_y))
        panorama_grad_masks.append((mask_x, mask_y))

        # laplacian map
        padded = np.pad(panorama_log_distance_map, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        laplacian = convolve(padded, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1]

        padded = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        mask = convolve(padded.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1] == 5

        panorama_log_distance_laplacian_maps.append(laplacian)
        panorama_laplacian_masks.append(mask)
        panorama_pred_masks.append(panorama_pred_mask)

    panorama_log_distance_grad_x = np.stack([grad_map[0] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_log_distance_grad_y = np.stack([grad_map[1] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_grad_mask_x = np.stack([mask_map[0] for mask_map in panorama_grad_masks], axis=0)
    panorama_grad_mask_y = np.stack([mask_map[1] for mask_map in panorama_grad_masks], axis=0)

    panorama_log_distance_grad_x = np.sum(panorama_log_distance_grad_x * panorama_grad_mask_x, axis=0) / np.sum(panorama_grad_mask_x, axis=0).clip(1e-3)
    panorama_log_distance_grad_y = np.sum(panorama_log_distance_grad_y * panorama_grad_mask_y, axis=0) / np.sum(panorama_grad_mask_y, axis=0).clip(1e-3)

    panorama_laplacian_maps = np.stack(panorama_log_distance_laplacian_maps, axis=0)
    panorama_laplacian_masks = np.stack(panorama_laplacian_masks, axis=0)
    panorama_laplacian_map = np.sum(panorama_laplacian_maps * panorama_laplacian_masks, axis=0) / np.sum(panorama_laplacian_masks, axis=0).clip(1e-3)

    grad_x_mask = np.any(panorama_grad_mask_x, axis=0).reshape(-1)
    grad_y_mask = np.any(panorama_grad_mask_y, axis=0).reshape(-1)
    grad_mask = np.concatenate([grad_x_mask, grad_y_mask])
    laplacian_mask = np.any(panorama_laplacian_masks, axis=0).reshape(-1)

    # Solve overdetermined system
    A = vstack([
        grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask],
        poisson_equation(width, height, wrap_x=True, wrap_y=False)[laplacian_mask],
    ])
    b = np.concatenate([
        panorama_log_distance_grad_x.reshape(-1)[grad_x_mask],
        panorama_log_distance_grad_y.reshape(-1)[grad_y_mask],
        panorama_laplacian_map.reshape(-1)[laplacian_mask]
    ])
    x, *_ = lsmr(
        A, b,
        atol=1e-5, btol=1e-5,
        x0=np.log(panorama_depth_init).reshape(-1) if panorama_depth_init is not None else None,
        show=False,
    )

    panorama_depth = np.exp(x).reshape(height, width).astype(np.float32)
    panorama_mask = np.any(panorama_pred_masks, axis=0)
    return panorama_depth, panorama_mask


@torch.no_grad()
def panorama_depth(model, image_rgb, device, resolution_level=9, merge_long=1920, merge_short=960,
                   batch_size=4):
    """MoGe equirect distance for an [H, W, 3] uint8 RGB pano -> (depth [H, W] float32, valid [H, W] bool).

    The 12 views are 768 px, 100 deg; the merge (the dominant cost, CPU) runs at most
    merge_long x merge_short and is resized back to H x W."""
    H, W = image_rgb.shape[:2]
    ext, intr = get_panorama_cameras(fov_x=100.0, fov_y=100.0)
    views = split_panorama_image(image_rgb, ext, intr, 768)

    dist_maps, masks = [], []
    for i in range(0, len(views), batch_size):
        batch = np.stack(views[i:i + batch_size]) / 255.0
        img_t = torch.tensor(batch, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        fov_x, _ = np.rad2deg(intrinsics_to_fov_np(np.array(intr[i:i + batch_size])))
        fov_x = torch.tensor(fov_x, dtype=torch.float32, device=device)
        out = model.infer(img_t, fov_x=fov_x, apply_mask=False, resolution_level=int(resolution_level))
        dist_maps.extend(list(out["points"].norm(dim=-1).cpu().numpy()))
        masks.extend(list(out["mask"].cpu().numpy()))

    depth, mask = merge_panorama_depth(min(merge_long, W), min(merge_short, H), dist_maps, masks, ext, intr)
    depth = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return depth, mask
