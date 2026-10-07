# The utils3d functions MoGe v1 inference and its panorama merge call, copied from
# utils3d (MIT, Copyright (c) 2022 EasternJournalist, https://github.com/EasternJournalist/utils3d;
# notice in LICENSE beside this file). Changed for ComfyUi-MpiNodes (Mad Pony Interactive, 2026):
# only these functions are kept, utils3d's @batched decorator is dropped (every call site here
# broadcasts with plain matmul), and numpy/torch twins live in one module with _np/_torch names.
import numpy as np
import torch


# ---- numpy -----------------------------------------------------------------------------------

def icosahedron_np():
    A = (1 + 5 ** 0.5) / 2
    vertices = np.array([
        [0, 1, A], [0, -1, A], [0, 1, -A], [0, -1, -A],
        [1, A, 0], [-1, A, 0], [1, -A, 0], [-1, -A, 0],
        [A, 0, 1], [A, 0, -1], [-A, 0, 1], [-A, 0, -1]
    ], dtype=np.float32)
    return vertices


def intrinsics_from_focal_center_np(fx, fy, cx, cy, dtype=np.float32):
    if any(isinstance(x, np.ndarray) for x in (fx, fy, cx, cy)):
        dtype = np.result_type(fx, fy, cx, cy)
    fx, fy, cx, cy = np.broadcast_arrays(fx, fy, cx, cy)
    ret = np.zeros((*fx.shape, 3, 3), dtype=dtype)
    ret[..., 0, 0] = fx
    ret[..., 1, 1] = fy
    ret[..., 0, 2] = cx
    ret[..., 1, 2] = cy
    ret[..., 2, 2] = 1.
    return ret


def intrinsics_from_fov_xy_np(fov_x, fov_y):
    """Normalized OpenCV intrinsics from both fields of view (radians)."""
    fx = 1 / (2 * np.tan(fov_x / 2))
    fy = 1 / (2 * np.tan(fov_y / 2))
    return intrinsics_from_focal_center_np(fx, fy, 0.5, 0.5)


def intrinsics_to_fov_np(intrinsics):
    fov_x = 2 * np.arctan(0.5 / intrinsics[..., 0, 0])
    fov_y = 2 * np.arctan(0.5 / intrinsics[..., 1, 1])
    return fov_x, fov_y


def extrinsics_look_at_np(eye, look_at, up):
    """OpenCV extrinsics [..., 4, 4]; eye / look_at / up broadcast against each other."""
    eye, look_at, up = np.broadcast_arrays(np.asarray(eye), np.asarray(look_at), np.asarray(up))
    z = look_at - eye
    x = np.cross(-up, z)
    y = np.cross(z, x)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    R = np.stack([x, y, z], axis=-2)
    t = -np.matmul(R, eye[..., None])
    bottom = np.broadcast_to(np.array([0., 0., 0., 1.], dtype=R.dtype), (*R.shape[:-2], 1, 4))
    return np.concatenate([np.concatenate([R, t], axis=-1), bottom], axis=-2)


def image_uv_np(height, width, dtype=np.float32):
    u = np.linspace(0.5 / width, (width - 0.5) / width, width, dtype=dtype)
    v = np.linspace(0.5 / height, (height - 0.5) / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing='xy')
    return np.stack([u, v], axis=2)


def uv_to_pixel_np(uv, width, height):
    return uv * np.stack([width, height], axis=-1).astype(uv.dtype) - 0.5


def project_cv_np(points, extrinsics=None, intrinsics=None):
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    if extrinsics is not None:
        points = points @ extrinsics.swapaxes(-1, -2)
    points = points[..., :3] @ intrinsics.swapaxes(-1, -2)
    with np.errstate(divide='ignore', invalid='ignore'):
        uv_coord = points[..., :2] / points[..., 2:]
    return uv_coord, points[..., 2]


def unproject_cv_np(uv_coord, depth=None, extrinsics=None, intrinsics=None):
    points = np.concatenate([uv_coord, np.ones_like(uv_coord[..., :1])], axis=-1)
    points = points @ np.linalg.inv(intrinsics).swapaxes(-1, -2)
    if depth is not None:
        points = points * depth[..., None]
    if extrinsics is not None:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
        points = (points @ np.linalg.inv(extrinsics).swapaxes(-1, -2))[..., :3]
    return points


# ---- torch -----------------------------------------------------------------------------------

def intrinsics_from_focal_center_torch(fx, fy, cx, cy):
    N = fx.shape[0]
    zeros = torch.zeros(N, dtype=fx.dtype, device=fx.device)
    ones = torch.ones(N, dtype=fx.dtype, device=fx.device)
    if not torch.is_tensor(cx):
        cx, cy = zeros + cx, zeros + cy
    return torch.stack([fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim=-1).unflatten(-1, (3, 3))


def image_uv_torch(height, width, device=None, dtype=None):
    u = torch.linspace(0.5 / width, (width - 0.5) / width, width, device=device, dtype=dtype)
    v = torch.linspace(0.5 / height, (height - 0.5) / height, height, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=-1)


def unproject_cv_torch(uv_coord, depth, intrinsics):
    points = torch.cat([uv_coord, torch.ones_like(uv_coord[..., :1])], dim=-1)
    points = points @ torch.inverse(intrinsics).transpose(-2, -1)
    return points * depth[..., None]
