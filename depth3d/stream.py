"""Video Depth Anything inference for MpiDepthImage / MpiDepthVideo (Cubric Studio MPI-1605).

No ComfyUI imports here, so tests/ can load it alone. Upstream's offline `infer_video_depth` holds
every frame and every depth map of the clip in RAM; `DepthWindows` runs the SAME windows (32 frames,
22 new per window, keyframes 0/12/24-31 carried over, scale/shift aligned to the first window, the
8-frame seam blended) as a stream: frames go in as they decode, finished depth frames come out in
order, and at most two windows of frames are held. Depth is relative inverse depth (bigger = nearer),
at the network's resolution; the caller normalises it with the clip's own min/max, as upstream does.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vda.dinov2 import DINOv2
from .vda.dpt_temporal import DPTHeadTemporal

INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
INTERP_LEN = 8
STEP = INFER_LEN - OVERLAP  # 22 new frames per window after the first
ALIGN_LEN = OVERLAP - INTERP_LEN  # 2 keyframes fit each window's scale/shift
KF_ALIGN = KEYFRAMES[:ALIGN_LEN]  # [0, 12]

CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}
_LAYERS = {"vits": [2, 5, 8, 11], "vitb": [2, 5, 8, 11], "vitl": [4, 11, 17, 23]}
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class VideoDepthAnything(nn.Module):
    """Upstream's offline model (video_depth.py) without its inference loop."""

    def __init__(self, encoder="vits", num_frames=INFER_LEN):
        super().__init__()
        cfg = CONFIGS[encoder]
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.head = DPTHeadTemporal(self.pretrained.embed_dim, cfg["features"], False,
                                    out_channels=cfg["out_channels"], use_clstoken=False,
                                    num_frames=num_frames, pe="ape")

    def forward(self, x, encoder_batch=8):
        b, t, _c, h, w = x.shape
        # The backbone sees each frame alone, so running it 8 frames at a time gives the same
        # features as upstream's one 32-frame batch, with a quarter of its activations in VRAM.
        frames = x.flatten(0, 1)
        parts = [self.pretrained.get_intermediate_layers(frames[i:i + encoder_batch], _LAYERS[self.encoder], return_class_token=True)
                 for i in range(0, len(frames), encoder_batch)]
        feats = [tuple(torch.cat([p[layer][k] for p in parts]) for k in range(2)) for layer in range(len(parts[0]))]
        depth = self.head(feats, h // 14, w // 14, t)[0]
        depth = F.relu(F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True))
        return depth.squeeze(1).unflatten(0, (b, t))  # [B, T, H, W]


def encoder_of(state_dict):
    """vits / vitb / vitl, read off the patch embedding's width (384 / 768 / 1024)."""
    width = state_dict["pretrained.patch_embed.proj.weight"].shape[0]
    return {384: "vits", 768: "vitb", 1024: "vitl"}[width]


def net_size(h, w, input_size=518):
    """Upstream's Resize(keep_aspect_ratio, lower_bound, multiple of 14), with its >16:9 shrink."""
    ratio = max(h, w) / min(h, w)
    if ratio > 1.78:
        input_size = round(int(input_size * 1.777 / ratio) / 14) * 14
    scale = max(input_size / h, input_size / w)

    def fit(x):
        y = round(x / 14) * 14
        return y if y >= input_size else math.ceil(x / 14) * 14

    return fit(scale * h), fit(scale * w)


def prepare(frames, size):
    """[N, H, W, 3] floats in 0..1 -> [N, 3, h, w] normalised network input (bicubic, like cv2's)."""
    x = frames.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=size, mode="bicubic", align_corners=False).clamp(0, 1)
    mean = torch.tensor(_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def scale_and_shift(pred, target):
    """Least-squares s, b with s * pred + b ~ target (upstream compute_scale_and_shift_full)."""
    pred = pred.astype(np.float64).ravel()
    target = target.astype(np.float64).ravel()
    a00, a01, a11 = np.dot(pred, pred), pred.sum(), float(pred.size)
    b0, b1 = np.dot(pred, target), target.sum()
    det = a00 * a11 - a01 * a01
    if det == 0:
        return 1.0, 0.0
    return (a11 * b0 - a01 * b1) / det, (-a01 * b0 + a00 * b1) / det


def _blend(pre, post):
    """Upstream get_interpolate_frames: a linear cross-fade, pre at weight 1 to post at weight 1."""
    n = len(pre)
    return [pre[i] * (1 - i / (n - 1)) + post[i] * (i / (n - 1)) for i in range(n)]


class DepthWindows:
    """Feed frames with push(), take finished depth frames from what push()/finish() return.

    `infer(x)` runs the model on a [1, T, 3, h, w] window and returns [T, h, w] float32 numpy, so
    the device and dtype stay the caller's business (and tests can pass a fake).
    """

    def __init__(self, infer, size):
        self.infer, self.size = infer, size
        self.pending = []  # prepared input frames not yet in a window, [3, h, w] each
        self.prev = None  # the last window's input, [32, 3, h, w]
        self.windows = 0
        self.fed = 0  # real frames pushed
        self.emitted = 0
        self.tail = []  # aligned depth not final yet (the next window re-blends it)
        self.ref = None  # aligned depth of the alignment keyframes

    def push(self, frames):
        """frames: [N, H, W, 3] floats 0..1. Returns finished depth frames (may be empty)."""
        x = prepare(frames, self.size)
        self.pending.extend(x.unbind(0))
        self.fed += len(x)
        out = []
        while len(self.pending) >= (INFER_LEN if self.windows == 0 else STEP):
            out += self._window()
        return out

    def finish(self):
        """Pads the last window with the last frame, as upstream does, and returns the rest."""
        if self.fed == 0:
            return []
        out = []
        last = self.pending[-1] if self.pending else self.prev[-1]
        while self.windows < math.ceil(self.fed / STEP):
            need = INFER_LEN if self.windows == 0 else STEP
            self.pending += [last] * (need - len(self.pending))
            out += self._window()
        out += self._cut(self.tail)
        self.tail = []
        return out

    def _cut(self, frames):
        keep = max(0, min(len(frames), self.fed - self.emitted))
        self.emitted += keep
        return frames[:keep]

    def _window(self):
        if self.windows == 0:
            cur = torch.stack(self.pending[:INFER_LEN])
            del self.pending[:INFER_LEN]
        else:
            new = torch.stack(self.pending[:STEP])
            del self.pending[:STEP]
            cur = torch.cat([self.prev[KEYFRAMES], new])
        depth = list(self.infer(cur.unsqueeze(0)))
        self.prev = cur
        self.windows += 1

        if self.windows == 1:
            self.tail = depth
            self.ref = [depth[k] for k in KF_ALIGN]
        else:
            s, b = scale_and_shift(np.concatenate(depth[:ALIGN_LEN]), np.concatenate(self.ref))
            fit = [np.maximum(d * s + b, 0) for d in depth]
            self.tail[-INTERP_LEN:] = _blend(self.tail[-INTERP_LEN:], fit[ALIGN_LEN:OVERLAP])
            self.tail += fit[OVERLAP:]
            self.ref = self.ref[:1] + [fit[k] for k in KF_ALIGN[1:]]
        done, self.tail = self.tail[:-INTERP_LEN], self.tail[-INTERP_LEN:]
        return self._cut(done)


def to_grey(depth, lo, hi):
    """Relative depth -> 0..1, near = white (upstream's grayscale output)."""
    return np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
