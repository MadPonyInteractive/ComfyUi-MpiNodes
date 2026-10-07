"""360 equirect wrap helpers: pad / crop around the wrap, soften the wrap edge, merge two renders of
the wrap by a cut. Pure numpy/torch on [H, W, C] float arrays (0..255 for soften / cut_merge, any
range for pad / crop) so the tests run without ComfyUI. Ported from the MPI-623 bench scripts
(run_animesharp2k.py, wrap_soften.py, wrap_cut.py); every constant below is the approved value."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, maximum_filter1d


def wrap_pad(img, pad):
    """[B, H, W, C] -> [B, H, W + 2*pad, C]: the opposite edge on each side, so a model sees the
    360 wrap like any other column (unpadded, AnimeSharp left a hairline at 8K)."""
    if pad <= 0:
        return img
    return torch.cat([img[:, :, -pad:], img, img[:, :, :pad]], dim=2)


def pad_at_scale(padded_w, ref_w, pad):
    """The pad in a padded image's own pixels after any resize: the padded source was ref_w + 2*pad
    wide and is now padded_w wide."""
    return round(pad * padded_w / (ref_w + 2 * pad))


def wrap_crop(img, ref_w, pad):
    """Undo wrap_pad on an image that may have been upscaled since."""
    p = pad_at_scale(img.shape[2], ref_w, pad)
    return img[:, :, p:img.shape[2] - p] if p > 0 else img


def _hblur_wrap(a, sigma):
    """Horizontal Gaussian, wrapping round the 360 (np.roll), radius 3 sigma."""
    k = int(3 * sigma)
    ker = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
    ker /= ker.sum()
    return sum(wt * np.roll(a, s, axis=1) for s, wt in zip(range(-k, k + 1), ker))


def wrap_soften(a, sigma=12.0, radius=48, threshold=6.0):
    """Blur [H, W, 3] (0..255) across the 360 wrap only, blended in by a smoothstep 1 at the wrap
    -> 0 at +-radius. threshold > 0 blurs only where it changes a sigma-2-smoothed copy by less
    than threshold/2 levels (ramp to 0 at threshold): flat sky softens, a cloud or tree the wrap
    cuts stays a hard edge for the seam pass to join (Fabio 2026-10-06, 12/48/6 approved).
    Columns further than 3*sigma + radius + 32 px from the wrap come back unchanged."""
    w = a.shape[1]
    r = np.roll(a, w // 2, axis=1)  # wrap at x = w/2
    blur = _hblur_wrap(r, sigma)
    t = np.clip(1 - np.abs(np.arange(w) - (w // 2 - 0.5)) / radius, 0, 1)
    t = (t * t * (3 - 2 * t))[None, :, None]
    if threshold > 0:
        # judge on a sigma-2 smoothed copy: paper texture moves under any blur, a cloud edge far more
        g = np.exp(-0.5 * (np.arange(-6, 7) / 2) ** 2)
        g /= g.sum()
        sm = r
        for ax in (0, 1):
            sm = sum(wt * np.roll(sm, s, axis=ax) for s, wt in zip(range(-6, 7), g))
        smb = _hblur_wrap(sm, sigma)
        m = np.clip((threshold - np.abs(smb - sm).max(axis=2)) / (threshold / 2), 0, 1)
        for ax in (0, 1):  # erode 8 px, then soften, so an edge's whole neighbourhood stays out
            m = np.min([np.roll(m, s, axis=ax) for s in range(-8, 9)], axis=0)
        g4 = np.exp(-0.5 * (np.arange(-12, 13) / 4) ** 2)
        g4 /= g4.sum()
        for ax in (0, 1):
            m = sum(wt * np.roll(m, s, axis=ax) for s, wt in zip(range(-12, 13), g4))
        t = t * m[:, :, None]
    return np.roll(r * (1 - t) + blur * t, -(w // 2), axis=1)


def wrap_cut_merge(pd, overlap):
    """Merge the two renders of the wrap by a CUT, not a cross-fade (a cross-fade shows a line the
    two renders put a few px apart TWICE, and AnimeSharp sharpens it into a ghost outline).

    pd: [H, overlap + W + overlap, 3] (0..255), a wrap-padded render. R = the right pad (core cols
    0..overlap, rendered beside col W-1), C = the same cols rendered in place. Tone: LP(R) - LP(C)
    ramps out across the overlap. Detail: each row takes R left of a min-cost top-to-bottom cut
    through the dilated |R - C| and C right of it, 2 px feather. Returns [H, W, 3]."""
    O = overlap
    W = pd.shape[1] - 2 * O
    H = pd.shape[0]
    R = pd[:, O + W:O + W + O]
    C = pd[:, O:2 * O]
    rest = pd[:, 2 * O:O + W]

    lp = lambda x: gaussian_filter(x, (32, 32, 0), mode="nearest")  # noqa: E731 - big sigma: line offsets cancel
    D = lp(R) - lp(C)
    t = np.linspace(1, 0, O, dtype=np.float32)[None, :, None]
    Rt, Ct = R - (1 - t) * D, C + t * D

    margin = O // 8  # ponytail: wrap_cut.py's 32 px at a 256 px overlap, scaled with the overlap
    E = gaussian_filter(np.abs(Rt - Ct).sum(axis=2), 1.5)
    E = maximum_filter1d(E, 9, axis=1)
    E[:, :margin] = E[:, O - margin:] = 1e9
    acc = E.copy()
    back = np.zeros((H, O), np.int8)
    for y in range(1, H):
        prev = acc[y - 1]
        cand = np.stack([np.r_[np.inf, prev[:-1]], prev, np.r_[prev[1:], np.inf]])
        k = cand.argmin(axis=0)
        back[y] = k - 1
        acc[y] += cand[k, np.arange(O)]
    path = np.empty(H, int)
    path[-1] = int(acc[-1].argmin())
    for y in range(H - 1, 0, -1):
        path[y - 1] = path[y] + back[y, path[y]]
    m = (np.arange(O)[None, :] < path[:, None]).astype(np.float32)
    m = gaussian_filter(m, (0, 2))[:, :, None]  # 2 px feather across the cut only
    return np.concatenate([m * Rt + (1 - m) * Ct, rest], axis=1)
