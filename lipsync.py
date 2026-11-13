"""Lipsync nodes (Cubric Studio MPI-538): a square crop that follows a tracked face, with a lower-face mask, and
the paste of a re-rendered crop back into the full frames, colour-matched. Only the lower face is re-made; the
rest of the shot stays the source's. torch + numpy only, no ComfyUI import, so tests/test_lipsync.py loads this
file as it is."""
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore

FACE_CROP = "MPI_FACE_CROP"
_MIN_PX = 20  # a mask with fewer pixels than this on a frame is "no face on this frame"
_GRADE_FRAMES = 3


def _box(m):
    """(x0, y0, x1, y1) of a (H, W) bool mask, or None when it is (near) empty."""
    if int(m.sum()) < _MIN_PX:
        return None
    ys = torch.nonzero(m.any(1)).flatten()
    xs = torch.nonzero(m.any(0)).flatten()
    return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])


def _masks(mask, n, h, w):
    m = (mask if mask.dim() == 3 else mask.unsqueeze(0)).float().cpu()
    if m.shape[1:] != (h, w):
        m = F.interpolate(m[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    if m.shape[0] == 1 and n > 1:
        m = m.expand(n, h, w)
    if m.shape[0] != n:
        raise ValueError(f"MpiFaceTrackCrop: mask has {m.shape[0]} frames, the video {n}")
    return m > 0.5


def _dilate(m, r):
    """Grow a (H, W) 0/1 float mask by r px: square and cross 3x3 passes in turn make an octagon, close to the
    ellipse kernel the bench used at a fraction of a big kernel's cost."""
    x = m[None, None]
    for k in range(int(r)):
        if k % 2 == 0:
            x = F.max_pool2d(x, 3, 1, 1)
        else:
            x = torch.maximum(F.max_pool2d(x, (1, 3), 1, (0, 1)), F.max_pool2d(x, (3, 1), 1, (1, 0)))
    return x[0, 0]


def track_crop(images, face_mask, mouth_mask=None, size=512, zoom=True, scale=1.9, top_cut=0.35, hard_cut=False):
    """Per frame: a square around the face (box smoothed over 7 frames so the crop does not shake), scaled to
    size x size, and the lower-face mask in the same space.

    zoom: the square's side follows the face per frame, so a subject walking at the camera stays the same size in
    the crop; off = one side for the whole clip (the largest face x scale).
    The mask is the face from a cut line down, plus the lips, grown by 12% of the face. The cut sits top_cut of
    the way from the lips' top up to the face's top (0.35 reaches the nose; ~0.1 is just above the lips, for
    cartoon faces whose big eyes sit low). hard_cut re-cuts after the grow, which pushes the edge back up.
    The mouth mask may hold every tracked object: only its pixels inside the face's box count.
    Frames with no face get an empty mask and an interpolated crop."""
    img = images[..., :3].float().cpu()
    n, h, w, _ = img.shape
    face = _masks(face_mask, n, h, w)
    mouth = _masks(mouth_mask, n, h, w) if mouth_mask is not None else torch.zeros_like(face)

    fb = [_box(face[i]) for i in range(n)]
    idx = [i for i, b in enumerate(fb) if b is not None]
    if not idx:
        raise ValueError("MpiFaceTrackCrop: no face found in any frame. The face must be visible and facing the camera.")
    arr = np.array([fb[i] for i in idx], float)
    full = np.stack([np.interp(range(n), idx, arr[:, k]) for k in range(4)], 1)
    pad = np.pad(full, ((3, 3), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(pad[:, c], np.ones(7) / 7, mode="valid") for c in range(4)], 1)
    fw, fh = sm[:, 2] - sm[:, 0], sm[:, 3] - sm[:, 1]
    fs = np.maximum(fw, fh)
    if zoom:
        sides = np.clip(np.round(fs * scale), 96, min(w, h)).astype(int)
        growf = np.maximum(6, 0.12 * fs).astype(int)
    else:
        sides = np.full(n, int(min(fs.max() * scale, w, h)))
        growf = np.full(n, max(6, int(0.12 * np.median(fs))))
    cx, cy = (sm[:, 0] + sm[:, 2]) / 2, (sm[:, 1] + sm[:, 3]) / 2 + 0.08 * fh
    ox = np.clip(np.round(cx - sides / 2), 0, w - sides).astype(int)
    oy = np.clip(np.round(cy - sides / 2), 0, h - sides).astype(int)

    crops, masks = [], []
    for i in range(n):
        s, x, y = int(sides[i]), int(ox[i]), int(oy[i])
        c = img[i, y:y + s, x:x + s].movedim(-1, 0)[None]
        crops.append(F.interpolate(c, size=(size, size), mode="bicubic", align_corners=False, antialias=True)[0].movedim(0, -1))
        m = torch.zeros(s, s)
        if fb[i] is not None:
            x0, y0, x1, y1 = fb[i]
            inbox = torch.zeros(h, w, dtype=torch.bool)
            inbox[y0:y1 + 1, x0:x1 + 1] = True
            lips = mouth[i] & inbox
            lb = _box(lips)
            cut = int(lb[1] - top_cut * (lb[1] - y0)) if lb else (y0 + y1) // 2
            fm = face[i].clone()
            fm[:cut] = False
            if lb:
                fm |= lips
            m = fm[y:y + s, x:x + s].float()
            cut_c = cut - y  # the cut in crop rows
        k = size / s
        m = F.interpolate(m[None, None], size=(size, size), mode="nearest")[0, 0]
        if fb[i] is not None:
            g = max(1, round(growf[i] * k))
            m = _dilate(m, g)
            if hard_cut:
                m[:max(0, int(round((cut_c - growf[i] / 2) * k)))] = 0
        masks.append((m > 0.5).float())
    info = {"src_w": w, "src_h": h, "size": size, "frames": n,
            "origins": np.stack([ox, oy], 1).tolist(), "sides": sides.tolist()}
    return torch.stack(crops).clamp_(0.0, 1.0), torch.stack(masks), info


# sRGB D65 <-> CIE LAB, the colour space the bench's grade was measured in.
_M = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
_WHITE = torch.tensor([0.950456, 1.0, 1.088754])


def rgb_to_lab(rgb):
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb.clamp_min(0) + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-12) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = torch.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = torch.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    lin = (xyz @ torch.linalg.inv(_M).T).clamp(0, 1)
    return torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055).clamp(0, 1)


def _blur(m, sigma):
    r = int(np.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.pad(m[None, None], (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return F.conv2d(x, k.view(1, 1, -1, 1))[0, 0]


def track_paste(images, generated, mask, info, color_match=True):
    """Paste each generated crop back at its frame's square, only inside the mask (edge feathered by side/80).

    color_match: a video model hands the lower face back ~5 L darker than the source. ONE LAB mean/std transfer
    is measured inside the mask on the first 3 masked frames (the mouth usually still closed there, so both show
    the same face) and held for the whole clip; a per-frame or ring fit only halved the seam (MPI-538 runs.md).
    With a zooming crop, frames whose face is under half its largest size are skipped as the reference: a tiny
    blurry face graded the rest of the clip lighter.
    Frame counts may differ (a video model rounds its length): the shortest wins."""
    src = images[..., :3].float().cpu()
    gen = generated[..., :3].float().cpu()
    mk = (mask if mask.dim() == 3 else mask.unsqueeze(0)).float().cpu()
    sides, org = info["sides"], info["origins"]
    n = min(src.shape[0], gen.shape[0], mk.shape[0] if mk.shape[0] > 1 else gen.shape[0], len(sides))

    gcs, mbs = [], []
    for i in range(n):
        s = int(sides[i])
        gcs.append(F.interpolate(gen[i].movedim(-1, 0)[None], size=(s, s), mode="bicubic", align_corners=False,
                                 antialias=True)[0].movedim(0, -1).clamp(0, 1))
        mi = mk[min(i, mk.shape[0] - 1)]
        mbs.append(F.interpolate(mi[None, None], size=(s, s), mode="bilinear", align_corners=False)[0, 0] > 0.5)
    on = [bool(m.any()) for m in mbs]

    grade = None
    if color_match and any(on):
        big = max(sides[i] for i in range(n) if on[i])
        ref = [i for i in range(n) if on[i] and sides[i] >= big / 2][:_GRADE_FRAMES]
        st = []
        for i in ref:
            x, y, s = org[i][0], org[i][1], int(sides[i])
            a = rgb_to_lab(gcs[i])[mbs[i]]
            b = rgb_to_lab(src[i, y:y + s, x:x + s])[mbs[i]]
            st.append(torch.stack([a.mean(0), a.std(0), b.mean(0), b.std(0)]))
        mg, sg, ms, ss = torch.stack(st).mean(0)
        grade = (mg, (ss / sg.clamp_min(1e-3)).clamp(0.8, 1.25), ms)

    out = src[:n].clone()
    for i in range(n):
        if not on[i]:
            continue
        x, y, s = org[i][0], org[i][1], int(sides[i])
        gi = gcs[i]
        if grade is not None:
            mg, k, ms = grade
            gi = lab_to_rgb((rgb_to_lab(gi) - mg) * k + ms)
        m = _blur(mbs[i].float(), max(2.0, s / 80))[..., None]
        out[i, y:y + s, x:x + s] = gi * m + out[i, y:y + s, x:x + s] * (1 - m)
    return out.clamp_(0.0, 1.0)


class MpiFaceTrackCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The video frames."}),
                "face_mask": ("MASK", {"tooltip": "One tracked face per frame (SAM3 'face', one object)."}),
                "size": ("INT", {"default": 512, "min": 256, "max": 1536, "step": 64, "tooltip": "Side of the square crop out."}),
                "zoom": ("BOOLEAN", {"default": True, "tooltip": "The square follows the face's size per frame, so a subject moving toward or away from the camera stays the same size in the crop. Off = one size for the whole clip."}),
                "scale": ("FLOAT", {"default": 1.9, "min": 1.2, "max": 4.0, "step": 0.05, "tooltip": "Square side = face size x this."}),
                "top_cut": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Where the mask starts, as a share of the way from the lips' top up to the face's top. 0.35 reaches the nose; about 0.1 is just above the lips (cartoon faces, whose big eyes sit low)."}),
                "hard_cut": ("BOOLEAN", {"default": False, "tooltip": "Re-cut the mask's top after it is grown, so it cannot reach the eyes."}),
            },
            "optional": {
                "mouth_mask": ("MASK", {"tooltip": "SAM3 'mouth, lips and jaw' (any number of objects): its pixels inside the face count as lips."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", FACE_CROP)
    RETURN_NAMES = ("crop", "mask", "crop_info")
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "A square crop that follows a tracked face through a video (smoothed so it does not shake), scaled to one "
        "size, plus the lower-face mask in the same space: the face from a cut line down, plus the lips, grown. "
        "For re-making only the mouth of a clip (lipsync); MpiFaceTrackPaste puts the result back."
    )
    FUNCTION = "doit"

    def doit(self, images, face_mask, size, zoom, scale, top_cut, hard_cut, mouth_mask=None):
        return track_crop(images, face_mask, mouth_mask, size, zoom, scale, top_cut, hard_cut)


class MpiFaceTrackPaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The same video frames MpiFaceTrackCrop cropped."}),
                "generated": ("IMAGE", {"tooltip": "The re-rendered crop frames."}),
                "mask": ("MASK", {"tooltip": "MpiFaceTrackCrop's mask: only this part of the crop is pasted."}),
                "crop_info": (FACE_CROP,),
                "color_match": ("BOOLEAN", {"default": True, "tooltip": "Grade the re-render to the source inside the mask, measured on the first masked frames."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "Paste re-rendered face crops back into the full video at the squares MpiFaceTrackCrop cut, only inside "
        "its mask with a feathered edge, after one LAB colour match measured on the first masked frames. "
        "Mismatched frame counts keep the shortest."
    )
    FUNCTION = "doit"

    def doit(self, images, generated, mask, crop_info, color_match=True):
        return (track_paste(images, generated, mask, crop_info, color_match),)
