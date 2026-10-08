"""Grade nodes. torch only, no ComfyUI import, so tests/test_grade.py loads this file as it is."""
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore

_MIN_PIXELS = 16  # below this a fit is noise, so the frame is left ungraded


def _solve(g, o):
    """Per-channel o ~= a*g + b over paired pixels (K, 3). A flat channel gets offset only."""
    gm, om = g.mean(0), o.mean(0)
    var = ((g - gm) ** 2).mean(0)
    a = torch.where(var > 1e-8, ((g - gm) * (o - om)).mean(0) / var.clamp_min(1e-8), torch.ones_like(var))
    a = a.clamp(0.2, 5.0)
    return a, om - a * gm


def fit_grade(g, o):
    """Fit, then refit without the pixels whose residual is past 2.5 sd in any channel: content that MOVES
    through the band (a hand crossing the box edge) is not grading drift and would skew the line. Three
    passes, because the first fit is skewed by the very pixels it then has to reject (one pass left a 6%
    patch of motion at 5/255 off; three remove it, tests/test_grade.py)."""
    a, b = _solve(g, o)
    for _ in range(3):
        resid = o - (a * g + b)
        keep = (resid.abs() <= 2.5 * resid.std(0).clamp_min(1e-8)).all(1)
        if int(keep.sum()) < _MIN_PIXELS:
            break
        a, b = _solve(g[keep], o[keep])
    return a, b


def grade_match(destination, source, mask, band):
    """Composite source over destination through mask, after grading source to destination.

    Outside the mask the two show the same content, so the difference there is grading drift alone (a
    re-render comes back a touch lighter or warmer). A per-channel gain + offset is fitted on those paired
    pixels, per frame, and applied to the source, so what the mask keeps lines up with the plate and keeps
    its own colours. band > 0 limits the fit to a ring that wide round the mask; 0 uses all of outside.
    """
    dest = destination[..., :3].float().cpu()
    _, H, W, _ = dest.shape
    src = source[..., :3].float().cpu()
    if src.shape[1:3] != (H, W):
        src = F.interpolate(src.movedim(-1, 1), size=(H, W), mode="bilinear", align_corners=False).movedim(1, -1)
    m = (mask if mask.dim() == 3 else mask.unsqueeze(0)).float().cpu()
    if m.shape[1:] != (H, W):
        m = F.interpolate(m[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]

    counts = (dest.shape[0], src.shape[0], m.shape[0])
    n = max(counts)
    if set(counts) - {1, n}:
        # A clamp-to-last-frame here would grade and paste the wrong frames with no error at all.
        raise ValueError(f"MpiGradeMatch: frame counts differ (destination {counts[0]}, source {counts[1]}, mask {counts[2]})")

    out = []
    for i in range(n):
        d, s, mi = dest[min(i, counts[0] - 1)], src[min(i, counts[1] - 1)], m[min(i, counts[2] - 1)]
        inside = mi > 0.5
        sel = ~inside
        if band > 0:
            ring = F.max_pool2d(inside.float()[None, None], 2 * band + 1, stride=1, padding=band)[0, 0] > 0.5
            ring &= ~inside
            if int(ring.sum()) >= _MIN_PIXELS:
                sel = ring
        if int(sel.sum()) >= _MIN_PIXELS:
            a, b = fit_grade(s[sel], d[sel])
            s = s * a + b
        w = mi[..., None]
        out.append(d * (1 - w) + s * w)
    return torch.stack(out).clamp_(0.0, 1.0)


class MpiGradeMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE", {"tooltip": "The original footage: kept outside the mask, and the grade reference."}),
                "source": ("IMAGE", {"tooltip": "The re-render of the SAME shot: kept inside the mask, after grading."}),
                "mask": ("MASK", {"tooltip": "Where the source is kept. A batch of 1 holds for every frame."}),
                "band": (
                    "INT",
                    {
                        "default": 48, "min": 0, "max": 1024,
                        "tooltip": "Width of the ring round the mask the grade is fitted on. 0 fits on all of outside.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = (
        "Composite a re-render over the original through a mask, graded to match it. Outside the mask both show "
        "the same content, so a per-channel gain + offset is fitted there per frame (content moving through the "
        "band is dropped from the fit) and applied to the source: the seam lines up and what the mask keeps keeps "
        "its own colours. Not MpiInpaintHeal, which moves a fill's colour toward its surroundings and would tint "
        "an added object; this one needs the original and the re-render of the same shot."
    )
    FUNCTION = "doit"

    def doit(self, destination, source, mask, band):
        return (grade_match(destination, source, mask, band),)
