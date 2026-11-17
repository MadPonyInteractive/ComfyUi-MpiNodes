"""MiniMax H3 keyframes: several pictures anchored at evenly spaced frames of one clip.

`MpiH3AddGuides` takes up to 8 picture slots, drops the empty ones, and anchors
the rest in slot order: the first on frame 0, the last on the last frame, the
others evenly between. Each anchor is built by core's own `MiniMaxH3AddGuide`, so
the tensor maths moves with ComfyUI instead of drifting in a private copy.

Why a node and not a chain of core nodes: a host app compiling a saved workflow
cannot leave a slot unconnected, and core would anchor an empty Mpi loader's
1x1 picture as a real keyframe. The spacing also depends on how many slots are
filled, which a fixed chain of `frame_idx` widgets cannot know.

`window_keyframes` re-anchors those keyframes to one temporal window of the
clip; the windowed samplers in sampler.py call it per window (Cubric Studio
MPI-1191). Deliberately free of ComfyUI imports at module scope, so the
arithmetic stays testable in bare Python.
"""

import math

try:
    from .h3 import is_blank_image, latent_frames
except ImportError:  # run outside the package (tests)
    from h3 import is_blank_image, latent_frames

SLOTS = 8
# comfy/ldm/minimax/model.py: video latent k carries FRAME_PER_TOKEN[k % 5] pixel
# frames (blocks of 1 + 4 + 4 + 4 + 4 = 17), and the audio track runs 5/3
# positions per pixel frame on the shared time axis.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def even_frames(count, frame_count):
    """Pixel frames `count` keys land on: first on 0, last on the last frame, even between.

    One key sits on frame 0. Raises when the clip is too short to give every key
    its own frame, which a silent duplicate would hide (two anchors on one frame).
    """
    if count <= 0:
        return []
    if count == 1:
        return [0]
    if frame_count < count:
        raise ValueError(f"{count} keyframes need at least {count} frames; the clip has {frame_count}")
    last = frame_count - 1
    return [round(i * last / (count - 1)) for i in range(count)]


def latent_pixel_starts(steps):
    """Pixel frame each of `steps` video latents starts at, plus the end of the last one."""
    starts = [0]
    for k in range(steps):
        starts.append(starts[-1] + FRAME_PER_TOKEN[k % 5])
    return starts


def _place_in_window(frame, full, win, start, end):
    """Full-clip pixel frame -> the same moment on the window's own timeline, or None.

    The window is sampled as a clip of its own, so the model reads its latent j at
    the window's own pixel start for j -- not at the clip's. A key is moved to the
    window latent that holds its frame, keeping its offset inside that latent. On
    H3's 5-latent grid (frame_grid 5) a window starts on a block, so this is just
    `frame - full[start]`; off the grid the offset is clamped to the latent's size.
    """
    if frame < 0 or frame >= full[-1]:
        return None
    latent = next(k for k in range(len(full) - 1) if full[k] <= frame < full[k + 1])
    if not start <= latent < end:
        return None
    j = latent - start
    return win[j] + min(frame - full[latent], win[j + 1] - win[j] - 1)


def window_keyframes(keyframes, start, end, total, audio_span=None):
    """H3 keyframes of the whole clip -> the ones inside latent window [start, end), re-anchored.

    Without this a windowed sampler hands every window the WHOLE clip's keyframes
    at the clip's frame numbers: from the second window on an anchor lands at the
    wrong moment (or past the window's end), silently. Mirrors what core does for
    its own context windows (`MiniMaxH3AV._window_payload`):
    - a picture anchor is kept when its frame is in the window, at the window's frame;
    - a clip anchor keeps the 5-latent chunks that overlap the window;
    - an audio anchor keeps the part inside the window's audio span `(a0, a1)`.
    Window 0 of a clip is the identity, so the first window is unchanged.
    """
    full = latent_pixel_starts(total)
    win = latent_pixel_starts(end - start)
    out = []
    for kf in keyframes or []:
        f = kf["resolved_frame_index"]
        latent = kf.get("latent")
        if latent is not None:
            vt = latent.shape[2]
            if vt == 1:
                placed = _place_in_window(f, full, win, start, end)
                if placed is not None:
                    out.append({"resolved_frame_index": placed, "latent": latent})
            else:
                guide = latent_pixel_starts(vt)
                lo, hi = full[start], full[end]
                chunks = [g for g in range(0, vt, 5)
                          if f + guide[g] < hi and f + guide[min(g + 5, vt)] > lo]
                if chunks:
                    first = max(f + guide[chunks[0]], lo)
                    placed = _place_in_window(first, full, win, start, end)
                    out.append({"resolved_frame_index": placed,
                                "latent": latent[:, :, chunks[0]:chunks[-1] + 5]})
        audio = kf.get("audio_latent")
        if audio is not None and audio_span is not None:
            a0, a1 = audio_span
            t = FRAME_RESCALE * f
            j0, j1 = max(0, math.ceil(a0 - t)), min(audio.shape[-1], math.ceil(a1 - t))
            if j1 > j0:
                out.append({"resolved_frame_index": (t + j0 - a0) / FRAME_RESCALE,
                            "audio_latent": audio[..., j0:j1]})
    return out


class MpiH3AddGuides:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "vae": ("VAE", {"tooltip": "The H3 video VAE."}),
                "latent": ("LATENT", {"tooltip": "The H3 AV latent this positive will sample. Its length sets where the keys land, so wire the latent of the SAME stage (a refine at another size needs its own node)."}),
            },
            "optional": {
                f"image_{i}": ("IMAGE", {"tooltip": f"Keyframe {i}, in order. Empty (unconnected, or an Mpi loader's 1x1 empty picture) is skipped and the rest close up."})
                for i in range(1, SLOTS + 1)
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "INT")
    RETURN_NAMES = ("positive", "frames", "count")
    CATEGORY = "MpiNodes/H3"
    DESCRIPTION = (
        "Anchor up to 8 pictures at evenly spaced frames of a MiniMax H3 clip: the "
        "first on frame 0, the last on the last frame, the others evenly between, in "
        "slot order. Empty slots are skipped (an Mpi loader's 1x1 empty picture "
        "included), so one graph covers 1 to 8 keys. Each anchor is core "
        "MiniMaxH3AddGuide's. Use one node per sampling stage, each with that stage's "
        "latent. `frames` reports the frame of each key, `count` how many were anchored."
    )
    FUNCTION = "doit"

    def doit(self, positive, vae, latent, **slots):
        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide  # type: ignore
        except ImportError as exc:
            raise RuntimeError("MiniMax H3 nodes are missing - MpiH3AddGuides needs a ComfyUI with H3.") from exc

        images = [slots.get(f"image_{i}") for i in range(1, SLOTS + 1)]
        images = [img for img in images if not is_blank_image(img)]
        samples = latent["samples"]
        video = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples
        frame_count = latent_frames(video.shape[2])
        frames = even_frames(len(images), frame_count)

        for image, frame in zip(images, frames):
            positive = MiniMaxH3AddGuide.execute(
                positive=positive, latent=latent, frame_idx=frame, vae=vae, image=image[:1]).result[0]

        report = ", ".join(str(f) for f in frames) if frames else "none"
        print(f"[MpiH3AddGuides] {len(frames)} keys of {frame_count} frames at {report}")
        return (positive, report, len(frames))
