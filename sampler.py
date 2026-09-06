"""Sampling nodes.

`MpiWindowedSampler` exists because a video DiT's attention cost scales with
T x H x W, so a clip that fits at one duration OOMs at another with every other
setting identical. Measured on MiniMax H3 at 2K (latent 92x160) on a 16GB card:
T=27 samples fine, T=32 and T=37 both OOM inside
`comfy_kitchen.prequantize_int8_attention`. H3's own trained minimum is 124
frames -- T=37. So the model's SHORTEST trained clip is the one the card cannot
refine, and no parameter fixes it: lowering the upscale factor moves cost into
stage 1 without touching stage-2 peak, which is set by the OUTPUT latent size.

Windowing is sound HERE, and would not be for a first-pass generation, because
this is a REFINE. The latent arrives globally coherent -- motion, identity and
framing were all decided upstream -- and a short sigma schedule only adds local
detail. Every window reads the same source and the same noise, so two windows
cannot disagree about content the way two independent generations would.
"""
import torch  # type: ignore

import comfy.model_management  # type: ignore
import comfy.sample  # type: ignore
import comfy.utils  # type: ignore
import latent_preview  # type: ignore
from comfy.nested_tensor import NestedTensor  # type: ignore


def latent_to_frames(t, grid, ratio):
    """Video frames a run of `t` latent frames carries.

    H3 packs `(1, 4, 4, 4, 4)` video frames per latent frame, indexed by the
    ABSOLUTE latent position mod 5 -- see `MiniMaxH3AV.fix_empty_latent` in
    comfy/latent_formats.py. So every `grid`-th token is short and the rest carry
    the model's `temporal_downscale_ratio`. Verified against the two clip lengths
    this card has measured: T=37 -> 124 frames and T=22 -> 73.

    With `grid=1` there is no packing model, so this is the identity and the
    node's frame widgets are really counting latent frames. The info output says
    which of the two happened rather than leaving it to be guessed.
    """
    return ratio * t - (ratio - 1) * -(-t // grid)


def frames_to_latent(frames, grid, ratio, cap):
    """Largest latent run of at most `cap` whose video length fits in `frames`.

    Searched rather than inverted: the exact inverse has to round the same way
    the ceiling above does, and a clip is a few dozen latent frames.
    """
    t = 0
    while t < cap and latent_to_frames(t + 1, grid, ratio) <= frames:
        t += 1
    return t


def plan_windows(total, window, overlap, grid=1):
    """Cover [0, total) with windows of AT MOST `window` frames, overlapping.

    `window` is a CEILING, not a target: it is the largest the card was measured
    against, so once the pass count is fixed the windows shrink to the smallest
    legal size that still covers. Same number of passes, fewer tokens in each.

    `grid` is the model's temporal quantum, and it is not cosmetic. H3 patchifies
    time as a 2-frame causal head plus blocks of 5, so a legal latent length is
    5k+2 and a legal cut point is a multiple of 5. A window of 21 ends 4 frames
    into a block; the trailing partial block is padded and decodes BLACK, which
    reads as a blend artefact rather than as an illegal length. 27 (5*5+2) has no
    partial block and is clean. `grid=1` means the model has no such constraint
    and reduces this to plain fixed-size windows.

    The grid beats the ceiling when the two disagree: the smallest tileable window
    is one whole block plus the head (7 for H3), so a ceiling under that is raised
    to it rather than honoured off-grid. An over-ceiling window risks an OOM; an
    off-grid one is guaranteed black frames. The log line reports both numbers.

    The last window is pulled back to full width rather than left short, so every
    window is the same size the card was measured against.
    """
    phase = total % grid  # 2 for a well-formed H3 clip; 0 when grid is 1
    floor = grid + phase  # one whole block plus the head: the smallest tileable

    def snap(n):
        """Largest legal window length <= n."""
        return max(floor, ((n - phase) // grid) * grid + phase)

    def step_for(w):
        """Stride for windows of `w`. Kept a multiple of `grid` so every window
        STARTS on a legal cut point too, and never wider than `w` so no frame is
        skipped. Rounding down only ever buys more overlap than was asked for."""
        return max(grid, ((w - overlap) // grid) * grid)

    window = snap(window)
    if window >= total:
        return [(0, total)]

    step = step_for(window)
    passes = 1 + -(-(total - window) // step)

    for candidate in range(floor, window + 1, grid):
        candidate_step = step_for(candidate)
        if candidate + (passes - 1) * candidate_step >= total:
            window, step = candidate, candidate_step
            break

    spans = []
    start = 0
    while True:
        end = min(start + window, total)
        spans.append((start, end))
        if end >= total:
            break
        start += step

    if spans[-1][1] - spans[-1][0] < window:
        spans[-1] = (max(0, total - window), total)
    return spans


def _edge_overlaps(spans, index):
    """(lead, tail): frames this window ACTUALLY shares with each neighbour.

    Not the requested overlap. Snapping the stride to the grid buys extra shared
    frames, and ramping over the requested number instead would crossfade across
    part of the shared region and hard-cut the rest -- a narrow fade between two
    independently denoised windows is where a ghost shows up. The clip's true
    first and last frames have no neighbour and so keep full weight, or the ends
    fade toward whatever the accumulator was initialised with.
    """
    start, end = spans[index]
    n = end - start
    lead = spans[index - 1][1] - start if index > 0 else 0
    tail = end - spans[index + 1][0] if index + 1 < len(spans) else 0
    return max(0, min(lead, n)), max(0, min(tail, n))


def _blend_weights(spans, index, device, dtype):
    """Linear ramp in, flat, linear ramp out -- over the real shared regions."""
    start, end = spans[index]
    w = torch.ones(end - start, device=device, dtype=dtype)
    lead, tail = _edge_overlaps(spans, index)
    if lead > 0:
        w[:lead] = torch.arange(1, lead + 1, device=device, dtype=dtype) / (lead + 1)
    if tail > 0:
        w[-tail:] = torch.arange(tail, 0, -1, device=device, dtype=dtype) / (tail + 1)
    return w


def _split_av(samples):
    """(video, rest) for a joint AV latent, (tensor, None) for a plain one.

    H3 and LTXV carry video+audio as a `NestedTensor` of two differently-shaped
    tensors, which is why the video cannot simply be sliced in place.
    """
    if getattr(samples, "is_nested", False):
        parts = samples.unbind()
        return parts[0], parts[1:]
    return samples, None


def _rejoin_av(video, rest):
    if rest is None:
        return video
    return NestedTensor([video] + list(rest))


def _audio_span(start, end, total, grid, ratio, ref_len):
    """Audio token span [a0, a1) matching the video latent window [start, end).

    Audio is NOT a global stream the model reads whole -- it is indexed in time
    against the video. `MiniMaxH3AV.fix_empty_latent` sizes it
    `round(frame_count * 5/3)`, so an audio token has a frame it belongs to, and
    a window handed the WHOLE track is a window whose mouth is reading the wrong
    moment. On a 124-frame clip planned as [(0,22),(15,37)] that is a ~2 s offset
    in the second window, and it shows up as a mouth that sticks at the seam.

    The rate is derived from the tensors rather than hardcoded as 5/3, so a model
    with a different packing (or a future H3 revision) needs no change here.
    Length is computed as the model would for a clip of this many frames and then
    placed at the window's offset, so the slice is exactly the size a standalone
    render of that window would have produced -- one frame short and the packed
    layout no longer describes the tensor it was built for.
    """
    frames_total = latent_to_frames(total, grid, ratio)
    if frames_total <= 0 or ref_len <= 0:
        return 0, ref_len
    rate = ref_len / frames_total
    f0 = latent_to_frames(start, grid, ratio)
    want = round((latent_to_frames(end, grid, ratio) - f0) * rate)
    want = max(1, min(want, ref_len))
    a0 = round(f0 * rate)
    a1 = min(a0 + want, ref_len)
    return a1 - want, a1


def _slice_rest(rest, a0, a1, ref_len):
    """Slice the audio side of an AV latent by its LAST (time) dim.

    Only tensors whose last dim matches the reference length are touched; anything
    else in the nested tuple is passed through, so a model carrying a non-temporal
    extra stream is unaffected rather than silently mangled.
    """
    if rest is None:
        return None
    return tuple(t[..., a0:a1] if t.ndim >= 1 and t.shape[-1] == ref_len else t
                 for t in rest)


class MpiWindowedSampler:
    """SamplerCustomAdvanced, run over overlapping temporal windows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "window_frames": ("INT", {
                    "default": 90, "min": 1, "max": 100000,
                    "tooltip": "CEILING on the VIDEO frames refined at once, not a "
                               "target. This is the number you discover by hitting an "
                               "OOM: if 124 frames dies and 90 survives, put 90 here. "
                               "90 is what a 16GB card manages for MiniMax H3 at 2K. "
                               "Windows are then shrunk to the smallest legal size "
                               "that still covers the clip in the same number of "
                               "passes, so a generous ceiling costs nothing and a low "
                               "one costs passes. A clip at or under this length takes "
                               "the plain single-pass path.",
                }),
                "overlap_frames": ("INT", {
                    "default": 17, "min": 0, "max": 100000,
                    "tooltip": "MINIMUM VIDEO frames neighbouring windows share and "
                               "cross-fade. Costs compute on every seam. 17 is one "
                               "whole block for H3. Snapping to the grid can only ever "
                               "hand you more than you ask for, never less.",
                }),
                "frame_grid": ("INT", {
                    "default": 5, "min": 1, "max": 64,
                    "tooltip": "ADVANCED - the model's temporal quantum, in LATENT "
                               "frames. Leave it at 5 for MiniMax H3, which packs time "
                               "as a 2-frame causal head plus blocks of 5; a window "
                               "off that grid ends in a padded part-block that decodes "
                               "BLACK. Set 1 for a model with no such structure, which "
                               "gives plain fixed-size windows and makes the two "
                               "widgets above count latent frames instead.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("output", "denoised_output", "info")
    CATEGORY = "MpiNodes/Sampling"
    DESCRIPTION = (
        "SamplerCustomAdvanced that samples the latent in "
        "overlapping temporal windows and cross-fades them, so a clip too long to "
        "sample in one pass still fits. Intended for a REFINE pass over an already "
        "coherent latent, not for first-pass generation. Audio in a joint AV latent "
        "is passed through untouched. Falls back to a single ordinary sample when "
        "the clip already fits, so short clips are unchanged. Both frame widgets are "
        "in VIDEO frames, the unit an OOM is actually measured in, and the info "
        "output reports what was planned."
    )
    FUNCTION = "doit"

    def doit(self, noise, guider, sampler, sigmas, latent_image, window_frames,
             overlap_frames, frame_grid=5):
        latent = latent_image.copy()
        samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent_image["samples"],
            latent_image.get("downscale_ratio_spacial", None),
            latent_image.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = samples

        video, audio = _split_av(samples)
        total = video.shape[2]
        # Reference length for the audio slice: every stream measured against the
        # first one, so a mismatched extra tensor is passed through untouched.
        audio_len = audio[0].shape[-1] if audio else 0

        grid = max(1, frame_grid)
        # The model itself says how many video frames a latent frame carries, so
        # the widgets can be in the unit an OOM is actually measured in without
        # asking the user for a second number.
        ratio = getattr(getattr(guider.model_patcher.model, "latent_format", None),
                        "temporal_downscale_ratio", 1)
        window = max(1, frames_to_latent(window_frames, grid, ratio, total))
        overlap = frames_to_latent(overlap_frames, grid, ratio, max(0, window - 1))

        spans = plan_windows(total, window, overlap, grid)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # One noise field for the whole clip, sliced per window. Generating noise
        # per window instead would hand the two sides of a seam different noise,
        # so the cross-fade would be averaging two genuinely different denoisings
        # rather than two near-identical ones.
        full_noise = noise.generate_noise(latent)
        noise_video, noise_rest = _split_av(full_noise)

        if len(spans) == 1:
            info = (f"single pass: {total} latent frames "
                    f"({latent_to_frames(total, grid, ratio)} video), at or under the "
                    f"{window_frames}-frame ceiling, so nothing was windowed")
            print(f"[MpiWindowedSampler] {info}")
            out, out_denoised = self._sample_once(
                guider, sampler, sigmas, latent, full_noise, noise.seed, disable_pbar)
            return (out, out_denoised, info)

        # Every window is grid-aligned and equal length, so one conversion
        # describes all of them.
        planned = spans[0][1] - spans[0][0]
        shared = spans[0][1] - spans[1][0]
        info = (
            f"{len(spans)} windows of {latent_to_frames(planned, grid, ratio)} video "
            f"frames ({planned} latent), sharing "
            f"{latent_to_frames(shared, grid, ratio)} video frames ({shared} latent). "
            f"Clip is {latent_to_frames(total, grid, ratio)} video frames ({total} "
            f"latent). You asked for a {window_frames}-frame ceiling and "
            f"{overlap_frames} frames of overlap; grid {grid}, "
            f"{ratio} video frames per latent frame. spans={spans}")
        print(f"[MpiWindowedSampler] {info}")

        out_full = torch.zeros_like(video)
        weight_shape = [1] * video.ndim
        weight_shape[2] = -1
        weight_full = torch.zeros(
            [1, 1, total] + [1] * (video.ndim - 3),
            device=video.device, dtype=torch.float32)
        # `denoised_output` is a real output some graphs consume (ltx_i2v_t2v wires
        # it), so x0 is accumulated through the same weights rather than aliased to
        # `output` -- an aliased second output is valid-looking and quietly wrong.
        x0_full = None

        for index, (start, end) in enumerate(spans):
            seg_latent = latent.copy()
            seg_video = video[:, :, start:end]
            # The audio travels with its OWN window, not whole. It is conditioning
            # here, not output -- the refined audio is discarded below and the
            # original restored -- but handing the model the entire track against a
            # sliced video misaligns every frame after the first window by that
            # window's offset, and lip-sync is exactly that alignment.
            a0, a1 = _audio_span(start, end, total, grid, ratio, audio_len)
            seg_latent["samples"] = _rejoin_av(
                seg_video, _slice_rest(audio, a0, a1, audio_len))

            mask = latent.get("noise_mask", None)
            if mask is not None:
                seg_latent["noise_mask"] = self._slice_mask(mask, start, end, total)

            seg_noise = _rejoin_av(noise_video[:, :, start:end],
                                   _slice_rest(noise_rest, a0, a1, audio_len))

            x0_output = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
            print(f"[MpiWindowedSampler] window {index + 1}/{len(spans)} "
                  f"latent frames {start}-{end}")

            # The model deliberately stays loaded across windows: unloading here
            # would re-stream the whole DiT over PCIe once per window, which costs
            # far more than the sampling it protects.
            result = guider.sample(
                seg_noise, seg_latent["samples"], sampler, sigmas,
                denoise_mask=seg_latent.get("noise_mask", None),
                callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
            result = result.to(comfy.model_management.intermediate_device())

            seg_out, _ = _split_av(result)
            w = _blend_weights(spans, index, out_full.device, out_full.dtype)
            out_full[:, :, start:end] += seg_out.to(out_full.device) * w.view(weight_shape)
            weight_full[:, :, start:end] += w.view(weight_shape).to(torch.float32)

            seg_x0 = self._extract_x0(x0_output, result)
            if seg_x0 is not None:
                if x0_full is None:
                    x0_full = torch.zeros_like(out_full)
                x0_full[:, :, start:end] += seg_x0.to(x0_full.device) * w.view(weight_shape)

        weight_full = weight_full.clamp(min=1e-8)
        out_full = out_full / weight_full.to(out_full.dtype)
        if x0_full is not None:
            x0_full = x0_full / weight_full.to(x0_full.dtype)

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        # Audio is the ORIGINAL stream, not any window's refined copy: the refine
        # exists to add spatial detail, and stitching audio across seams would put
        # a discontinuity in the one channel a listener notices immediately.
        out["samples"] = _rejoin_av(out_full, audio)

        if x0_full is None:
            return (out, out, info)

        # process_latent_out is affine, so running it once on the blended x0 is the
        # same as blending per-window results -- and keeps one call, as core does.
        out_denoised = latent.copy()
        out_denoised.pop("downscale_ratio_spacial", None)
        out_denoised.pop("downscale_ratio_temporal", None)
        out_denoised["samples"] = _rejoin_av(
            guider.model_patcher.model.process_latent_out(x0_full.cpu()), audio)
        return (out, out_denoised, info)

    @staticmethod
    def _extract_x0(x0_output, result):
        """The window's denoised prediction, unpacked to match `result`'s shape.

        The callback may hand back a flat packed tensor even when the latent is a
        nested AV pair, which is what core's own unpack guards against.
        """
        x0 = x0_output.get("x0", None)
        if x0 is None:
            return None
        if getattr(result, "is_nested", False) and not getattr(x0, "is_nested", False):
            shapes = [t.shape for t in result.unbind()]
            x0 = NestedTensor(comfy.utils.unpack_latents(x0, shapes))
        head, _ = _split_av(x0)
        return head

    @staticmethod
    def _slice_mask(mask, start, end, total):
        """Slice a noise mask alongside its latent, when it is shaped to match."""
        if getattr(mask, "is_nested", False):
            parts = mask.unbind()
            head = parts[0]
            if head.ndim >= 3 and head.shape[2] == total:
                return NestedTensor([head[:, :, start:end]] + list(parts[1:]))
            return mask
        if mask.ndim >= 3 and mask.shape[2] == total:
            return mask[:, :, start:end]
        return mask

    @classmethod
    def _sample_once(cls, guider, sampler, sigmas, latent, seed_noise, seed, disable_pbar):
        """The clip already fits: behave exactly like SamplerCustomAdvanced."""
        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
        samples = guider.sample(
            seed_noise, latent["samples"], sampler, sigmas,
            denoise_mask=latent.get("noise_mask", None),
            callback=callback, disable_pbar=disable_pbar, seed=seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples

        x0 = x0_output.get("x0", None)
        if x0 is None:
            return (out, out)
        if getattr(samples, "is_nested", False) and not getattr(x0, "is_nested", False):
            shapes = [t.shape for t in samples.unbind()]
            x0 = NestedTensor(comfy.utils.unpack_latents(x0, shapes))
        out_denoised = out.copy()
        out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0.cpu())
        return (out, out_denoised)


NODE_CLASS_MAPPINGS = {
    "MpiWindowedSampler": MpiWindowedSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiWindowedSampler": "Mpi Windowed Sampler",
}


if __name__ == "__main__":
    # Self-check for the audio window. Run it with ComfyUI on sys.path, e.g.
    #   python -c "import sys; sys.path.insert(0, r'<ComfyUI>'); \
    #              exec(open(r'<this file>').read(), {'__name__': '__main__'})"
    # The numbers are H3's real ones, taken from the case that exposed the bug:
    # a 124-frame clip (T=37) planned as [(0, 22), (15, 37)], audio sized by
    # MiniMaxH3AV as round(124 * 5/3) = 207 tokens.
    G, R, TOTAL, REF = 5, 4, 37, 207
    assert latent_to_frames(TOTAL, G, R) == 124, latent_to_frames(TOTAL, G, R)
    assert latent_to_frames(22, G, R) == 73

    full = _audio_span(0, TOTAL, TOTAL, G, R, REF)
    assert full == (0, REF), f"a window covering the clip must take all the audio: {full}"

    spans = [(0, 22), (15, 37)]
    got = [_audio_span(s, e, TOTAL, G, R, REF) for s, e in spans]
    assert got == [(0, 122), (85, 207)], got
    # Equal-length windows must get equal-length audio, or the packed layout
    # describes a different tensor in each pass.
    lengths = {a1 - a0 for a0, a1 in got}
    assert lengths == {122}, lengths
    # The tail window has to reach the end of the track: dropping the last tokens
    # would desync the closing frames, which is where a listener checks first.
    assert got[-1][1] == REF, got[-1]
    # And the offset has to advance with the window, which is the whole bug: the
    # second window previously started at audio 0 like the first.
    assert got[1][0] > got[0][0], got

    # Degenerate inputs stay harmless rather than raising mid-render.
    assert _audio_span(0, 0, 0, G, R, REF) == (0, REF)
    assert _audio_span(0, TOTAL, TOTAL, G, R, 0) == (0, 0)

    a = torch.zeros((1, 32, 2, REF))
    other = torch.zeros((1, 8, 3))          # not time-shaped: must pass through
    sliced = _slice_rest((a, other), 85, 207, REF)
    assert sliced[0].shape[-1] == 122, sliced[0].shape
    assert sliced[1] is other, "a non-matching stream must not be sliced"
    assert _slice_rest(None, 0, 1, REF) is None

    print("sampler self-check OK")
