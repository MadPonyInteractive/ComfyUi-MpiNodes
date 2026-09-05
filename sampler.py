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


def plan_windows(total, window, overlap):
    """Cover [0, total) with windows of `window` frames overlapping by `overlap`.

    The last window is pulled back to full width rather than left short, so every
    window is the same size the card was measured against -- a runt final window
    would be cheap, but a short window is not the failure mode worth guarding.
    """
    if window >= total:
        return [(0, total)]

    step = max(1, window - overlap)
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


def _blend_weights(start, end, total, overlap, device, dtype):
    """Linear ramp in, flat, linear ramp out -- but only on interior edges.

    The clip's true first and last frames must keep full weight or the ends fade
    toward whatever the accumulator was initialised with.
    """
    n = end - start
    w = torch.ones(n, device=device, dtype=dtype)
    if overlap <= 0:
        return w

    ramp = min(overlap, n)
    if start > 0:
        w[:ramp] = torch.arange(1, ramp + 1, device=device, dtype=dtype) / (ramp + 1)
    if end < total:
        w[-ramp:] = torch.arange(ramp, 0, -1, device=device, dtype=dtype) / (ramp + 1)
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
                "window": ("INT", {
                    "default": 27, "min": 1, "max": 4096,
                    "tooltip": "Latent frames sampled at once. Set it to the largest "
                               "your card handles: 27 is what a 16GB card manages for "
                               "MiniMax H3 at 2K. A clip at or under this size takes "
                               "the plain single-pass path.",
                }),
                "overlap": ("INT", {
                    "default": 4, "min": 0, "max": 256,
                    "tooltip": "Latent frames shared between neighbouring windows and "
                               "cross-faded. Costs compute on every seam, so raise it "
                               "only if a seam is visible.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    CATEGORY = "MpiNodes/Sampling"
    DESCRIPTION = (
        "Drop-in replacement for SamplerCustomAdvanced that samples the latent in "
        "overlapping temporal windows and cross-fades them, so a clip too long to "
        "sample in one pass still fits. Intended for a REFINE pass over an already "
        "coherent latent, not for first-pass generation. Audio in a joint AV latent "
        "is passed through untouched. Falls back to a single ordinary sample when "
        "the clip already fits, so short clips are unchanged."
    )
    FUNCTION = "doit"

    def doit(self, noise, guider, sampler, sigmas, latent_image, window, overlap):
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

        if overlap >= window:
            overlap = max(0, window - 1)

        spans = plan_windows(total, window, overlap)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # One noise field for the whole clip, sliced per window. Generating noise
        # per window instead would hand the two sides of a seam different noise,
        # so the cross-fade would be averaging two genuinely different denoisings
        # rather than two near-identical ones.
        full_noise = noise.generate_noise(latent)
        noise_video, noise_rest = _split_av(full_noise)

        if len(spans) == 1:
            return self._sample_once(
                guider, sampler, sigmas, latent, full_noise, noise.seed, disable_pbar)

        print(f"[MpiWindowedSampler] T={total} windows={len(spans)} "
              f"window={window} overlap={overlap} spans={spans}")

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
            seg_latent["samples"] = _rejoin_av(seg_video, audio)

            mask = latent.get("noise_mask", None)
            if mask is not None:
                seg_latent["noise_mask"] = self._slice_mask(mask, start, end, total)

            seg_noise = _rejoin_av(noise_video[:, :, start:end], noise_rest)

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
            w = _blend_weights(start, end, total, overlap,
                               out_full.device, out_full.dtype)
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
            return (out, out)

        # process_latent_out is affine, so running it once on the blended x0 is the
        # same as blending per-window results -- and keeps one call, as core does.
        out_denoised = latent.copy()
        out_denoised.pop("downscale_ratio_spacial", None)
        out_denoised.pop("downscale_ratio_temporal", None)
        out_denoised["samples"] = _rejoin_av(
            guider.model_patcher.model.process_latent_out(x0_full.cpu()), audio)
        return (out, out_denoised)

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
