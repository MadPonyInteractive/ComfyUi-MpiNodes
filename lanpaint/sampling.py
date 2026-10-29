"""LanPaint sampler hooks, ported from LanPaint (src/LanPaint/nodes.py) for Cubric Studio (MPI-1343).

GPL-3.0, Copyright (c) LanPaint authors, pin 9fe91955. See LICENSE-LanPaint. Carried as is except:

  * Only what MpiLanPaintKSampler reaches: the guider hooks (outer_sample / predict_noise), the
    inpaint-aware KSAMPLER.sample, the denoise-mask preparation and the per-step wrapper. The Advanced,
    SamplerCustom, MiniMax audio, AV and video-mask-editor nodes, the early-stop hook and the UpSale
    node are not carried.
  * Everything the original keys by a plain string is renamed so both packs can be loaded at once:
    the ModelPatcher attributes are MpiLanPaint_* (the original uses LanPaint_*) and the model_options
    flag is "mpi_lanpaint_video_inpainting".
  * The hooks are applied only inside override_sample_function() and restored on exit, exactly like the
    original; nothing stays patched, and the original pack's own functions are never touched.
"""
import logging
import math
from contextlib import contextmanager

import torch
import comfy  # type: ignore
import comfy.sampler_helpers  # type: ignore
import comfy.samplers  # type: ignore
from comfy.utils import repeat_to_batch_size  # type: ignore
from comfy.samplers import calc_cond_batch, cfg_function, cast_to_load_options  # type: ignore
from comfy.model_base import ModelType, WAN22  # type: ignore

from .langevin import LanPaint

try:
    import comfyui_version  # type: ignore
    _VERSION = comfyui_version.__version__
except ImportError:
    _VERSION = "0.0.0"

try:
    from comfy.ldm.minimax.model import time_shift_sigma, time_shift_slope  # type: ignore
except Exception:  # older engines do not have the MiniMax H3 model
    time_shift_sigma = None
    time_shift_slope = None

# MiniMaxH3 was ModelType.FLOW before ComfyUI 0.31.0 and ModelType.FLOW_AV afterwards. Both are
# rectified-flow schedules; treat them alike.
FLOW_MODEL_TYPES = (ModelType.FLOW, getattr(ModelType, "FLOW_AV", None))


def _detect_minimax_h3_audio(model_patcher, model_options, latent_shapes):
    """Detect a MiniMax H3 AV pack: a nested (video, audio) latent whose audio
    stream rides on a shifted sigma schedule. The diffusion model's own
    sigma_shift_video/sigma_shift_audio attributes mark the H3 schedule
    machinery; every other model keeps the plain video-schedule dynamics.
    Returns (latent_shapes, shift_v, shift_a) or None."""
    if latent_shapes is None or len(latent_shapes) < 2:
        return None
    model = getattr(model_patcher, "model", None)
    diff_model = getattr(model, "diffusion_model", None)
    shift_v = getattr(diff_model, "sigma_shift_video", None)
    shift_a = getattr(diff_model, "sigma_shift_audio", None)
    if shift_v is None or shift_a is None:
        return None
    topts = model_options.get("transformer_options", {}) if isinstance(model_options, dict) else {}
    shift_v = topts.get("minimax_h3_sigma_shift_video", shift_v)
    shift_a = topts.get("minimax_h3_sigma_shift_audio", shift_a)
    return (latent_shapes, float(shift_v), float(shift_a))


def _version_tuple(value):
    return tuple(int(part) if part.isdigit() else 0 for part in value.split("."))


COMFYUI_VERSION_060_OR_NEWER = _version_tuple(_VERSION) >= (0, 6, 0)


def reshape_mask(input_mask, output_shape, video_inpainting=False):
    dims = len(output_shape) - 2
    scale_mode = "nearest-exact"

    if video_inpainting:
        # video masks -> (1, 1, F, H, W): raw [F, H, W], or the [F, 1, H, W]
        # shape SetLatentNoiseMask produces (frames at batch); a plain
        # [H, W] image mask is broadcast over the whole video; 5D passes
        if input_mask.ndim == 3:
            input_mask = input_mask.unsqueeze(0).unsqueeze(1)
        elif input_mask.ndim == 4:
            input_mask = input_mask.permute(1, 0, 2, 3).unsqueeze(0)
        elif input_mask.ndim == 2:
            input_mask = input_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif input_mask.ndim == 1 and len(output_shape) == 4:
        # audio mask [F] at video frame rate -> the audio latent's time axis
        t = output_shape[-1]
        input_mask = torch.nn.functional.interpolate(
            input_mask.float().unsqueeze(0).unsqueeze(0),
            size=(t,),
            mode="nearest-exact",
        )
        input_mask = input_mask.expand(1, 1, output_shape[-2], t)
    elif input_mask.ndim == 4 and len(output_shape) == 4 and input_mask.shape[1] == 1 and input_mask.shape[3] == 1:
        # audio mask [F, 1] that SetLatentNoiseMask reshaped to [1, 1, F, 1]
        t = output_shape[-1]
        input_mask = torch.nn.functional.interpolate(input_mask, size=(t, 1), mode="nearest-exact")
        input_mask = input_mask.permute(0, 1, 3, 2).expand(1, 1, output_shape[-2], t)
    elif input_mask.ndim == 2:
        input_mask = input_mask.unsqueeze(0).unsqueeze(0)
    elif input_mask.ndim == 3:
        input_mask = input_mask.unsqueeze(1)

    # Handle 5D output shape (B, C, F, H, W) by ensuring input is 5D
    if len(output_shape) == 5 and input_mask.ndim == 4:
        if COMFYUI_VERSION_060_OR_NEWER:
            input_mask = input_mask.unsqueeze(2)  # (B, C, 1, H, W)

    if video_inpainting:  # (batch, channels, frames, height, width)
        target_frames = output_shape[2]
        target_height, target_width = output_shape[-2:]

        # nearest-exact first, then the slice-level union: the resample snaps each latent slice
        # to one picked frame, then the sliding max over ~5 slices spreads the value to neighbours.
        input_mask = torch.nn.functional.interpolate(
            input_mask,
            size=(target_frames, target_height, target_width),
            mode=scale_mode,
        )

        mask = torch.nn.functional.max_pool3d(
            input_mask, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0))
        if mask.shape[1] < output_shape[1]:
            mask = mask.repeat(1, output_shape[1], 1, 1, 1)[:, :output_shape[1]]
        mask = repeat_to_batch_size(mask, output_shape[0])
    else:  # 2D image case
        if not COMFYUI_VERSION_060_OR_NEWER:
            mask = torch.nn.functional.interpolate(input_mask, size=output_shape[-2:], mode=scale_mode)
        else:
            mask = torch.nn.functional.interpolate(input_mask, size=output_shape[2:], mode=scale_mode)
        if mask.shape[1] < output_shape[1]:
            mask = mask.repeat((1, output_shape[1]) + (1,) * dims)[:, :output_shape[1]]
        mask = repeat_to_batch_size(mask, output_shape[0])

    return mask


def min_step_frac_effective_steps(n_steps, frac, min_frac):
    """Inner-step count under the MinStepFrac tail ramp.

    While the remaining-noise fraction stays above min_frac (or with
    min_frac == 0, the feature disabled) the count is unchanged and the step
    size keeps scaling with the noise inside LanPaint. Below the fraction the
    step size is pinned and the count ramps down linearly,
    NSteps * frac / min_frac, to zero."""
    if min_frac <= 0 or frac >= min_frac or n_steps <= 0:
        return n_steps
    return max(0, round(n_steps * frac / min_frac))


def prepare_mask(noise_mask, shape, device, video_inpainting=False):
    return reshape_mask(noise_mask, shape, video_inpainting).to(device)


def sampling_function_LanPaint(model, x, timestep, uncond, cond, cond_scale, cond_scale_BIG, model_options={}, seed=None):
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    out = calc_cond_batch(model, conds, x, timestep, model_options)

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds": conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = fn(args)

    return cfg_function(model, out[0], out[1], cond_scale, x, timestep, model_options=model_options, cond=cond, uncond=uncond_), cfg_function(model, out[0], out[1], cond_scale_BIG, x, timestep, model_options=model_options, cond=cond, uncond=uncond_)


class CFGGuider_LanPaint:
    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None, **kwargs):
        logging.debug("MpiLanPaint CFGGuider outer_sample")
        self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
        device = self.model_patcher.load_device

        if isinstance(self.inner_model, WAN22):
            self.inner_model.extra_conds = super(WAN22, self.inner_model).extra_conds

        # MiniMax H3 AV packs carry an audio stream on a shifted sigma schedule;
        # the paint loop needs the per-stream times only for that exact model
        self.minimax_h3_audio = _detect_minimax_h3_audio(
            self.model_patcher, self.model_options, kwargs.get("latent_shapes", None))

        if denoise_mask is not None:
            video_inpainting = self.model_options.get("mpi_lanpaint_video_inpainting", False)
            if tuple(denoise_mask.shape) != tuple(noise.shape):
                denoise_mask = prepare_mask(denoise_mask, noise.shape, device, video_inpainting)

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)
        cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, **kwargs)
        finally:
            self.model_patcher.cleanup()

        comfy.sampler_helpers.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.loaded_models
        return output

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return sampling_function_LanPaint(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, self.cfg_BIG, model_options=model_options, seed=seed)


class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas
        self.audio_indicator = None  # [1, 1, N] flat pack: 1 on audio rows (MiniMax H3 only)
        self.audio_shifts = None  # (shift_video, shift_audio) for the audio sigma schedule

    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None, **kwargs):
        ### For 1.5 and XL model
        # x is x_t in the notation of variance exploding diffusion model, x_t = x_0 + sigma * noise
        # sigma is the noise level
        ### For flux model
        # x is rectified flow x_t = sigma * noise + (1.0 - sigma) * x_0

        IS_FLUX = self.inner_model.inner_model.model_type == ModelType.FLUX
        IS_FLOW = self.inner_model.inner_model.model_type in FLOW_MODEL_TYPES
        # unify the notations into variance exploding diffusion model
        if IS_FLUX or IS_FLOW:
            Flow_t = sigma
            abt = (1 - Flow_t)**2 / ((1 - Flow_t)**2 + Flow_t**2)
            VE_Sigma = Flow_t / (1 - Flow_t)
        else:
            VE_Sigma = sigma
            abt = 1 / (1 + VE_Sigma**2)
            Flow_t = (1 - abt)**0.5 / ((1 - abt)**0.5 + abt**0.5)

        # MiniMax H3 audio stream runs on its own shifted sigma schedule:
        # sigma_audio = time_shift_sigma(sigma_video, shift_video, shift_audio).
        current_times_audio = None
        audio_correction = None
        if self.audio_indicator is not None and self.audio_shifts is not None and time_shift_sigma is not None:
            shift_v, shift_a = self.audio_shifts
            Flow_a = time_shift_sigma(Flow_t, shift_v, shift_a)
            abt_a = (1 - Flow_a)**2 / ((1 - Flow_a)**2 + Flow_a**2)
            VE_a = Flow_a / (1 - Flow_a)
            current_times_audio = (VE_a, abt_a, Flow_a)
            # Langevin target correction: x0_true = x + c*(x0_flat - x) with
            # c = sigma_a / (sigma_v * slope_a) on audio rows, 1 on video rows.
            ft = float(Flow_t)
            c = 1.0
            if ft > 1e-4 and time_shift_slope is not None:
                slope_a = time_shift_slope(Flow_t, shift_v, shift_a)
                c = float(Flow_a) / (ft * float(slope_a))
            audio_correction = (1.0 - self.audio_indicator) + c * self.audio_indicator

        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})

            denoise_mask = (denoise_mask > 0.5).float()

            latent_mask = 1 - denoise_mask
            current_times = (VE_Sigma, abt, Flow_t)

            current_step = torch.argmin(torch.abs(self.sigmas - torch.mean(sigma)))
            total_steps = len(self.sigmas) - 1

            # While the remaining noise fraction (1 - abt) stays above MinStepFrac the inner step
            # size scales with it and the count is constant; below it the count ramps down.
            n_eff = self.PaintMethod.n_steps
            if total_steps - current_step <= self.LanPaint_early_stop:
                n_eff = 0
            else:
                n_eff = min_step_frac_effective_steps(
                    n_eff, float((1.0 - abt).mean()), getattr(self, "LanPaint_min_step_frac", 1.0))
            out = self.PaintMethod(x, self.latent_image, self.noise, sigma, latent_mask, current_times, model_options, seed, n_steps=n_eff, current_times_audio=current_times_audio, audio_indicator=self.audio_indicator, audio_correction=audio_correction)
        else:
            out, _ = self.inner_model(x, sigma, model_options=model_options, seed=seed)

        # TAESD preview support - call the preview callback if it exists
        current_step = model_options.get("i", kwargs.get("i", 0))
        total_steps = model_options.get("total_steps", 0)

        if current_step % 2 == 0:
            callback = model_options.get("callback", None)
            if callback is not None:
                callback({"i": current_step, "denoised": out, "x": x})

        return out


# Custom sampler class extending ComfyUI's KSAMPLER for LanPaint
class KSAMPLER(comfy.samplers.KSAMPLER):
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        # noise here is a randn noise from comfy.sample.prepare_noise
        # latent_image is the latent image as input of the KSampler node. For inpainting, it is the masked latent image.
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False):
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        IS_FLUX = model_wrap.inner_model.model_type == ModelType.FLUX
        IS_FLOW = model_wrap.inner_model.model_type in FLOW_MODEL_TYPES
        if IS_FLUX:
            model_wrap.cfg_BIG = 1.0
        else:
            model_wrap.cfg_BIG = model_wrap.model_patcher.MpiLanPaint_cfg_BIG
        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

        # MiniMax H3 AV pack: mark the audio rows of the flat pack so the paint
        # loop can apply the audio's own shifted sigma schedule to them
        audio_layout = getattr(model_wrap, "minimax_h3_audio", None)
        if audio_layout is not None and time_shift_sigma is not None:
            latent_shapes, shift_v, shift_a = audio_layout
            video_n = math.prod(latent_shapes[0][1:])
            audio_indicator = torch.zeros(noise.shape, dtype=torch.float32, device=noise.device)
            audio_indicator[..., video_n:] = 1.0
            model_k.audio_indicator = audio_indicator
            model_k.audio_shifts = (shift_v, shift_a)

        patcher = model_wrap.model_patcher
        model_k.PaintMethod = LanPaint(model_k.inner_model,
                                       patcher.MpiLanPaint_NumSteps,
                                       patcher.MpiLanPaint_Friction,
                                       patcher.MpiLanPaint_Lambda,
                                       patcher.MpiLanPaint_Beta,
                                       patcher.MpiLanPaint_StepSize,
                                       IS_FLUX=IS_FLUX,
                                       IS_FLOW=IS_FLOW,
                                       MinStepFrac=getattr(patcher, "MpiLanPaint_MinStepFrac", 1.0))
        model_k.LanPaint_early_stop = patcher.MpiLanPaint_EarlyStop
        model_k.LanPaint_min_step_frac = getattr(patcher, "MpiLanPaint_MinStepFrac", 1.0)
        # if not inpainting, after noise_scaling, noise = noise * sigma; if inpainting,
        # noise = latent_image + noise * sigma, which is x_t for the known region.
        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)
        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples


_override_active = False


@contextmanager
def override_sample_function():
    # guard against nested entry: a second entry must not capture the already
    # patched functions as "originals" (that would restore the patches, not
    # the real functions, on exit)
    global _override_active
    if _override_active:
        yield
        return
    _override_active = True
    try:
        original_outer_sample = comfy.samplers.CFGGuider.outer_sample
        comfy.samplers.CFGGuider.outer_sample = CFGGuider_LanPaint.outer_sample

        original_predict_noise = comfy.samplers.CFGGuider.predict_noise
        comfy.samplers.CFGGuider.predict_noise = CFGGuider_LanPaint.predict_noise

        original_sample = comfy.samplers.KSAMPLER.sample
        comfy.samplers.KSAMPLER.sample = KSAMPLER.sample

        # Route the stock per-stream mask prep (inside CFGGuider.sample) through
        # our prepare_mask so video masks get the temporal-union aggregation.
        original_prepare_mask = comfy.sampler_helpers.prepare_mask

        def _prepare_mask_with_union(noise_mask, shape, device):
            return prepare_mask(noise_mask, shape, device, video_inpainting=(len(shape) == 5))

        comfy.sampler_helpers.prepare_mask = _prepare_mask_with_union

        yield
    finally:
        comfy.sampler_helpers.prepare_mask = original_prepare_mask
        comfy.samplers.KSAMPLER.sample = original_sample
        comfy.samplers.CFGGuider.predict_noise = original_predict_noise
        comfy.samplers.CFGGuider.outer_sample = original_outer_sample
        _override_active = False
