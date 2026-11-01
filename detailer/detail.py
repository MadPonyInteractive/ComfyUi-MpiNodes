# The detail pass: crop every SEG, enlarge it, inpaint it with a short sampler run, shrink it back and
# paste it through the feathered mask.
#
# Ported from ComfyUI-Impact-Pack (GPL-3.0: modules/impact/core.py enhance_detail,
# modules/impact/impact_pack.py DetailerForEach.do_detail, modules/impact/impact_sampling.py).
# Not ported: refiner model, detailer hooks, ControlNet/IPAdapter SEG wrappers, the "DUMMY" model and
# NegativePlaceholder (Flux guider) paths, AYS/GITS/OSS scheduler extras, wildcard files / {a|b} /
# <lora:> expansion (the wildcard text is a plain prompt; BREAK and # comment lines still work).
# comfy is imported inside the functions that need it, so this module loads without ComfyUI.
import logging
import math
import time

import torch

from . import segs as S

SDE_SAMPLERS = ["dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu"]


# --- sampling ---------------------------------------------------------------------------------

def calculate_sigmas(model, sampler, scheduler, steps):
    import comfy.samplers

    discard_penultimate_sigma = False
    if sampler in ["dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"]:
        steps += 1
        discard_penultimate_sigma = True

    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


def get_noise_sampler(x, cpu, total_sigmas, **kwargs):
    from comfy.k_diffusion import sampling as k_diffusion_sampling

    if "extra_args" in kwargs and "seed" in kwargs["extra_args"]:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs["extra_args"].get("seed", None)
        return k_diffusion_sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=cpu)
    return None


def make_sampler(sampler_name, total_sigmas):
    """comfy's sampler object; the SDE samplers get a Brownian-tree noise sampler seeded over the whole schedule."""
    import comfy.samplers
    from comfy.k_diffusion import sampling as k_diffusion_sampling

    if sampler_name not in SDE_SAMPLERS:
        return comfy.samplers.sampler_object(sampler_name)

    orig_sampler_function = getattr(k_diffusion_sampling, f"sample_{sampler_name}")

    def sampler_function_wrapper(model, x, sigmas, **kwargs):
        if "noise_sampler" not in kwargs:
            kwargs["noise_sampler"] = get_noise_sampler(x, "gpu" not in sampler_name, total_sigmas, **kwargs)
        return orig_sampler_function(model, x, sigmas, **kwargs)

    return comfy.samplers.KSAMPLER(sampler_function_wrapper, {}, {})


def sample_with_custom_noise(model, noise_seed, cfg, positive, negative, sampler, sigmas, latent):
    """SamplerCustom with seeded random noise. Returns (latent, denoised latent)."""
    import comfy.model_management as mm
    import comfy.sample
    import comfy.utils
    import latent_preview

    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    out = latent.copy()
    out["samples"] = latent_image

    noise = comfy.sample.prepare_noise(latent_image, noise_seed, out["batch_index"] if "batch_index" in out else None)

    noise_mask = latent["noise_mask"] if "noise_mask" in latent else None

    x0_output = {}
    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    device = mm.get_torch_device()
    noise = noise.to(device)
    latent_image = latent_image.to(device)
    if noise_mask is not None:
        noise_mask = noise_mask.to(device)

    samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                                         noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar,
                                         seed=noise_seed)
    samples = samples.to(mm.intermediate_device())

    out["samples"] = samples
    if "x0" in x0_output:
        out_denoised = latent.copy()
        out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
    else:
        out_denoised = out
    return out, out_denoised


def ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
    """Impact's img2img sample: `steps` real steps at `denoise`, run as the tail of floor(steps/denoise) steps."""
    advanced_steps = math.floor(steps / denoise)
    start_at_step = advanced_steps - steps
    end_at_step = start_at_step + steps

    total_sigmas = calculate_sigmas(model, sampler_name, scheduler, advanced_steps)
    sigmas = total_sigmas

    if end_at_step < (len(total_sigmas) - 1):
        sigmas = total_sigmas[:end_at_step + 1]
        sigmas[-1] = 0

    if start_at_step < (len(sigmas) - 1):
        sigmas = sigmas[start_at_step:].clone()
    else:
        return latent_image

    sampler = make_sampler(sampler_name, total_sigmas)

    if len(sigmas) == 0 or (len(sigmas) == 1 and sigmas[0] == 0):
        return latent_image

    return sample_with_custom_noise(model, seed, cfg, positive, negative, sampler, sigmas, latent_image)[1]


# --- one SEG ----------------------------------------------------------------------------------

def encode_wildcard_prompt(text, clip):
    """CLIP-encode the per-SEG prompt: '#' comment lines dropped, 'BREAK' splits into chunks that are concatenated."""
    import nodes

    text = S.strip_comment_lines(text)
    result = None
    for prompt in S.split_break(text):
        cur = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        result = cur if result is None else nodes.ConditioningConcat().concat(result, cur)[0]
    return result


def apply_differential_diffusion(model):
    from comfy_extras import nodes_differential_diffusion

    return nodes_differential_diffusion.DifferentialDiffusion.execute(model)[0]


def sample_size(w, h, bbox, guide_size, guide_size_for_bbox, max_size, force_inpaint):
    """(upscale, new_w, new_h) the crop is sampled at, or None when the SEG is skipped.

    max_size caps the SAMPLED size. With force_inpaint a crop larger than max_size shrinks to it
    (Impact sampled it at native size: 3375 px for a face cut 1:1 out of a 16K photo, MPI-1392);
    a crop already under max_size keeps its own size."""
    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # Skip processing if the detected bbox is already larger than the guide_size
    if not force_inpaint and bbox_h >= guide_size and bbox_w >= guide_size:
        logging.info("Detailer: segment skip (enough big)")
        return None

    if guide_size_for_bbox:  # == "bbox"
        # Scale up based on the smaller dimension between width and height.
        upscale = guide_size / min(bbox_w, bbox_h)
    else:
        # for cropped_size
        upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if not force_inpaint:
        if upscale <= 1.0:
            logging.info(f"Detailer: segment skip [determined upscale factor={upscale}]")
            return None

        if new_w == 0 or new_h == 0:
            logging.info(f"Detailer: segment skip [zero size={new_w, new_h}]")
            return None
    else:
        if upscale <= 1.0 or new_w == 0 or new_h == 0:
            logging.info("Detailer: force inpaint")
            upscale = min(1.0, max_size / max(w, h))
            new_w = int(w * upscale)
            new_h = int(h * upscale)

    return upscale, new_w, new_h


def enhance_detail(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                   sampler_name, scheduler, positive, negative, denoise, noise_mask, force_inpaint,
                   wildcard_opt=None, wildcard_opt_concat_mode=None, cycle=1, inpaint_model=False,
                   noise_mask_feather=0, vae_tiled_encode=False, vae_tiled_decode=False):
    """Enlarge one crop to guide_size, resample it, decode, shrink back. (None) when the SEG is skipped."""
    import comfy.model_management as mm
    import nodes

    if noise_mask is not None:
        noise_mask = S.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
        noise_mask = noise_mask.squeeze(3)

    if wildcard_opt is not None and wildcard_opt != "":
        wildcard_positive = encode_wildcard_prompt(wildcard_opt, clip)

        if wildcard_opt_concat_mode == "concat":
            positive = nodes.ConditioningConcat().concat(positive, wildcard_positive)[0]
        else:
            positive = wildcard_positive

    h = image.shape[1]
    w = image.shape[2]

    size = sample_size(w, h, bbox, guide_size, guide_size_for_bbox, max_size, force_inpaint)
    if size is None:
        return None
    upscale, new_w, new_h = size
    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    logging.info(f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}")

    upscaled_image = S.tensor_resize(image, new_w, new_h)

    # prepare mask and latent
    if noise_mask is not None and inpaint_model:
        positive, negative, latent_image = nodes.InpaintModelConditioning().encode(
            positive, negative, upscaled_image, vae, mask=noise_mask, noise_mask=True)
    else:
        start = time.time()
        if vae_tiled_encode:
            latent_image = nodes.VAEEncodeTiled().encode(vae, upscaled_image, 512, overlap=64)[0]
        else:
            latent_image = nodes.VAEEncode().encode(vae, upscaled_image)[0]
        logging.info(f"[MpiDetailer] vae encoded in {time.time() - start:.1f}s")
        if noise_mask is not None:
            latent_image["noise_mask"] = noise_mask

    refined_latent = latent_image
    for i in range(0, cycle):
        refined_latent = ksampler_wrapper(model, seed + i, steps, cfg, sampler_name, scheduler, positive, negative,
                                          refined_latent, denoise)

    # non-latent downscale - latent downscale causes bad quality
    start = time.time()
    if vae_tiled_decode:
        refined_image = nodes.VAEDecodeTiled().decode(vae, refined_latent, 512)[0]
    else:
        try:
            refined_image = vae.decode(refined_latent["samples"])
        except mm.OOM_EXCEPTION:
            logging.warning(f"[MpiDetailer] vae.decode ran out of memory after {time.time() - start:.1f}s, decoding tiled")
            refined_image = vae.decode_tiled(refined_latent["samples"], tile_x=64, tile_y=64)
    logging.info(f"[MpiDetailer] vae decoded in {time.time() - start:.1f}s")

    # a video VAE gives 5 dims
    if len(refined_image.shape) == 5:
        refined_image = refined_image.squeeze(0)

    refined_image = S.tensor_resize(refined_image, w, h)
    return refined_image.cpu()


# --- all SEGS ---------------------------------------------------------------------------------

def do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg, sampler_name,
              scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint, wildcard_opt=None, cycle=1,
              inpaint_model=False, noise_mask_feather=0, tiled_encode=False, tiled_decode=False):
    """-> (image, cropped, enhanced, enhanced_alpha, new_segs). Pastes every SEG's result into a copy of `image`."""
    if len(image) > 1:
        raise Exception("[MpiDetailer] the detailer does not take image batches; run the images one at a time")

    image = image.clone()
    enhanced_alpha_list = []
    enhanced_list = []
    cropped_list = []

    segs = S.segs_scale_match(segs, image.shape)
    new_segs = []

    wildcard_concat_mode = None
    if wildcard_opt is not None:
        if wildcard_opt.startswith("[CONCAT]"):
            wildcard_concat_mode = "concat"
            wildcard_opt = wildcard_opt[8:]
        wmode, wildcard_chooser = S.process_wildcard_for_segs(wildcard_opt)
    else:
        wmode, wildcard_chooser = None, None

    if noise_mask_feather > 0 and "denoise_mask_function" not in model.model_options:
        model = apply_differential_diffusion(model)

    for i, seg in enumerate(segs[1]):
        cropped_image = S.crop_ndarray4(image.cpu().numpy(), seg.crop_region)  # never seg.cropped_image: overlapping areas
        cropped_image = S.to_tensor(cropped_image)
        mask = S.to_tensor(seg.cropped_mask)
        mask = S.tensor_gaussian_blur_mask(mask, feather)

        if (seg.cropped_mask == 0).all().item():
            logging.info("Detailer: segment skip [empty mask]")
            continue

        cropped_mask = seg.cropped_mask if noise_mask else None

        wildcard_item = wildcard_chooser.get(seg) if wildcard_chooser is not None else None
        seg_seed = seed + i

        cropped_positive = [
            [condition, {k: S.crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                         for k, v in details.items()}]
            for condition, details in positive
        ]
        cropped_negative = [
            [condition, {k: S.crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                         for k, v in details.items()}]
            for condition, details in negative
        ]

        if wildcard_item and wildcard_item.strip() == "[SKIP]":
            continue

        if wildcard_item and wildcard_item.strip() == "[STOP]":
            break

        orig_cropped_image = cropped_image.clone()
        enhanced_image = enhance_detail(
            cropped_image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seg.bbox, seg_seed, steps, cfg,
            sampler_name, scheduler, cropped_positive, cropped_negative, denoise, cropped_mask, force_inpaint,
            wildcard_opt=wildcard_item, wildcard_opt_concat_mode=wildcard_concat_mode, cycle=cycle,
            inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, vae_tiled_encode=tiled_encode,
            vae_tiled_decode=tiled_decode)

        if enhanced_image is not None:
            # don't latent composite: converting to latent caused poor quality, so use an image paste
            image = image.cpu()
            enhanced_image = enhanced_image.cpu()
            S.tensor_paste(image, enhanced_image, (seg.crop_region[0], seg.crop_region[1]), mask)
            enhanced_list.append(enhanced_image)

            enhanced_image_alpha = S.tensor_convert_rgba(enhanced_image)
            new_seg_image = enhanced_image.numpy()  # alpha should not be applied to seg_image

            mask = S.tensor_resize(mask, *S.tensor_get_size(enhanced_image))
            S.tensor_putalpha(enhanced_image_alpha, mask)
            enhanced_alpha_list.append(enhanced_image_alpha)
        else:
            new_seg_image = None

        cropped_list.append(orig_cropped_image)

        new_segs.append(S.SEG(new_seg_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label,
                              seg.control_net_wrapper))

    image_tensor = S.tensor_convert_rgb(image)

    cropped_list.sort(key=lambda x: x.shape, reverse=True)
    enhanced_list.sort(key=lambda x: x.shape, reverse=True)
    enhanced_alpha_list.sort(key=lambda x: x.shape, reverse=True)

    return image_tensor, cropped_list, enhanced_list, enhanced_alpha_list, (segs[0], new_segs)
