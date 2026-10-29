"""Inpainting nodes ported from LanPaint for Cubric Studio (MPI-1343).

Ported from LanPaint (src/LanPaint/nodes.py), pin 9fe91955, GPL-3.0, Copyright (c) LanPaint authors.
See LICENSE-LanPaint. Each node keeps the original's input names, defaults and choices so a workflow
only has to change its class_type. Carried:

  * MpiLanPaintKSampler    - every input of LanPaint_KSampler (both prompt modes, image and video
                             inpainting, the hidden retired LanPaint_MinStepFrac, ignored). The
                             inner loop is in langevin.py, the sampler hooks in sampling.py.
  * MpiLanPaintImageEncode - as LanPaint_ImageEncode (4D and 5D latents, optional mask).
  * MpiLanPaintImageDecode - as LanPaint_ImageDecode (optional image and mask, blend_overlap).

Not carried: the other LanPaint nodes (KSamplerAdvanced, SamplerCustom, SamplerCustomAdvanced, MaskBlend,
MiniMax audio, AV, video mask editor), the early-stop hook, videomask.py / videometa.py (video mask
editor and its HTTP routes) and the web/ frontend. The original's registrations are not touched;
ModelPatcher attributes and model_options keys carry an Mpi prefix, so both packs can load together.

Needs only torch and comfy (engine-owned).
"""
import torch

import nodes  # type: ignore
import comfy.samplers  # type: ignore

from .sampling import override_sample_function

CATEGORY = "MpiNodes/Sampling"

KSAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
                  "dpm_fast", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm",
                  "deis", "res_multistep", "res_multistep_ancestral",
                  "gradient_estimation", "er_sde", "seeds_2", "seeds_3"]

IMAGE_MODE = "\U0001F5BC️ Image Inpainting"
VIDEO_MODE = "\U0001F3AC Video Inpainting"


def _sanitize_param(value, default, allowed=None):
    """Coerce a widget value to a sane default (old workflows can carry values that no longer fit)."""
    if allowed is not None:
        return value if value in allowed else default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value


class MpiLanPaintKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (KSAMPLER_NAMES, {"tooltip": "Recommended: euler."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", "tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "LanPaint_NumSteps": ("INT", {"default": 5, "min": 0, "max": 100, "tooltip": "The number of steps for the Langevin dynamics, representing the turns of thinking per step."}),
                "LanPaint_PromptMode": (["Image First", "Prompt First"], {"tooltip": "Image First: emphasis image quality, Prompt First: emphasis prompt following"}),
                "LanPaint_Info": ("STRING", {"default": "LanPaint KSampler.", "tooltip": "For more info, visit https://github.com/scraed/LanPaint. If you find it useful, please give a star ⭐️!"}),
                "Inpainting_mode": ([IMAGE_MODE, VIDEO_MODE], {"default": IMAGE_MODE, "tooltip": "Choose Image mode for photos or Video mode for video frames with temporal consistency"}),
            },
            "hidden": {
                # Retired hyperparameter. Old prompts still pass it; accepted and ignored.
                "LanPaint_MinStepFrac": "DEFAULT",
            },
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = CATEGORY
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, LanPaint_NumSteps=5, LanPaint_PromptMode="Image First", LanPaint_Info="", Inpainting_mode=IMAGE_MODE, **kwargs):
        LanPaint_NumSteps = _sanitize_param(LanPaint_NumSteps, 5)
        LanPaint_PromptMode = _sanitize_param(LanPaint_PromptMode, "Image First", allowed=("Image First", "Prompt First"))
        Inpainting_mode = _sanitize_param(Inpainting_mode, IMAGE_MODE, allowed=(IMAGE_MODE, VIDEO_MODE))

        model.MpiLanPaint_StepSize = 0.2
        model.MpiLanPaint_Lambda = 5.0
        model.MpiLanPaint_Beta = 1.
        model.MpiLanPaint_NumSteps = LanPaint_NumSteps
        model.MpiLanPaint_MinStepFrac = 1.0
        model.MpiLanPaint_Friction = 15.
        model.MpiLanPaint_EarlyStop = 1
        if LanPaint_PromptMode == "Image First":
            model.MpiLanPaint_cfg_BIG = cfg
        else:
            model.MpiLanPaint_cfg_BIG = 0 * cfg - 0.5

        video_inpainting = (Inpainting_mode == VIDEO_MODE)
        if not hasattr(model, 'model_options') or model.model_options is None:
            model.model_options = {}
        model.model_options["mpi_lanpaint_video_inpainting"] = video_inpainting

        with override_sample_function():
            return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)


def gaussian_kernel_2d(kernel_size):
    """2D Gaussian kernel, sigma = (size - 1) / 4 (same as LanPaint's MaskBlend)."""
    if kernel_size <= 1:
        return torch.ones(1, 1)  # no blending: identity kernel
    sigma = (kernel_size - 1) / 4
    x = torch.arange(kernel_size).float() - kernel_size // 2
    x_grid, y_grid = torch.meshgrid(x, x, indexing="ij")
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def merge_video_with_mask(orig, inpainted, mask, blend_overlap):
    """Blend the inpainted frames into the original with a MaskBlend-style
    boundary: dilate the mask (max_pool), smooth it (Gaussian), then
    ``orig * (1 - m) + inpainted * m``."""
    m = mask.float()
    if m.ndim == 4:  # [F, 1, H, W] from SetLatentNoiseMask
        m = m[:, 0]
    elif m.ndim == 2:  # single image mask [H, W]
        m = m.unsqueeze(0)
    count = min(orig.shape[0], inpainted.shape[0])
    orig = orig[:count]
    inpainted = inpainted[:count]
    if m.shape[0] == 1:  # one mask frame covers every frame (image case)
        m = m[:1].expand(count, -1, -1)
    elif m.shape[0] < count:
        raise ValueError("the mask has fewer frames than the image")
    else:
        m = m[:count]
    if tuple(m.shape[1:]) != tuple(orig.shape[1:3]):
        m = torch.nn.functional.interpolate(
            m.unsqueeze(1), size=orig.shape[1:3], mode="nearest-exact"
        )[:, 0]
    m = m.unsqueeze(1)
    m = torch.nn.functional.max_pool2d(
        m, kernel_size=blend_overlap, stride=1, padding=blend_overlap // 2
    )
    kernel = gaussian_kernel_2d(blend_overlap).to(orig.device)[None, None]
    m = torch.nn.functional.conv2d(m, kernel, padding=blend_overlap // 2)[:, 0]
    return orig * (1 - m[..., None]) + inpainted * m[..., None]


class MpiLanPaintImageEncode:
    """Encode an image and attach the inpainting mask to the latent.

    The mask is snapped to the latent's actual spatial size (nearest-exact). Works with image VAEs
    (4D latent) and video VAEs encoding a single image (5D latent [B, C, T, H, W]). Without a mask
    this is a plain encode.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to encode (1 = regenerate region comes from the mask)."}),
                "vae": ("VAE", {"tooltip": "The VAE."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Inpainting mask [H, W] (1 = regenerate, 0 = keep). Snapped to the latent size automatically."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Encode an image and attach an inpainting mask to the latent (replaces VAEEncode + SetLatentNoiseMask)."

    def encode(self, image, vae, mask=None):
        z = vae.encode(image)
        ndim = len(z.shape)
        if ndim not in (4, 5):
            raise ValueError(
                f"MpiLanPaintImageEncode expects a 4D or 5D latent, got {ndim}D"
            )
        latent = {"samples": z}
        if mask is not None:
            m = mask.float()
            if m.ndim == 4:  # [1, 1, H, W] from SetLatentNoiseMask
                m = m[0, 0]
            elif m.ndim == 3:
                m = m[0]
            if ndim == 4:
                target = z.shape[-2:]
                if tuple(m.shape) != target:
                    m = torch.nn.functional.interpolate(
                        m.unsqueeze(0).unsqueeze(0), size=target, mode="nearest-exact"
                    )[0, 0]
                latent["noise_mask"] = m.unsqueeze(0).unsqueeze(0)
            else:  # 5D video-style VAE: snap to (T, H, W), one mask slice per latent frame
                t, h, w = z.shape[-3:]
                if tuple(m.shape) != (h, w):
                    m = torch.nn.functional.interpolate(
                        m.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest-exact"
                    )[0, 0]
                latent["noise_mask"] = m.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(1, 1, t, h, w)
        return (latent,)


class MpiLanPaintImageDecode:
    """Decode an inpainted latent and merge it with the original image.

    The decoded image is resized to the original's exact dimensions, then the inpainted pixels
    replace the original only inside the mask, with a MaskBlend-style boundary (``blend_overlap``
    pixels). Without an image this is a plain decode.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The inpainted latent to decode."}),
                "vae": ("VAE", {"tooltip": "The VAE."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The original image. When given, the decoded output is resized to its exact dimensions."}),
                "mask": ("MASK", {"tooltip": "The inpainting mask (1 = take the inpainted pixels, 0 = keep the original)."}),
                "blend_overlap": ("INT", {"default": 9, "min": 1, "max": 51, "step": 2, "tooltip": "Boundary blend width in pixels between the inpainted and original image (MaskBlend-style)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Decode an inpainted latent, resize to the original image's exact dimensions, and merge with the original inside the mask (replaces VAEDecode + MaskBlend)."

    def decode(self, samples, vae, image=None, mask=None, blend_overlap=9):
        z = samples["samples"]
        img = vae.decode(z)
        if len(img.shape) == 5:  # [1, F, H, W, C]: combine batches (video-style VAE)
            img = img.reshape(
                -1, img.shape[-3], img.shape[-2], img.shape[-1]
            )
        if image is None:
            return (img,)
        target_h, target_w = image.shape[1], image.shape[2]
        if tuple(img.shape[1:3]) != (target_h, target_w):
            img = torch.nn.functional.interpolate(
                img.movedim(-1, 1),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
        if mask is None:
            return (img,)
        return (merge_video_with_mask(image, img, mask, blend_overlap),)


NODE_CLASS_MAPPINGS = {
    "MpiLanPaintKSampler": MpiLanPaintKSampler,
    "MpiLanPaintImageEncode": MpiLanPaintImageEncode,
    "MpiLanPaintImageDecode": MpiLanPaintImageDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiLanPaintKSampler": "Mpi LanPaint KSampler",
    "MpiLanPaintImageEncode": "Mpi LanPaint Image Encode",
    "MpiLanPaintImageDecode": "Mpi LanPaint Image Decode",
}
