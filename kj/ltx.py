"""LTX 2.x nodes ported from ComfyUI-KJNodes for Cubric Studio (MPI-1341).

Ported from comfyui-kjnodes (nodes/ltxv_nodes.py, nodes/nodes.py), GPL-3.0, Copyright (c) kijai.
See LICENSE-KJNodes. Input names and accepted values are the originals'.

  * MpiLTX2NAG            - Normalized Attention Guidance for LTX 2 (video and audio cross-attention).
                            Patches attn2.forward / audio_attn2.forward on every transformer block, so it
                            reaches into comfy.ldm.lightricks internals (caption projection, embeddings
                            connectors, to_gate_logits); an LTX release that renames those breaks it.
  * MpiLTXAudioVideoMask  - noise masks for a video latent and an audio latent from time ranges.
  * MpiVAELoader          - loads a VAE file onto a chosen device in a chosen dtype; an LTX audio VAE
                            (vocoder keys) is loaded the way ComfyUI-LTXVideo expects. The original also
                            listed the tiny TAE decoders and "pixel_space"; Cubric Studio uses none of them.

Needs only torch/numpy and ComfyUI itself.
"""
import types

import numpy as np
import torch

import comfy.ldm.modules.attention  # type: ignore
import comfy.model_management as mm  # type: ignore
import folder_paths  # type: ignore
from comfy.utils import load_torch_file, state_dict_prefix_replace  # type: ignore

CATEGORY = "MpiNodes/KJ"


# ------------------------------------------------------------------------ NAG

def _compute_attention(self, query, context, attn_precision=None, transformer_options=None, mask=None):
    """Attention of `query` over `context`. With a mask it bypasses wrap_attn and calls
    attention_pytorch directly, because sage and similar backends drop arbitrary masks."""
    transformer_options = {} if transformer_options is None else transformer_options
    k = self.k_norm(self.to_k(context)).to(query.dtype)
    v = self.to_v(context).to(query.dtype)
    if mask is None:
        x = comfy.ldm.modules.attention.optimized_attention(
            query, k, v, heads=self.heads, attn_precision=attn_precision,
            transformer_options=transformer_options).flatten(2)
    else:
        x = comfy.ldm.modules.attention.attention_pytorch(
            query, k, v, heads=self.heads, mask=mask, attn_precision=attn_precision,
            _inside_attn_wrapper=True, transformer_options=transformer_options).flatten(2)
    del k, v
    return x


def nag_attention(self, query, context_positive, nag_context, attn_precision=None, transformer_options=None, mask=None):
    # a mask (e.g. PromptRelay's temporal routing) only applies to the real context, not the NAG one
    x_positive = _compute_attention(self, query, context_positive, attn_precision, transformer_options, mask=mask)
    x_negative = _compute_attention(self, query, nag_context, attn_precision, transformer_options)
    return x_positive, x_negative


def normalized_attention_guidance(self, x_positive, x_negative):
    """The NAG maths: extrapolate away from the negative branch, clip the L1 norm ratio to nag_tau,
    then mix back toward the positive branch by nag_alpha."""
    if self.inplace:
        nag_guidance = x_negative.mul_(self.nag_scale - 1).neg_().add_(x_positive, alpha=self.nag_scale)
        del x_negative
    else:
        nag_guidance = x_negative * (self.nag_scale - 1)
        del x_negative
        nag_guidance = (x_positive * self.nag_scale).sub_(nag_guidance)

    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
    norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)

    scale = norm_guidance / norm_positive
    torch.nan_to_num_(scale, nan=10.0)
    mask = scale > self.nag_tau
    del scale

    adjustment = (norm_positive * self.nag_tau) / (norm_guidance + 1e-7)
    del norm_positive, norm_guidance

    nag_guidance.mul_(torch.where(mask, adjustment, 1.0))
    del mask, adjustment

    if self.inplace:
        return nag_guidance.sub_(x_positive).mul_(self.nag_alpha).add_(x_positive)
    nag_guidance.mul_(self.nag_alpha)
    return nag_guidance.add_(x_positive * (1 - self.nag_alpha))


def ltxv_crossattn_forward_nag(self, x, context, mask=None, transformer_options=None, **kwargs):
    transformer_options = {} if transformer_options is None else transformer_options
    if mask is None:
        mask_provider = transformer_options.get("promptrelay_mask_fn")
        if mask_provider is not None:
            mask = mask_provider(x.shape[1], context.shape[1], x.dtype, x.device, transformer_options)

    # a single context, or a [positive, negative] pair
    if context.shape[0] == 1:
        x_pos, context_pos = x, context
        x_neg, context_neg = None, None
    else:
        x_pos, x_neg = torch.chunk(x, 2, dim=0)
        context_pos, context_neg = torch.chunk(context, 2, dim=0)

    q_pos = self.q_norm(self.to_q(x_pos))
    del x_pos

    x_positive, x_negative = nag_attention(self, q_pos, context_pos, self.nag_context,
                                           attn_precision=self.attn_precision,
                                           transformer_options=transformer_options, mask=mask)
    del context_pos, q_pos

    x_pos_out = normalized_attention_guidance(self, x_positive, x_negative)
    del x_positive, x_negative

    # the negative half of the batch is plain attention
    if x_neg is not None and context_neg is not None:
        q_neg = self.q_norm(self.to_q(x_neg))
        k_neg = self.k_norm(self.to_k(context_neg))
        v_neg = self.to_v(context_neg)
        x_neg_out = comfy.ldm.modules.attention.optimized_attention(
            q_neg, k_neg, v_neg, heads=self.heads, attn_precision=self.attn_precision,
            transformer_options=transformer_options)
        out = torch.cat([x_pos_out, x_neg_out], dim=0)
    else:
        out = x_pos_out

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)  # (B, T, H)
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)  # zero-init -> identity
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    return self.to_out(out)


class LTXVCrossAttentionPatch:
    def __init__(self, context, nag_scale, nag_alpha, nag_tau, inplace=True):
        self.nag_context = context
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau
        self.inplace = inplace

    def __get__(self, obj, objtype=None):
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.nag_context = self.nag_context
            self_module.nag_scale = self.nag_scale
            self_module.nag_alpha = self.nag_alpha
            self_module.nag_tau = self.nag_tau
            self_module.inplace = self.inplace
            return ltxv_crossattn_forward_nag(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)


class MpiLTX2NAG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.001,
                                        "tooltip": "Strength of negative guidance effect"}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001,
                                        "tooltip": "Mixing coefficient between the normalized guided representation and the original positive representation."}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.001,
                                      "tooltip": "Clipping threshold: how far the guided attention may deviate from the positive attention."}),
            },
            "optional": {
                "nag_cond_video": ("CONDITIONING",),
                "nag_cond_audio": ("CONDITIONING",),
                "inplace": ("BOOLEAN", {"default": True,
                                        "tooltip": "Modify tensors in place to save memory. Gives slightly different numbers."}),
            },
        }

    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    DESCRIPTION = "Normalized Attention Guidance for LTX 2 (https://github.com/ChenDarYen/Normalized-Attention-Guidance)."

    def patch(self, model, nag_scale, nag_alpha, nag_tau, nag_cond_video=None, nag_cond_audio=None, inplace=True):
        if nag_scale == 0:
            return (model,)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = model.model.manual_cast_dtype
        if dtype is None:
            dtype = model.model.diffusion_model.dtype

        model_clone = model.clone()

        diffusion_model = model_clone.get_model_object("diffusion_model")
        img_dim = diffusion_model.inner_dim
        audio_dim = diffusion_model.audio_inner_dim

        if nag_cond_video is not None:
            context_video = nag_cond_video[0][0].to(device, dtype)
            vid_split = diffusion_model.cross_attention_dim if hasattr(diffusion_model, "cross_attention_dim") else None
            if vid_split is not None and context_video.shape[-1] == vid_split + diffusion_model.audio_cross_attention_dim:
                context_video = context_video[:, :, :vid_split]
            if diffusion_model.caption_proj_before_connector and diffusion_model.caption_projection_first_linear:
                diffusion_model.caption_projection.to(device)
                context_video = diffusion_model.caption_projection(context_video)
                diffusion_model.caption_projection.to(offload_device)
            if hasattr(diffusion_model, "video_embeddings_connector"):
                diffusion_model.video_embeddings_connector.to(device)
                context_video = diffusion_model.video_embeddings_connector(context_video)[0]
                diffusion_model.video_embeddings_connector.to(offload_device)
            context_video = context_video.view(1, -1, img_dim)
            for idx, block in enumerate(diffusion_model.transformer_blocks):
                patched = LTXVCrossAttentionPatch(context_video, nag_scale, nag_alpha, nag_tau, inplace=inplace).__get__(block.attn2, block.__class__)
                model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn2.forward", patched)

        if nag_cond_audio is not None and diffusion_model.audio_caption_projection is not None:
            context_audio = nag_cond_audio[0][0].to(device, dtype)
            vid_split = diffusion_model.cross_attention_dim if hasattr(diffusion_model, "cross_attention_dim") else None
            if vid_split is not None and context_audio.shape[-1] == vid_split + diffusion_model.audio_cross_attention_dim:
                context_audio = context_audio[:, :, vid_split:]
            if diffusion_model.caption_proj_before_connector and diffusion_model.caption_projection_first_linear:
                diffusion_model.audio_caption_projection.to(device)
                context_audio = diffusion_model.audio_caption_projection(context_audio)
                diffusion_model.audio_caption_projection.to(offload_device)
            if hasattr(diffusion_model, "audio_embeddings_connector"):
                diffusion_model.audio_embeddings_connector.to(device)
                context_audio = diffusion_model.audio_embeddings_connector(context_audio)[0]
                diffusion_model.audio_embeddings_connector.to(offload_device)
            context_audio = context_audio.view(1, -1, audio_dim)
            for idx, block in enumerate(diffusion_model.transformer_blocks):
                patched = LTXVCrossAttentionPatch(context_audio, nag_scale, nag_alpha, nag_tau, inplace=inplace).__get__(block.audio_attn2, block.__class__)
                model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.audio_attn2.forward", patched)

        return (model_clone,)


# ------------------------------------------------------------ audio/video mask

def _zeros_like_tail(t, frames):
    """`frames` zero frames along dim 2, shaped like t otherwise."""
    shape = list(t.shape)
    shape[2] = frames
    return torch.zeros(shape, dtype=t.dtype, device=t.device)


class MpiLTXAudioVideoMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_fps": ("FLOAT", {"default": 25, "min": 0.0, "max": 100.0, "step": 0.01}),
                "video_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01,
                                               "tooltip": "Start time in seconds for the video mask."}),
                "video_end_time": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10000.0, "step": 0.01,
                                             "tooltip": "End time in seconds for the video mask."}),
                "audio_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01,
                                               "tooltip": "Start time in seconds for the audio mask."}),
                "audio_end_time": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10000.0, "step": 0.01,
                                             "tooltip": "End time in seconds for the audio mask."}),
                "max_length": (["truncate", "pad", "partial"], {
                    "default": "truncate",
                    "tooltip": "'truncate': cut latent to end_time length. 'pad': extend latent to end_time. 'partial': mask range within existing latent."}),
            },
            "optional": {
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
                "existing_mask_mode": (["add", "subtract", "overwrite"], {
                    "default": "add",
                    "tooltip": "How to combine with existing noise masks if present. 'add' takes the max of existing and new, 'overwrite' replaces with the new mask, 'subtract' sets the masked region to 0, effectively unmasking it."}),
            },
        }

    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "execute"
    DESCRIPTION = "Creates noise masks for video and audio latents from time ranges. New content is generated inside the masked regions."

    def execute(self, video_fps, video_start_time, video_end_time, audio_start_time, audio_end_time,
                max_length="truncate", existing_mask_mode="add", video_latent=None, audio_latent=None):
        time_scale_factor = 8
        mel_hop_length = 160
        sampling_rate = 16000
        latent_downsample_factor = 4
        audio_latents_per_second = sampling_rate / mel_hop_length / latent_downsample_factor  # 25
        mask_value = 1.0 if existing_mask_mode != "subtract" else 0.0

        if video_latent is not None:
            samples = video_latent["samples"]
            video_latent_frame_count = samples.shape[2]

            video_pixel_frame_start_raw = int(round(video_start_time * video_fps))
            video_pixel_frame_end_raw = int(round(video_end_time * video_fps))
            required_latent_frames = (video_pixel_frame_end_raw - 1) // time_scale_factor + 1

            if max_length == "pad" and required_latent_frames > video_latent_frame_count:
                video_samples = torch.cat([samples, _zeros_like_tail(samples, required_latent_frames - video_latent_frame_count)], dim=2)
                video_latent_frame_count = video_samples.shape[2]
            elif max_length == "truncate":
                video_samples = samples[:, :, :required_latent_frames]
                video_latent_frame_count = video_samples.shape[2]
            else:  # partial
                video_samples = samples

            video_pixel_frame_count = (video_latent_frame_count - 1) * time_scale_factor + 1
            xp = np.array([0] + list(range(1, video_pixel_frame_count + time_scale_factor, time_scale_factor)))
            start_idx = np.searchsorted(xp, video_pixel_frame_start_raw, side="left")
            end_idx = np.searchsorted(xp, video_pixel_frame_end_raw, side="right") - 1
            start_idx = max(0, start_idx)
            end_idx = min(end_idx, video_latent_frame_count)

            padded = max_length == "pad" and video_samples.shape[2] > samples.shape[2]
            if "noise_mask" in video_latent and video_latent["noise_mask"] is not None and existing_mask_mode != "overwrite":
                video_mask = video_latent["noise_mask"].clone()
                if padded:
                    video_mask = torch.cat([video_mask, _zeros_like_tail(video_mask, video_samples.shape[2] - video_mask.shape[2])], dim=2)
                elif max_length == "truncate":
                    video_mask = video_mask[:, :, :video_samples.shape[2]]
            else:
                video_mask = torch.zeros_like(video_samples)[:, :1]

            video_mask[:, :, start_idx:end_idx] = mask_value
            if padded:  # every padded frame is masked too
                video_mask[:, :, samples.shape[2]:] = mask_value
            video_latent = video_latent.copy()
            video_latent["samples"] = video_samples
            video_latent["noise_mask"] = video_mask

        if audio_latent is not None:
            samples = audio_latent["samples"]
            audio_latent_frame_count = samples.shape[2]

            start_idx = int(round(audio_start_time * audio_latents_per_second))
            end_idx = int(round(audio_end_time * audio_latents_per_second)) + 1

            if max_length == "pad" and end_idx > audio_latent_frame_count:
                audio_samples = torch.cat([samples, _zeros_like_tail(samples, end_idx - audio_latent_frame_count)], dim=2)
                audio_latent_frame_count = audio_samples.shape[2]
            elif max_length == "truncate":
                audio_samples = samples[:, :, :end_idx]
                audio_latent_frame_count = audio_samples.shape[2]
            else:  # partial
                audio_samples = samples

            start_idx = max(0, start_idx)
            end_idx = min(end_idx, audio_latent_frame_count)

            padded = max_length == "pad" and audio_samples.shape[2] > samples.shape[2]
            if "noise_mask" in audio_latent and audio_latent["noise_mask"] is not None and existing_mask_mode != "overwrite":
                audio_mask = audio_latent["noise_mask"].clone()
                if padded:
                    audio_mask = torch.cat([audio_mask, _zeros_like_tail(audio_mask, audio_samples.shape[2] - audio_mask.shape[2])], dim=2)
                elif max_length == "truncate":
                    audio_mask = audio_mask[:, :, :audio_samples.shape[2]]
            else:
                audio_mask = torch.zeros_like(audio_samples)

            # the original always writes 1.0 here, even for "subtract" (only the video side unmasks)
            audio_mask[:, :, start_idx:end_idx] = 1.0
            if padded:
                audio_mask[:, :, samples.shape[2]:] = 1.0
            audio_latent = audio_latent.copy()
            audio_latent["samples"] = audio_samples
            audio_latent["noise_mask"] = audio_mask

        return (video_latent, audio_latent)


# ------------------------------------------------------------------ VAE loader

_AUDIO_VAE_MARKERS = (
    "vocoder.conv_post.weight",
    "vocoder.vocoder.conv_post.weight",
    "vocoder.resblocks.0.convs1.0.weight",
    "vocoder.vocoder.resblocks.0.convs1.0.weight",
)


class MpiVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "device": (["main_device", "cpu"],),
                "weight_dtype": (["bf16", "fp16", "fp32"],),
            },
        }

    CATEGORY = CATEGORY
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    DESCRIPTION = "Loads a VAE file on the main device or the CPU in bf16/fp16/fp32. An LTX audio VAE keeps its own dtype."

    def load_vae(self, vae_name, device, weight_dtype):
        from comfy.sd import VAE  # type: ignore
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[weight_dtype]
        device = mm.get_torch_device() if device == "main_device" else torch.device("cpu")

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd, metadata = load_torch_file(vae_path, return_metadata=True)

        if any(k in sd for k in _AUDIO_VAE_MARKERS):
            sd_audio = state_dict_prefix_replace(dict(sd), {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True)
            vae = VAE(sd=sd_audio, metadata=metadata)
        else:
            vae = VAE(sd=sd, device=device, dtype=dtype, metadata=metadata)
        vae.throw_exception_if_invalid()
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "MpiLTX2NAG": MpiLTX2NAG,
    "MpiLTXAudioVideoMask": MpiLTXAudioVideoMask,
    "MpiVAELoader": MpiVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiLTX2NAG": "Mpi LTX2 NAG",
    "MpiLTXAudioVideoMask": "Mpi LTX Audio Video Mask",
    "MpiVAELoader": "Mpi VAE Loader",
}
