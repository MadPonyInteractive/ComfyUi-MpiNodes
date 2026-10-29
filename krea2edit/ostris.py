# Ported from ComfyUI-Krea2-Ostris-Edit (nodes.py, pin 7756566), MIT, Copyright (c) 2026 Ostris, LLC.
# See LICENSE-Krea2-Ostris-Edit. Cubric Studio MPI-1343: class names TextEncodeKrea2OstrisEdit ->
# MpiKrea2OstrisEditTextEncode, Krea2OstrisEditModelPatch -> MpiKrea2OstrisEditModelPatch. Every input
# and the patch are unchanged.
"""Krea 2 reference-image (edit) conditioning for ComfyUI.

Adds Kontext-style multi-reference support to Krea 2 without touching core:

  - ``MpiKrea2OstrisEditTextEncode`` encodes the prompt + reference images through the
    Krea 2 Qwen3-VL text encoder (edit-plus style ``Picture N:`` vision
    placeholders inside Krea's own conditioning template) and attaches the
    VAE reference latents to the conditioning.
  - ``MpiKrea2OstrisEditModelPatch`` patches the Krea 2 model (via ModelPatcher object
    patches, applied/removed per-workflow) so those reference latents are
    appended to the image token sequence with RoPE axis-0 index 1, 2, 3... and
    conditioned at t=0 -- the ComfyUI Flux/QwenImage "index_timestep_zero"
    reference method.

Both mirror the ai-toolkit ``krea2`` training implementation exactly:
VL images are downscaled (never upscaled) to fit 384x384 total pixels,
reference latents to fit 1MP, snapped to /16 so the latent grid patchifies.

``MpiKrea2OstrisEditModelPatch``'s ``kv_cache`` toggle (default off) is for LoRAs
trained with ai-toolkit's ``kv_cache`` model kwarg, where reference tokens
attend only to each other. Their per-block K/V are then timestep-invariant, so
they are precomputed in a single ref-only pass at t=0 and injected as extra
attention keys on every denoising step -- the ref tokens never ride along in
the per-step sequence (OminiControl2-style conditioning feature reuse). The
LoRA must be trained with ``kv_cache`` enabled for this to work properly.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.conds
import comfy.ldm.common_dit
import comfy.utils
import node_helpers
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.text_encoders.krea2 import KREA2_TEMPLATE

# ---------------------------------------------------------------------------
# Text encode (VL images -> conditioning, refs -> reference_latents)
# ---------------------------------------------------------------------------

VLM_MAX_PIXELS = 384 * 384  # Qwen3-VL sees a coarse reference
REF_LATENT_MAX_PIXELS = 1024 * 1024  # VAE ref latents carry the detail
REF_SNAP = 16  # VAE f8 * patch 2 -> latent grid patchifies


def _fit_area(samples, max_pixels, snap=1):
    """Downscale (never upscale) a (B, C, H, W) image to fit ``max_pixels``,
    keeping aspect, snapping dims to ``snap``."""
    h, w = samples.shape[2], samples.shape[3]
    scale = min(1.0, math.sqrt(max_pixels / (w * h)))
    nw = max(round(w * scale / snap) * snap, snap)
    nh = max(round(h * scale / snap) * snap, snap)
    if (nh, nw) == (h, w):
        return samples
    return comfy.utils.common_upscale(samples, nw, nh, "area", "disabled")


class MpiKrea2OstrisEditTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "ostris/krea2"
    DESCRIPTION = (
        "Encode a prompt with optional reference images for a Krea 2 edit "
        "LoRA. Images are fed to the Qwen3-VL text encoder (needs a text "
        "encoder checkpoint that includes the vision weights) and, when a VAE "
        "is connected, attached as reference latents for MpiKrea2OstrisEditModelPatch."
    )

    def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None):
        images_vl = []
        ref_latents = []
        image_prompt = ""

        for image in (image1, image2, image3):
            if image is None:
                continue
            samples = image.movedim(-1, 1)
            images_vl.append(_fit_area(samples, VLM_MAX_PIXELS).movedim(1, -1))
            if vae is not None:
                s = _fit_area(samples, REF_LATENT_MAX_PIXELS, snap=REF_SNAP)
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))
            image_prompt += (
                "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(
                    len(images_vl)
                )
            )

        # Krea's own template (not the Qwen-Edit one): the trained LoRA saw the
        # base T2I system prompt with the Picture markers in the user section.
        tokens = clip.tokenize(
            image_prompt + prompt, images=images_vl, llama_template=KREA2_TEMPLATE
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )
        return (conditioning,)


# ---------------------------------------------------------------------------
# Model patch (reference latents -> t=0 tokens in the sequence)
# ---------------------------------------------------------------------------


def _pack_refs(dit, ref_latents, bs, device, dtype):
    """Patchify reference latents into tokens + RoPE positions: each reference
    gets axis-0 index i+1 with its own y/x grid from 0. Returns
    ``(reftok (B, Lr, C*p*p), refpos (B, Lr, 3))``."""
    patch = dit.patch
    ref_tokens = []
    ref_pos = []
    for i, ref in enumerate(ref_latents):
        if ref.ndim == 5:  # (B, C, T, H, W) Wan21 layout, T == 1 for images
            rb, rc, rt, rh5, rw5 = ref.shape
            ref = ref.reshape(rb * rt, rc, rh5, rw5)
        ref = comfy.ldm.common_dit.pad_to_patch_size(
            ref.to(device, dtype), (patch, patch)
        )
        ref = comfy.utils.repeat_to_batch_size(ref, bs)
        rh, rw = ref.shape[-2] // patch, ref.shape[-1] // patch
        ref_tokens.append(
            rearrange(ref, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        )
        rid = torch.zeros(rh, rw, 3, device=device, dtype=torch.float32)
        rid[..., 0] = i + 1.0
        rid[..., 1] = torch.arange(rh, device=device, dtype=torch.float32)[:, None]
        rid[..., 2] = torch.arange(rw, device=device, dtype=torch.float32)[None, :]
        ref_pos.append(rid.reshape(1, rh * rw, 3).repeat(bs, 1, 1))
    return torch.cat(ref_tokens, dim=1), torch.cat(ref_pos, dim=1)


def _attn_kv(attn, x, freqs, mask=None, kv_capture=None, kv_cache=None,
             transformer_options={}):
    """comfy krea2 ``Attention.forward`` with optional K/V capture or cached-K/V
    injection. Both happen post-RoPE and pre-GQA-expansion."""
    q, k, v, gate = attn.wq(x), attn.wk(x), attn.wv(x), attn.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=attn.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=attn.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=attn.kvheads)
    q, k = attn.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if kv_capture is not None:
        kv_capture.append((k, v))
    if kv_cache is not None:
        # Cached ref K/V are already RoPE'd at their original positions.
        k = torch.cat((k, kv_cache[0].to(k.dtype)), dim=2)
        v = torch.cat((v, kv_cache[1].to(v.dtype)), dim=2)
    if attn.kvheads != attn.heads:
        rep = attn.heads // attn.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = optimized_attention_masked(q, k, v, attn.heads, mask=mask,
                                     skip_reshape=True,
                                     transformer_options=transformer_options)
    return attn.wo(out * F.sigmoid(gate))


def _block_kv_forward(block, x, vec, freqs, kv_capture=None, kv_cache=None,
                      transformer_options={}):
    """SingleStreamBlock forward (single-span modulation) with K/V capture /
    injection in the attention."""
    prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)
    x = x + pregate * _attn_kv(
        block.attn, (1 + prescale) * block.prenorm(x) + preshift, freqs,
        kv_capture=kv_capture, kv_cache=kv_cache,
        transformer_options=transformer_options,
    )
    x = x + postgate * block.mlp((1 + postscale) * block.postnorm(x) + postshift)
    return x


def _block_ref_forward(block, x, vec, refvec, split, freqs, transformer_options):
    """SingleStreamBlock forward with per-span modulation: tokens ``[:split]``
    (text + noisy image) use ``vec`` (real t), tokens ``[split:]`` (clean
    reference tokens) use ``refvec`` (t=0)."""
    m = block.mod(vec)
    r = block.mod(refvec)

    def mod(h, scale, shift):
        return torch.cat(
            (
                (1 + m[scale]) * h[:, :split] + m[shift],
                (1 + r[scale]) * h[:, split:] + r[shift],
            ),
            dim=1,
        )

    def gate(h, g):
        return torch.cat((m[g] * h[:, :split], r[g] * h[:, split:]), dim=1)

    x = x + gate(
        block.attn(
            mod(block.prenorm(x), 0, 1),
            freqs,
            None,
            transformer_options=transformer_options,
        ),
        2,
    )
    x = x + gate(block.mlp(mod(block.postnorm(x), 3, 4)), 5)
    return x


def _forward_with_refs(self, x, timesteps, context, ref_latents, transformer_options):
    """Krea 2 SingleStreamDiT forward with reference latents appended at t=0
    ("index_timestep_zero"). Mirrors comfy.ldm.krea2.model.SingleStreamDiT._forward
    (including the 5D Wan21-latent handling) with the ref packing / modulation
    additions."""
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, c, H_orig, W_orig = x.shape
    patch = self.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch
    device = x.device

    context = self._unpack_context(context)

    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    # Pack each reference: RoPE axis-0 index i+1, own y/x grid from 0.
    reftok, refpos = _pack_refs(self, ref_latents, bs, device, x.dtype)
    reflen = reftok.shape[1]

    img = self.first(torch.cat((img, reftok), dim=1))

    t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec = self.tproj(t)
    t0 = self.tmlp(
        timestep_embedding(torch.zeros_like(timesteps), self.tdim)
        .unsqueeze(1)
        .to(img.dtype)
    )
    tvec0 = self.tproj(t0)

    context = self.txtfusion(
        context, mask=None, transformer_options=transformer_options
    )
    context = self.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]
    combined = torch.cat((context, img), dim=1)
    split = txtlen + imglen - reflen  # ref span start

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    pos = torch.cat((txtpos, imgpos, refpos), dim=1)

    freqs = self.pe_embedder(pos)

    for block in self.blocks:
        combined = _block_ref_forward(
            block, combined, tvec, tvec0, split, freqs, transformer_options
        )

    final = self.last(combined, t)
    out = final[:, txtlen:split, :]  # noisy target tokens only
    out = rearrange(
        out,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h_,
        w=w_,
        ph=patch,
        pw=patch,
        c=self.channels,
    )
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)
    return out


# ---------------------------------------------------------------------------
# Isolated ref attention: one-pass ref K/V precompute + cached denoising
# ---------------------------------------------------------------------------


def _precompute_ref_kv(dit, x, timesteps, ref_latents, transformer_options):
    """Run only the clean reference tokens through the blocks at t=0 and record
    each block's post-RoPE K/V.

    An isolate-ref-attention LoRA never lets ref tokens see the text / noisy
    tokens, so this standalone pass produces exactly the K/V the refs would
    contribute inside a joint pass -- and they are timestep-invariant, so one
    pass serves the whole denoise."""
    bs = x.shape[0] * (x.shape[2] if x.ndim == 5 else 1)
    reftok, refpos = _pack_refs(dit, ref_latents, bs, x.device, x.dtype)
    h = dit.first(reftok)
    t0 = dit.tmlp(
        timestep_embedding(torch.zeros_like(timesteps), dit.tdim)
        .unsqueeze(1)
        .to(h.dtype)
    )
    tvec0 = dit.tproj(t0)
    freqs = dit.pe_embedder(refpos)

    ref_kv = []
    for block in dit.blocks:
        cap = []
        h = _block_kv_forward(
            block, h, tvec0, freqs, kv_capture=cap,
            transformer_options=transformer_options,
        )
        ref_kv.append(cap[0])
    return ref_kv


def _forward_with_cached_refs(self, x, timesteps, context, ref_kv, transformer_options):
    """Denoising pass with the reference tokens replaced by their cached K/V:
    the live sequence is just text + noisy image tokens, and every block's
    attention appends the cached ref K/V as extra keys. Matches the joint pass
    of an isolate-ref-attention LoRA without recomputing the refs each step."""
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, c, H_orig, W_orig = x.shape
    patch = self.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch
    device = x.device

    context = self._unpack_context(context)

    img = self.first(
        rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    )

    t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec = self.tproj(t)

    context = self.txtfusion(
        context, mask=None, transformer_options=transformer_options
    )
    context = self.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]
    combined = torch.cat((context, img), dim=1)

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    pos = torch.cat((txtpos, imgpos), dim=1)

    freqs = self.pe_embedder(pos)

    for block, kv in zip(self.blocks, ref_kv):
        combined = _block_kv_forward(
            block, combined, tvec, freqs, kv_cache=kv,
            transformer_options=transformer_options,
        )

    final = self.last(combined, t)
    out = final[:, txtlen : txtlen + imglen, :]
    out = rearrange(
        out,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h_,
        w=w_,
        ph=patch,
        pw=patch,
        c=self.channels,
    )
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)
    return out


def _ref_fingerprint(ref_latents, bs):
    """Cheap content key for the ref K/V cache: batch size + per-ref shape and
    two reduction sums (the ref tensors are rebuilt per step by comfy's cond
    batching, so object identity can't be used)."""
    key = [bs]
    for r in ref_latents:
        rf = r.float()
        key.append((tuple(r.shape), float(rf.sum()), float(rf.square().sum())))
    return tuple(key)


class MpiKrea2OstrisEditModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "kv_cache": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Cache the reference tokens' attention K/V: they "
                            "are precomputed in one t=0 pass and reused every "
                            "denoising step, so the refs never ride along in "
                            "the per-step sequence. The LoRA must be trained "
                            "with ai-toolkit's kv_cache model kwarg for this "
                            "to work properly. Leave off for normal edit "
                            "LoRAs."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "ostris/krea2"
    DESCRIPTION = (
        "Enable reference latents on a Krea 2 model (index_timestep_zero "
        "method, as trained by ai-toolkit). Chain conditioning from "
        "MpiKrea2OstrisEditTextEncode or Set Reference Latent nodes. "
        "kv_cache enables the cached one-pass reference mode; the LoRA must "
        "be trained with ai-toolkit's kv_cache option for it to work properly."
    )

    def patch(self, model, kv_cache=False):
        m = model.clone()
        base_model = m.model
        dit = m.get_model_object("diffusion_model")

        orig_extra_conds = base_model.extra_conds
        orig_extra_conds_shapes = base_model.extra_conds_shapes
        orig_forward = dit.forward

        def extra_conds(**kwargs):
            out = orig_extra_conds(**kwargs)
            ref_latents = kwargs.get("reference_latents", None)
            if ref_latents is not None:
                out["ref_latents"] = comfy.conds.CONDList(
                    [base_model.process_latent_in(lat) for lat in ref_latents]
                )
            return out

        def extra_conds_shapes(**kwargs):
            out = orig_extra_conds_shapes(**kwargs)
            ref_latents = kwargs.get("reference_latents", None)
            if ref_latents is not None:
                out["ref_latents"] = list(
                    [1, 16, sum(map(lambda a: math.prod(a.size()), ref_latents)) // 16]
                )
            return out

        # Ref K/V cache for the kv_cache toggle: filled by a single ref-only
        # pass per run (t=0, refs attend only to each other, matching training)
        # and reused by every denoising call. Cleared at the start of each
        # sampling run so LoRA/weight changes between runs never serve stale
        # K/V; keyed on ref content + batch size so cond/uncond with different
        # refs coexist.
        state = {"last_sigma": None, "caches": {}}

        def forward(
            x,
            timesteps,
            context,
            attention_mask=None,
            transformer_options={},
            ref_latents=None,
            **kwargs,
        ):
            if ref_latents is None or len(ref_latents) == 0:
                return orig_forward(
                    x,
                    timesteps,
                    context,
                    attention_mask=attention_mask,
                    transformer_options=transformer_options,
                    **kwargs,
                )
            if not kv_cache:
                return _forward_with_refs(
                    dit, x, timesteps, context, ref_latents, transformer_options
                )

            # New-run detection: sigmas only decrease within a run, and the
            # first call of a run sits at sample_sigmas[0] when available.
            sig = float(timesteps.max())
            sample_sigmas = transformer_options.get("sample_sigmas", None)
            new_run = state["last_sigma"] is None or sig > state["last_sigma"]
            if (
                sample_sigmas is not None
                and sig == float(sample_sigmas[0])
                and sig != state["last_sigma"]
            ):
                new_run = True
            if new_run:
                state["caches"].clear()
            state["last_sigma"] = sig

            bs = x.shape[0] * (x.shape[2] if x.ndim == 5 else 1)
            key = _ref_fingerprint(ref_latents, bs)
            ref_kv = state["caches"].get(key)
            if ref_kv is None:
                ref_kv = _precompute_ref_kv(
                    dit, x, timesteps, ref_latents, transformer_options
                )
                state["caches"][key] = ref_kv
            return _forward_with_cached_refs(
                dit, x, timesteps, context, ref_kv, transformer_options
            )

        m.add_object_patch("extra_conds", extra_conds)
        m.add_object_patch("extra_conds_shapes", extra_conds_shapes)
        m.add_object_patch("diffusion_model.forward", forward)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "MpiKrea2OstrisEditTextEncode": MpiKrea2OstrisEditTextEncode,
    "MpiKrea2OstrisEditModelPatch": MpiKrea2OstrisEditModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiKrea2OstrisEditTextEncode": "Mpi Text Encode Krea 2 Ostris Edit",
    "MpiKrea2OstrisEditModelPatch": "Mpi Krea 2 Ostris Edit Model Patch",
}
