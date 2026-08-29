"""Live sampling previews for video models core cannot preview.

ComfyUI's previewer picks a decoder from `latent_format.taesd_decoder_name`.
Several video formats — MiniMax H3's `MiniMaxH3Video`/`MiniMaxH3AV` among them —
name none, so every preview falls back to `Latent2RGB`: blocky colour blobs. The
weight exists (madebyollin's `taeh3`), core just has no way to reach it, and its
`TAESD` class could not load it anyway (that Decoder hardcodes width 64; taeh3 is
256 wide).

`MpiVideoSamplingPreview` does what `kjnodes`' LTX previewer does, without the
LTX-specific gates: it wraps OUTER_SAMPLE, decodes `x0` through a tiny TAEHV VAE,
and streams the frames as ordinary binary PREVIEW_IMAGE messages — a whole clip
per sampler step, so the consumer can play the motion back at the clip's own
speed instead of flickering one still per step.

WHY THAT TRANSPORT AND NOT A NODE WIDGET: a host app driving ComfyUI over the
websocket already consumes binary previews. `ModelPreviewOverrideKJ` base64s its
frames onto a private `kj_preview_override` event bound to a DOM widget on the
node, which is invisible to anything but the ComfyUI web UI. Same picture, wrong
road. The `VHS_latentpreview` marker sent once at the start is what tells a
consumer these frames are a CLIP to accumulate rather than stills to replace.

The tiny VAE arrives as a normal VAE input: `comfy/sd.py` recognises a TAEHV from
`decoder.22.bias`, so a plain VAELoader on `taeh3.safetensors` produces one.
Nothing here is H3-specific — any video model with a TAEHV-shaped tiny decoder
works — but H3 is what it was built and tested for.
"""

import math
import struct
from io import BytesIO

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from PIL import Image

import comfy.model_management as mm  # type: ignore
import comfy.patcher_extension  # type: ignore
import comfy.sd  # type: ignore
import comfy.taesd.taehv as taehv  # type: ignore
import comfy.utils  # type: ignore
import folder_paths  # type: ignore
import server  # type: ignore

_PREVIEW_MAX = 512
_PREVIEW_MIN = 256


def _decode_clip(model, latent):
    """[B,C,T,H,W] latent -> [B,3,T,H,W] pixels, in the decoder's own chunking.

    A TAEHV is TEMPORAL: its MemBlocks chain state forward, so frame N is only
    correct if every frame before it was decoded in the same pass. Always decode
    from frame 0. Sampling a window out of the middle with cold state is not a
    cheaper approximation, it is garbage — that is what green previews were.

    H3 additionally codes 17 pixel frames per 5 latent tokens, and `TAEHV.decode`
    trims `frames_to_trim` ONCE globally, which leaves every later chunk carrying
    its own 3 pad frames. Trim per chunk instead and drop the encoder's 3-token
    tail pad, as kjnodes' `TAEHVDecoder._decode_h3_full` does. Any other tiny
    decoder core sizes correctly (taeltx2_3, taehv, taew) takes the plain path.
    """
    if getattr(model, "latent_channels", None) != 24 or getattr(model, "patch_size", 1) != 2:
        return model.decode(latent)

    x = model.process_in(latent).movedim(2, 1)  # [B,C,T,H,W] -> [B,T,C,H,W]
    x = taehv.apply_model_with_memblocks(
        model.decoder, x, model.parallel, False,
        output_device=mm.intermediate_device(), patch_size=model.patch_size, decode=True,
    )
    chunk = 5 * model.t_upscale
    x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, -x.shape[1] % chunk))
    x = x.unflatten(1, (-1, chunk))[:, :, model.frames_to_trim:].flatten(1, 2)
    x = x[:, :-3 * model.t_upscale]
    return x.movedim(2, 1) if x.shape[1] else model.decode(latent)


class _TinyVaePreviewer:
    """Decodes the in-progress latent as a WHOLE CLIP, once per sampler step.

    One preview call does NOT mean one frame: the sampler calls back once per
    step and each call bursts the entire clip, which the `VHS_latentpreview`
    marker tells the consumer to accumulate and loop at `rate` fps. Decoding the
    whole thing is not a cost choice either — see `_decode_clip`, a temporal
    decoder cannot produce frame N without the frames before it.
    """

    def __init__(self, vae, rate: float, latent_shapes=None, num_keyframes=0):
        self.vae = vae
        self.rate = max(1.0, float(rate))
        self.latent_shapes = latent_shapes
        self.num_keyframes = num_keyframes
        self.first = True

    def push(self, x0):
        srv = server.PromptServer.instance
        # H3 packs video+audio as a nested pair; the video half comes first.
        #
        # WHICH VIEW ARRIVES HERE IS NOT A CHOICE. A multi-part latent reaches the
        # sampler as ONE flat pack, and core restores the nested view in a callback
        # wrapper it builds BEFORE calling outer_sample — so our OUTER_SAMPLE wrapper
        # installs its callback INSIDE that unpacker and sees the flat pack, while
        # core's own previewer, further out, sees the nested one. Measured on H3:
        # x0 arrives as (1, 1, 658752) with is_nested False, against latent_shapes
        # [[1,24,17,40,40], [1,32,2,93]] — video + audio, and those two do sum to
        # 658752. Unpack it ourselves; `latent_shapes` is handed to the wrapper for
        # exactly this. Handle the nested view too, in case a path delivers it.
        if getattr(x0, "is_nested", False):
            x0 = x0.tensors[0]
        elif self.latent_shapes and len(self.latent_shapes) > 1:
            x0 = comfy.utils.unpack_latents(x0, self.latent_shapes)[0]
        if x0.ndim == 4:
            x0 = x0.unsqueeze(2)  # a still latent [B,C,H,W] decodes as a 1-frame clip
        if x0.ndim != 5:
            return

        # Guide latents ride at the END of the video stream. LTXVAddGuide appends
        # one per conditioning image and records their positions in the positive
        # cond's `keyframe_idxs`; they are model input, not clip, so previewing
        # them shows the user their own input frames pasted onto the tail. H3
        # never had any, which is why this node did not need it until LTX.
        if 0 < self.num_keyframes < x0.shape[2]:
            x0 = x0[:, :, :-self.num_keyframes]

        # ponytail: whole clip, every step. 6-step H3 makes that ~6 tiny-VAE
        # decodes a run; a long sampler on a long clip would want a wall-clock
        # throttle here.
        images = self._decode(x0[:1])
        if images.size(0) == 0:
            return

        if self.first:
            self.first = False
            # Tells the consumer this is a clip: how many frames it will cycle
            # through and how fast. Sent once, before any frame.
            srv.send_sync(
                "VHS_latentpreview",
                {"length": images.size(0), "rate": self.rate, "id": srv.last_node_id},
            )
        self._send(images)

    def _decode(self, latent):
        """[1,C,T,H,W] -> [T,H,W,3].

        Runs on the calling thread so the GPU work stays serialised with
        sampling. A second CUDA stream here fights the sampler for VRAM
        mid-step, which is how a preview turns into an OOM.
        """
        model = self.vae.first_stage_model
        dtype = model.decoder[1].weight.dtype
        latent = latent.to(dtype=dtype, device=mm.get_torch_device())
        # Decode returns [B,3,T,H,W]; drop batch and put channels last for PIL.
        return _decode_clip(model, latent)[0].permute(1, 2, 3, 0)

    def _send(self, images):
        srv = server.PromptServer.instance

        if images.size(1) < _PREVIEW_MIN or images.size(2) < _PREVIEW_MIN:
            images = F.interpolate(images.movedim(-1, 0), scale_factor=4, mode="nearest").movedim(0, -1)
        if images.size(1) > _PREVIEW_MAX or images.size(2) > _PREVIEW_MAX:
            chw = images.movedim(-1, 0)
            if chw.size(2) < chw.size(3):
                height = (_PREVIEW_MAX * chw.size(2)) // chw.size(3)
                chw = F.interpolate(chw, (height, _PREVIEW_MAX), mode="nearest")
            else:
                width = (_PREVIEW_MAX * chw.size(3)) // chw.size(2)
                chw = F.interpolate(chw, (_PREVIEW_MAX, width), mode="nearest")
            images = chw.movedim(0, -1)

        frames = (images.clamp(0, 1).mul(0xFF)).to(device="cpu", dtype=torch.uint8)
        for index, frame in enumerate(frames):
            message = BytesIO()
            # Same envelope kjnodes' LTX previewer uses, which is what the
            # VHS-style consumer expects: event type 1 twice, frame index, then
            # the node id as a 16-byte pascal string.
            message.write((1).to_bytes(length=4, byteorder="big") * 2)
            message.write(index.to_bytes(length=4, byteorder="big"))
            message.write(struct.pack("16p", (srv.last_node_id or "").encode("ascii")))
            Image.fromarray(frame.numpy()).save(message, format="JPEG", quality=95, compress_level=1)
            srv.send_sync(
                server.BinaryEventTypes.PREVIEW_IMAGE, message.getvalue(), srv.client_id
            )


def _count_keyframes(executor):
    """How many guide latents the positive cond appended, 0 if none.

    Same read kjnodes' LTX previewer does. `keyframe_idxs` is [B,1,N,3] and one
    guide image can occupy several rows, so the count is the unique frame index.
    """
    try:
        positive = executor.class_obj.conds["positive"]
        idxs = positive[0].get("keyframe_idxs") if positive else None
        return 0 if idxs is None else len(torch.unique(idxs[0, 0, :, 0]))
    except Exception:
        return 0


class _PreviewWrapper:
    def __init__(self, vae, rate):
        self.vae = vae
        self.rate = rate

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask,
                 callback, disable_pbar, seed, latent_shapes=None):
        previewer = _TinyVaePreviewer(self.vae, self.rate, latent_shapes,
                                      _count_keyframes(executor))
        # Pin the tiny VAE to the sampling device: it is ~22MB and reloading it
        # per step costs more than the decode.
        try:
            self.vae.first_stage_model.to(mm.get_torch_device())
        except Exception:
            pass

        original = callback

        def combined(step, x0, x, total_steps):
            try:
                previewer.push(x0)
            except Exception as exc:
                # A preview must never take the generation down with it. Report
                # once and let the run finish with whatever core was showing.
                if not getattr(previewer, "_failed", False):
                    previewer._failed = True
                    print(f"[MpiVideoSamplingPreview] preview disabled after: {exc}")
            if original is not None:
                original(step, x0, x, total_steps)

        if latent_shapes is None:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            combined, disable_pbar, seed)
        return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                        combined, disable_pbar, seed, latent_shapes=latent_shapes)


class MpiVideoSamplingPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE", {
                    "tooltip": "A TINY TAEHV decoder, not the model's real VAE — taeh3 for "
                               "MiniMax H3. Load it with MpiTinyVaeLoader: a plain VAELoader "
                               "CANNOT build taeh3 (see that node). Handing over the full VAE "
                               "here decodes at full cost every step and will crawl.",
                }),
                "preview_rate": ("FLOAT", {
                    "default": 8.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "tooltip": "Playback speed in frames per second, announced to the "
                               "consumer with the clip. Every step decodes the whole clip "
                               "(a temporal decoder cannot skip frames), so this paces "
                               "playback, not cost.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "MpiNodes/Sampling"
    DESCRIPTION = (
        "Live RGB sampling previews for video models ComfyUI cannot preview. Formats like "
        "MiniMax H3's name no taesd decoder, so their previews fall back to blocky "
        "latent2rgb blobs. Wire a tiny TAEHV VAE (taeh3) into `vae` and this decodes the "
        "in-progress latent for real, streaming frames on the standard binary preview "
        "channel so a host app sees them with no extra plumbing. Sits anywhere on the "
        "model wire before the sampler; passes the model through untouched."
    )
    FUNCTION = "doit"

    def doit(self, model, vae, preview_rate):
        patched = model.clone()
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "mpi_video_sampling_preview",
            _PreviewWrapper(vae, preview_rate),
        )
        return (patched,)


# ── Loading a tiny decoder core cannot build ────────────────────────────────
#
# `TAEHV.__init__` sizes its edge convs as `image_channels * patch_size**2` and
# picks `patch_size = 2` for `latent_channels in [48, 32]` only. taeh3 is a
# 24-channel latent with a 12-wide decoder (3 RGB x 4 temporal frames), so core
# builds it patch_size 1 / 3 wide and `VAELoader` dies inside `comfy.sd.VAE`:
#
#     size mismatch for decoder.22.bias: checkpoint [12] vs model [3]
#
# There is no branch in `comfy/sd.py` for that shape and no argument that
# reaches it — patch_size is derived from latent_channels alone. Verified the
# same on engine 0.30.0 and bench 0.30.2, whose `taehv.py` are byte-identical,
# so this is a missing case rather than a version we can wait out.
_PATCHED_LATENT_CHANNELS = (24,)


class _TAEHV(taehv.TAEHV):
    """TAEHV that honours a 12-wide decoder on a 24-channel latent.

    Only the two edge convs differ from what core builds; every other layer,
    width and activation already matches, which a strict state-dict load
    proves (0 missing, 0 unexpected across all 128 tensors).
    """

    def __init__(self, latent_channels, **kwargs):
        super().__init__(latent_channels, **kwargs)
        if latent_channels in _PATCHED_LATENT_CHANNELS:
            self.patch_size = 2
            self.encoder[0] = taehv.conv(3 * self.patch_size ** 2, 64)
            self.decoder[-1] = taehv.conv(64, 3 * self.patch_size ** 2)


def _load_tiny_vae(path):
    """Build a VAE for a tiny decoder, correcting core only where it is wrong."""
    sd = comfy.utils.load_torch_file(path)
    weight = sd.get("decoder.1.weight")
    channels = weight.shape[1] if weight is not None else None
    if channels not in _PATCHED_LATENT_CHANNELS:
        # Every shape core already knows (taeltx2_3, taehv, taew, lighttae).
        return comfy.sd.VAE(sd=sd)

    # `comfy.sd.VAE` does all the field setup we want; it just constructs the
    # inner model wrong. Swap the class for the duration of the call rather
    # than duplicating that setup, which drifts silently on a ComfyUI bump.
    original = taehv.TAEHV
    taehv.TAEHV = _TAEHV
    try:
        vae = comfy.sd.VAE(sd=sd)
    finally:
        taehv.TAEHV = original

    # Core reached this weight through its fallback branch, which assumes an 8x
    # spatial tiny VAE and applies HunyuanVideo latent scaling to anything fp16.
    # Both are wrong here: taeh3 is 16x spatially (a 30x54 latent decodes to
    # 480x864) and needs no scaling, the same call core makes for Wan 2.2/LTX2.
    vae.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
    vae.upscale_index_formula = (4, 16, 16)
    vae.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
    vae.downscale_index_formula = (4, 16, 16)
    vae.first_stage_model.process_in = lambda x: x
    vae.first_stage_model.process_out = lambda x: x
    return vae


class MpiTinyVaeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "A tiny TAEHV decoder in models/vae — taeh3 for MiniMax H3. "
                               "Pass the result to MpiVideoSamplingPreview.",
                }),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    CATEGORY = "MpiNodes/Sampling"
    DESCRIPTION = (
        "Loads a tiny TAEHV preview decoder that ComfyUI's own VAELoader cannot build. "
        "Core picks the decoder width from the latent channel count and has no case for "
        "a 24-channel latent with a 12-wide decoder, which is what MiniMax H3's taeh3 is "
        "— VAELoader raises a state-dict size mismatch on it. Anything core already "
        "handles is passed straight through to VAELoader's own code path, so this is a "
        "safe drop-in for any tiny decoder."
    )
    FUNCTION = "doit"

    def doit(self, vae_name):
        return (_load_tiny_vae(folder_paths.get_full_path_or_raise("vae", vae_name)),)
