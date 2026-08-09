"""Live sampling previews for video models core cannot preview.

ComfyUI's previewer picks a decoder from `latent_format.taesd_decoder_name`.
Several video formats — MiniMax H3's `MiniMaxH3Video`/`MiniMaxH3AV` among them —
name none, so every preview falls back to `Latent2RGB`: blocky colour blobs. The
weight exists (madebyollin's `taeh3`), core just has no way to reach it, and its
`TAESD` class could not load it anyway (that Decoder hardcodes width 64; taeh3 is
256 wide).

`MpiVideoSamplingPreview` does what `kjnodes`' LTX previewer does, without the
LTX-specific gates: it wraps OUTER_SAMPLE, decodes `x0` through a tiny TAEHV VAE,
and streams the frames as ordinary binary PREVIEW_IMAGE messages, paced in real
time so the preview plays at the clip's own speed instead of flickering once per
step.

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

import struct
import time
from io import BytesIO
from threading import Thread

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from PIL import Image

import comfy.model_management as mm  # type: ignore
import comfy.patcher_extension  # type: ignore
import server  # type: ignore

_PREVIEW_MAX = 512
_PREVIEW_MIN = 256


def _temporal_ratio(model_patcher, fallback: int = 1) -> int:
    """How many pixel frames the tiny decoder returns per latent frame.

    Read off the model's own latent format (`temporal_downscale_ratio`) rather
    than hardcoded: it is 4 for H3 and 8 for LTX, and getting it wrong only
    shows up as a frame index that wraps at the wrong point — no error, just a
    preview that stutters or repeats.
    """
    try:
        ratio = int(getattr(model_patcher.model.latent_format, "temporal_downscale_ratio", fallback))
        return max(1, ratio)
    except Exception:
        return fallback


class _TinyVaePreviewer:
    """Rate-limited frame cursor over the in-progress latent.

    One preview call does NOT mean one frame. The sampler calls back once per
    step, but a step is slow and a clip has many frames, so each call decodes
    however many frames real time has earned since the last one (`rate` fps) and
    walks a cursor around the clip. That is what makes the preview play at the
    video's own speed instead of jumping a frame per step.
    """

    def __init__(self, vae, rate: float, ratio: int):
        self.vae = vae
        self.rate = max(1.0, float(rate))
        self.ratio = ratio
        self.first = True
        self.last_time = 0.0
        self.cursor = 0

    def push(self, x0):
        srv = server.PromptServer.instance
        # H3 packs video+audio as a nested pair; the video half is tensors[0].
        # Core's own previewer does exactly this before decoding.
        if getattr(x0, "is_nested", False):
            x0 = x0.tensors[0]
        if x0.ndim == 5:
            # [B,C,T,H,W] -> frames as batch, so a slice of frames decodes as a batch.
            x0 = x0.movedim(2, 1).reshape((-1,) + tuple(x0.shape[-3:]))
        elif x0.ndim != 4:
            return

        num_latent_frames = x0.size(0)
        now = time.time()
        earned = int((now - self.last_time) * self.rate)
        if earned > num_latent_frames:
            earned = num_latent_frames
        elif earned <= 0:
            return
        self.last_time = self.last_time + earned / self.rate

        if self.first:
            self.first = False
            # Tells the consumer this is a clip: how many frames it will cycle
            # through and how fast. Sent once, before any frame.
            srv.send_sync(
                "VHS_latentpreview",
                {"length": num_latent_frames * self.ratio, "rate": self.rate, "id": srv.last_node_id},
            )
            self.last_time = now + 1.0 / self.rate

        if self.cursor + earned > num_latent_frames:
            batch = x0.roll(-self.cursor, 0)[:earned]
        else:
            batch = x0[self.cursor:self.cursor + earned]
        # .run() not .start(): decode on the calling thread so the GPU work stays
        # serialised with sampling. A second CUDA stream here fights the sampler
        # for VRAM mid-step, which is how a preview turns into an OOM.
        Thread(target=self._send, args=(batch, self.cursor * self.ratio, num_latent_frames)).run()
        self.cursor = (self.cursor + earned) % num_latent_frames

    def _decode(self, latent):
        model = self.vae.first_stage_model
        dtype = model.decoder[1].weight.dtype
        latent = latent.unsqueeze(0).to(dtype=dtype, device=mm.get_torch_device())
        # TAEHV returns [B,C,T,H,W]; drop batch and put channels last for PIL.
        return model.decode(latent)[0].permute(1, 2, 3, 0)

    def _send(self, latent, index, num_latent_frames):
        srv = server.PromptServer.instance
        images = self._decode(latent)

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
        span = max(1, (num_latent_frames - 1) * self.ratio + 1)
        for frame in frames:
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
            index = (index + 1) % span


class _PreviewWrapper:
    def __init__(self, vae, rate):
        self.vae = vae
        self.rate = rate

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask,
                 callback, disable_pbar, seed, latent_shapes=None):
        previewer = _TinyVaePreviewer(
            self.vae, self.rate, _temporal_ratio(executor.class_obj.model_patcher)
        )
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
                               "MiniMax H3. Load it with a plain VAELoader; ComfyUI detects "
                               "TAEHV from the state dict. Handing over the full VAE here "
                               "decodes at full cost every step and will crawl.",
                }),
                "preview_rate": ("FLOAT", {
                    "default": 8.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "tooltip": "Preview playback speed in frames per second. Frames are "
                               "decoded as real time earns them, so this is a cost dial as "
                               "much as a smoothness one.",
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
