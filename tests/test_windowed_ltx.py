"""CPU check for MpiWindowedSampler on an LTXV latent. Real ComfyUI core, a fake model. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_windowed_ltx.py

LTXV differs from MiniMax H3 in three ways the node has to know (Cubric Studio MPI-1450):
  - its causal VAE gives the clip's first latent frame ONE video frame and every later one 8: T -> 8(T-1)+1;
  - its audio latent is (batch, channels, TIME, frequency): time is dim 2, not the last dim;
  - an i2v latent carries a NESTED noise mask (video + audio), so the audio half must be cut with the audio.
And LTXVAddGuide / IC-LoRA guides append frames to the latent, which a time window cannot slice.

A frame-local model cannot tell a window from the whole clip, so the windowed video must equal one pass.
"""
import os
import sys
import types

import torch

COMFY = os.environ.get("COMFY_ROOT", "G:/ComfyUi/ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = sys.argv[:1] + ["--cpu"]

import comfy.latent_formats  # noqa: E402
import comfy.model_sampling  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.nested_tensor import NestedTensor  # noqa: E402

import sampler  # noqa: E402

sampler.comfy.sample.fix_empty_latent_channels = lambda patcher, samples, *a: samples
sampler.latent_preview.prepare_callback = lambda *a: (lambda *b: None)

FREQ = 16  # LTXV audio latent frequency bins


class _Sampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    pass


class _Wrap:
    """model_wrap: denoised = a frame-local nonlinear function of x and sigma."""
    def __init__(self):
        # LTXV's own scale_latent_inpaint (comfy/model_base.py): a masked frame is held at the latent itself
        self.inner_model = types.SimpleNamespace(model_sampling=_Sampling(),
                                                 scale_latent_inpaint=lambda latent_image, **k: latent_image)
        self.model_patcher = types.SimpleNamespace(
            model=self, get_model_object=lambda name: self.inner_model.model_sampling)
        self.cfg = 1.0

    def __call__(self, x, sigma, model_options={}, seed=None):
        s = sigma.view([-1] + [1] * (x.ndim - 1))
        return 0.6 * x / (1 + s) + 0.1 * torch.sin(3 * x)


class _Guider:
    """Mirrors CFGGuider.sample's packing around sampler.sample, and records what each pass was handed."""
    def __init__(self, conds=None):
        self.model_patcher = types.SimpleNamespace(
            model=types.SimpleNamespace(latent_format=comfy.latent_formats.LTXAV(),
                                        process_latent_out=lambda x: x),
            load_device="cpu")
        self.original_conds = conds or {"positive": [{"cross_attn": None}]}
        self.seen = []

    def sample(self, noise, latent, smp, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        parts = latent.unbind() if getattr(latent, "is_nested", False) else [latent]
        masks = denoise_mask.unbind() if getattr(denoise_mask, "is_nested", False) else [denoise_mask]
        self.seen.append(([tuple(p.shape) for p in parts], [None if m is None else tuple(m.shape) for m in masks]))
        for p, m in zip(parts, masks):  # a mask longer than its window is exactly the bug
            assert m is None or m.shape[2] == p.shape[2], (tuple(m.shape), tuple(p.shape))
        shapes = None
        if len(parts) > 1:
            latent, shapes = comfy.utils.pack_latents(parts)
            noise, _ = comfy.utils.pack_latents(noise.unbind())
            if denoise_mask is not None:
                denoise_mask, _ = comfy.utils.pack_latents(denoise_mask.unbind())
        out = smp.sample(_Wrap(), sigmas, {}, None, noise, latent, denoise_mask, disable_pbar)
        return NestedTensor(comfy.utils.unpack_latents(out, shapes)) if shapes else out


class _Noise:
    seed = 7

    def generate_noise(self, latent):
        g = torch.Generator().manual_seed(self.seed)
        s = latent["samples"]
        if getattr(s, "is_nested", False):
            return NestedTensor([torch.randn(t.shape, generator=g) for t in s.unbind()])
        return torch.randn(s.shape, generator=g)


def _latent(t=16, audio_t=None):
    """An LTX AV latent of `t` latent frames: 8(t-1)+1 video frames, ~25 audio latents a second at 24 fps."""
    frames = 8 * (t - 1) + 1
    audio_t = audio_t or round(frames * 25 / 24)
    video = torch.randn(1, 8, t, 3, 5)
    audio = torch.randn(1, 8, audio_t, FREQ)
    vmask = torch.ones(1, 1, t, 1, 1)
    vmask[:, :, 0] = 0  # LTXVImgToVideoInplace: the first frame is the given image, not denoised
    mask = NestedTensor([vmask.expand_as(video).clone(), torch.ones_like(audio)])
    return {"samples": NestedTensor([video, audio]), "noise_mask": mask}


def _run(latent, window, guider=None, overlap=17):
    guider = guider or _Guider()
    sigmas = torch.tensor([0.85, 0.725, 0.4219, 0.0])
    out, _, info = sampler.MpiWindowedSampler().doit(
        _Noise(), guider, comfy.samplers.sampler_object("euler"), sigmas, latent, window, overlap, 1)
    return out["samples"].unbind(), info, guider


def check_frame_count():
    # 121 frames = 16 latent, 241 = 31: the LTX lengths the app asks for
    assert sampler.latent_to_frames(16, 1, 8, causal=True) == 121
    assert sampler.latent_to_frames(31, 1, 8, causal=True) == 241
    assert sampler.latent_to_frames(0, 1, 8, causal=True) == 0
    assert sampler.frames_to_latent(97, 1, 8, 100, causal=True) == 13  # 8*12+1
    assert sampler.frames_to_latent(17, 1, 8, 100, causal=True) == 3
    # H3 is untouched
    assert sampler.latent_to_frames(37, 5, 4) == 124
    print("ok frame count: 16 latent = 121 frames, 31 = 241, H3 unchanged")


def check_windows_equal_one_pass():
    latent = _latent(16)
    one, info1, _ = _run(latent, 10_000)
    many, info, guider = _run(latent, 49)  # 49 frames = 7 latent per window
    assert info1.startswith("single pass"), info1
    n = int(info.split()[0])
    assert n > 1, info
    assert "121 video frames (16 latent)" in info, info
    diff = (one[0] - many[0]).abs().max().item()
    assert diff < 1e-5, f"windowed video differs from one pass by {diff}"
    # The audio comes back as the original stream, whole.
    assert many[1].shape == latent["samples"].unbind()[1].shape
    assert torch.equal(many[1], latent["samples"].unbind()[1])
    # Every pass got its audio cut along TIME (dim 2) and the 16 bins intact, and a mask that matches.
    audio_t = latent["samples"].unbind()[1].shape[2]
    for shapes, masks in guider.seen:
        assert shapes[1][3] == FREQ, shapes
        assert 0 < shapes[1][2] < audio_t, shapes
        assert masks[1] == shapes[1], (masks, shapes)
    # The pinned first frame stayed pinned.
    assert torch.allclose(many[0][:, :, 0], latent["samples"].unbind()[0][:, :, 0])
    print(f"ok windows == one pass: {info.split('.')[0]}; audio cut on time, bins intact, mask follows")


def check_guides_stay_one_pass():
    kf = torch.zeros(1, 3, 15)  # LTXVAddGuide records appended frames' positions here
    guider = _Guider({"positive": [{"keyframe_idxs": kf}], "negative": [{"keyframe_idxs": kf}]})
    out, info, guider = _run(_latent(16), 49, guider)
    assert info.startswith("single pass") and "guide frames" in info, info
    assert len(guider.seen) == 1, guider.seen
    print("ok appended guides: one pass, said so")


def check_step_windows_equal_one_pass():
    """MpiStepWindowedSampler reads the same LTXV facts: causal count, audio on dim 2, nested mask."""
    latent = _latent(16)
    sigmas = torch.tensor([0.85, 0.725, 0.4219, 0.0])
    euler = comfy.samplers.sampler_object("euler")
    one, _, info1 = sampler.MpiStepWindowedSampler().doit(
        _Noise(), _Guider(), euler, sigmas, latent, 10_000, 17, 1)
    guider = _Guider()
    many, _, info = sampler.MpiStepWindowedSampler().doit(
        _Noise(), guider, euler, sigmas, latent, 49, 17, 1)
    assert info1.startswith("single pass") and "121 video" in info1, info1
    assert int(info.split()[0]) > 1 and "121 video frames (16 latent)" in info, info
    one, many = one["samples"].unbind(), many["samples"].unbind()
    for k in range(2):
        diff = (one[k] - many[k]).abs().max().item()
        assert diff < 1e-4, f"stream {k}: step-windowed differs from one pass by {diff}"
    for shapes, masks in guider.seen:
        assert shapes[1][3] == FREQ and masks[1] == shapes[1], (shapes, masks)
    print(f"ok step windows == one pass: {info.split('.')[0]}")


def check_later_windows_read_real_frame_times():
    """LTX 2.3/2.5 ship causal_temporal_positioning on. A window past latent 0 must be sampled with it OFF,
    or the model reads its picture 7 frames (0.29 s) ahead of its audio: failed lip-sync + ghosting in the
    talking-people test (Cubric Studio MPI-1450). Window 0 keeps it, and the model is left as found."""
    sigmas = torch.tensor([0.85, 0.725, 0.4219, 0.0])
    euler = comfy.samplers.sampler_object("euler")
    for node in (sampler.MpiWindowedSampler(), sampler.MpiStepWindowedSampler()):
        guider = _Guider()
        dm = types.SimpleNamespace(causal_temporal_positioning=True)
        guider.model_patcher.model.diffusion_model = dm
        flags = []
        inner = guider.sample

        def sample(*a, **k):
            flags.append(dm.causal_temporal_positioning)
            return inner(*a, **k)
        guider.sample = sample
        _, _, info = node.doit(_Noise(), guider, euler, sigmas, _latent(16), 49, 17, 1)
        n = int(info.split()[0])
        per_pass = [True] + [False] * (n - 1)
        want = per_pass * (3 if isinstance(node, sampler.MpiStepWindowedSampler) else 1)
        assert flags == want, (type(node).__name__, flags)
        assert dm.causal_temporal_positioning is True, "the flag must be restored"
    print(f"ok later windows: causal positions off past window 1 ({n} windows), restored after")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_frame_count()
    check_windows_equal_one_pass()
    check_guides_stay_one_pass()
    check_step_windows_equal_one_pass()
    check_later_windows_read_real_frame_times()
