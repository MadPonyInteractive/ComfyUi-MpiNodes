"""CPU check for MpiStepWindowedSampler: windows fused every step equal one pass when the model is frame-local.
Real ComfyUI core (KSAMPLER, euler, NestedTensor, pack/unpack), a fake model. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_step_windowed.py

A model whose prediction for a frame reads only that frame cannot tell a window from the whole clip, so any
difference from the single pass is the node's own slicing, start injection or blending -- exactly what is tested.
"""
import os
import sys
import types

import torch

COMFY = os.environ.get("COMFY_ROOT", "G:/ComfyUi/ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = sys.argv[:1] + ["--cpu"]

import comfy.model_sampling  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.nested_tensor import NestedTensor  # noqa: E402

import sampler  # noqa: E402

sampler.comfy.sample.fix_empty_latent_channels = lambda patcher, samples, *a: samples
sampler.latent_preview.prepare_callback = lambda *a: None


class _Sampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    pass


class _Wrap:
    """model_wrap: denoised = a frame-local nonlinear function of x and sigma."""
    def __init__(self):
        self.inner_model = types.SimpleNamespace(model_sampling=_Sampling())
        self.model_patcher = types.SimpleNamespace(  # read by core's step log and by lcm
            model=self, get_model_object=lambda name: self.inner_model.model_sampling)
        self.cfg = 1.0

    def __call__(self, x, sigma, model_options={}, seed=None):
        s = sigma.view([-1] + [1] * (x.ndim - 1))
        return 0.6 * x / (1 + s) + 0.1 * torch.sin(3 * x)


class _Guider:
    """Mirrors CFGGuider.sample's packing around sampler.sample, nothing else."""
    model_patcher = types.SimpleNamespace(
        model=types.SimpleNamespace(latent_format=types.SimpleNamespace(temporal_downscale_ratio=1)),
        load_device="cpu")

    def sample(self, noise, latent, smp, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        shapes = None
        if getattr(latent, "is_nested", False):
            latent, shapes = comfy.utils.pack_latents(latent.unbind())
            noise, _ = comfy.utils.pack_latents(noise.unbind())
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


def _run(samples, window, overlap, name="euler"):
    sigmas = torch.linspace(1.0, 0.0, 9)
    smp = comfy.samplers.sampler_object(name)
    node = sampler.MpiStepWindowedSampler()
    out, _, info = node.doit(_Noise(), _Guider(), smp, sigmas, {"samples": samples}, window, overlap, 1)
    return out["samples"], info


def check_noisy(samples, label):
    """lcm adds fresh noise every step, drawn by the node clip-wide, so the window LAYOUT must not matter."""
    a, _ = _run(samples, 12, 4, "lcm")
    b, _ = _run(samples, 20, 6, "lcm")
    for x, y in zip(_parts(a), _parts(b)):
        diff = (x - y).abs().max().item()
        assert diff < 1e-5, f"{label}: lcm result depends on the window layout by {diff}"
    print(f"ok {label}")


def _parts(t):
    return t.unbind() if getattr(t, "is_nested", False) else [t]


def check(samples, label):
    one, info1 = _run(samples, 10_000, 4)
    many, info = _run(samples, 12, 4)
    assert info1.startswith("single pass"), info1
    assert int(info.split()[0]) > 1, info
    for a, b in zip(_parts(one), _parts(many)):
        diff = (a - b).abs().max().item()
        assert diff < 1e-5, f"{label}: fused windows differ from one pass by {diff}"
    print(f"ok {label}: {info.split('.')[0]}")


if __name__ == "__main__":
    torch.manual_seed(0)
    check(torch.randn(1, 4, 30, 3, 5), "video only")
    check(NestedTensor([torch.randn(1, 4, 30, 3, 5), torch.randn(1, 8, 60)]), "video + audio")
    check_noisy(NestedTensor([torch.randn(1, 4, 30, 3, 5), torch.randn(1, 8, 60)]), "lcm, two window layouts agree")
