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


class _BiasedGuider(_Guider):
    """Same model, other conditioning: shifts every prediction, as another prompt would."""
    calls = 0

    def sample(self, *a, **k):
        self.calls += 1
        return super().sample(*a, **k) + 0.5


def _run(samples, window, overlap, name="euler", **window_guiders):
    sigmas = torch.linspace(1.0, 0.0, 9)
    smp = comfy.samplers.sampler_object(name)
    node = sampler.MpiStepWindowedSampler()
    out, _, info = node.doit(_Noise(), _Guider(), smp, sigmas, {"samples": samples}, window, overlap, 1,
                             **window_guiders)
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


def check_window_guiders():
    """window_guider_N drives window N only; unwired windows keep the main guider."""
    samples = torch.randn(1, 4, 30, 3, 5)
    base, info = _run(samples, 12, 4)
    spans = eval(info.split("spans=")[1].split(" audio=")[0])
    same, _ = _run(samples, 12, 4, window_guider_2=_Guider(), window_guider_8=None)
    assert (base - same).abs().max().item() < 1e-6, "an equivalent window guider changed the result"
    biased = _BiasedGuider()
    got, info = _run(samples, 12, 4, window_guider_2=biased, window_guider_8=_Guider())
    assert biased.calls == 8, biased.calls  # one call per step, window 2 only
    assert "Own guider on windows [2]" in info and "window_guider_8" in info, info
    # A frame-local model: frames outside window 2 never see its guider.
    touched = set(range(*spans[1]))
    diff = (got - base).abs().amax(dim=(0, 1, 3, 4))
    for t in range(samples.shape[2]):
        assert (diff[t] > 1e-3) == (t in touched), f"frame {t}: diff {diff[t]}, window 2 is {spans[1]}"
    other = _Guider()
    other.model_patcher = types.SimpleNamespace(model=types.SimpleNamespace(), load_device="cpu")
    try:
        _run(samples, 12, 4, window_guider_1=other)
    except ValueError as e:
        assert "window_guider_1" in str(e), e
    else:
        raise AssertionError("a guider on another model must be refused")
    types_ = sampler.MpiStepWindowedSampler.INPUT_TYPES()
    assert list(types_["required"]) == list(sampler.MpiWindowedSampler.INPUT_TYPES()["required"])
    assert "optional" not in sampler.MpiWindowedSampler.INPUT_TYPES()
    print("ok window guiders: window 2 only, unwired = main, other model refused")


def check_commit_steps():
    """commit_steps: the earlier window alone writes each overlap for those steps."""
    samples = torch.randn(1, 4, 30, 3, 5)
    base, info = _run(samples, 12, 4)
    spans = eval(info.split("spans=")[1].split(" audio=")[0])
    same, info = _run(samples, 12, 4, commit_steps=3)
    assert (base - same).abs().max().item() < 1e-5, "committing with one guider changed a frame-local result"
    assert "first 3 steps" in info, info
    # Every step committed: window 2's guider reaches only the frames window 2 owns, never window 1's overlap.
    got, _ = _run(samples, 12, 4, window_guider_2=_BiasedGuider(), commit_steps=8)
    owned = set(range(spans[0][1], spans[1][1]))
    diff = (got - base).abs().amax(dim=(0, 1, 3, 4))
    for t in range(samples.shape[2]):
        assert (diff[t] > 1e-3) == (t in owned), f"frame {t}: diff {diff[t]}, window 2 owns {sorted(owned)}"
    # Committed for some steps only: the overlap with window 1 still feels window 2 on the cross-fade steps.
    part, _ = _run(samples, 12, 4, window_guider_2=_BiasedGuider(), commit_steps=4)
    assert (part - base).abs().amax(dim=(0, 1, 3, 4))[spans[1][0]].item() > 1e-3
    print(f"ok commit steps: window 2 owns {min(owned)}-{max(owned)}, the overlap with window 1 stays window 1's")


if __name__ == "__main__":
    torch.manual_seed(0)
    check(torch.randn(1, 4, 30, 3, 5), "video only")
    check(NestedTensor([torch.randn(1, 4, 30, 3, 5), torch.randn(1, 8, 60)]), "video + audio")
    check_noisy(NestedTensor([torch.randn(1, 4, 30, 3, 5), torch.randn(1, 8, 60)]), "lcm, two window layouts agree")
    check_window_guiders()
    check_commit_steps()
