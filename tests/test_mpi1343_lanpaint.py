"""CPU checks for the lanpaint/ port (MPI-1343): tiny tensors, stub VAE and stub model, no weights. Each node
and the LanPaint inner step / whole sampler loop are compared with the original LanPaint pack (pin 9fe91955),
loaded from the engine folder when it is on disk. Run with:

    G:/ComfyUi/python_embeded/python.exe tests/test_mpi1343_lanpaint.py
"""
import os
import sys
import warnings
from importlib.machinery import SourceFileLoader

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "G:/ComfyUi/ComfyUI"
ORIG = ("C:/AI/Mpi/Cubric-Vision/engine/ComfyUI_windows_portable/ComfyUI/custom_nodes/"
        "LanPaint/src/LanPaint")
sys.path.insert(0, COMFY)

import comfy.model_management as mm  # noqa: E402
mm.get_torch_device = lambda: torch.device("cpu")
from comfy.model_base import ModelType  # noqa: E402
import comfy.samplers  # noqa: E402
from comfy.k_diffusion import sampling as kds  # noqa: E402
import comfy.sampler_helpers  # noqa: E402,F401
import comfy.utils  # noqa: E402,F401
import comfyui_version  # noqa: E402,F401
import nodes  # noqa: E402,F401


def _load_pkg(name, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SourceFileLoader(name, os.path.join(path, "__init__.py")).load_module()


# ---- import the port: sys.path and the module table must only grow by the port and comfy itself
path_before = list(sys.path)
mods_before = set(sys.modules)
port = _load_pkg("mpi_lp_port", os.path.join(ROOT, "lanpaint"))
nodes_port = sys.modules["mpi_lp_port.nodes"]
sampling_port = sys.modules["mpi_lp_port.sampling"]
lang_port = sys.modules["mpi_lp_port.langevin"]
new_mods = set(sys.modules) - mods_before

orig = None
if os.path.isdir(ORIG):
    orig = _load_pkg("mpi_lp_orig", ORIG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lang_orig = SourceFileLoader("mpi_lp_orig.lanpaint", ORIG + "/lanpaint.py").load_module()
        nodes_orig = SourceFileLoader("mpi_lp_orig.nodes", ORIG + "/nodes.py").load_module()

NAMES = {"MpiLanPaintKSampler": "LanPaint_KSampler",
         "MpiLanPaintImageEncode": "LanPaint_ImageEncode",
         "MpiLanPaintImageDecode": "LanPaint_ImageDecode"}


def same(a, b, what=""):
    assert a.shape == b.shape and a.dtype == b.dtype, (what, a.shape, b.shape, a.dtype, b.dtype)
    assert torch.equal(a, b), (what, float((a - b).abs().max()))


def test_import_side_effects():
    assert sys.path == path_before, "sys.path changed"
    own = [m for m in new_mods if not m.startswith("mpi_lp_port")]
    assert not own, own
    assert set(port.NODE_CLASS_MAPPINGS) == set(NAMES)
    assert set(port.NODE_DISPLAY_NAME_MAPPINGS) == set(NAMES)
    for k, v in port.NODE_DISPLAY_NAME_MAPPINGS.items():
        assert v.startswith("Mpi LanPaint "), v
        assert port.NODE_CLASS_MAPPINGS[k].CATEGORY == "MpiNodes/Sampling"


def test_input_types():
    if not orig:
        return
    for new, old in NAMES.items():
        a = port.NODE_CLASS_MAPPINGS[new]
        b = nodes_orig.NODE_CLASS_MAPPINGS[old]
        assert a.INPUT_TYPES() == b.INPUT_TYPES(), new
        assert a.RETURN_TYPES == b.RETURN_TYPES and a.FUNCTION == b.FUNCTION, new
        assert getattr(a, "RETURN_NAMES", None) == getattr(b, "RETURN_NAMES", None), new
    print("   INPUT_TYPES equal for all three nodes")


class StubVAE:
    """Deterministic: encode = 8x subsample padded to 4 channels; decode = 8x nearest upsample, 2 px short."""
    def __init__(self, five_d=False):
        self.five_d = five_d

    def encode(self, image):
        x = image.movedim(-1, 1)[:, :, ::8, ::8]
        z = torch.cat([x, x[:, :1] * 0.5], dim=1)
        return z.unsqueeze(2).repeat(1, 1, 2, 1, 1) if self.five_d else z

    def decode(self, z):
        if self.five_d:
            z = z[:, :3, 0]
        x = torch.nn.functional.interpolate(z[:, :3], scale_factor=8, mode="nearest")
        x = x[:, :, :-2, :-2].movedim(1, -1)
        return x.unsqueeze(0) if self.five_d else x


def rnd(*shape, seed=0):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed))


def test_encode_decode():
    if not orig:
        return
    enc, dec = port.NODE_CLASS_MAPPINGS["MpiLanPaintImageEncode"](), port.NODE_CLASS_MAPPINGS["MpiLanPaintImageDecode"]()
    oenc, odec = nodes_orig.NODE_CLASS_MAPPINGS["LanPaint_ImageEncode"](), nodes_orig.NODE_CLASS_MAPPINGS["LanPaint_ImageDecode"]()
    n = 0
    for five_d in (False, True):
        vae = StubVAE(five_d)
        img = rnd(1, 32, 40, 3)
        masks = [None, (rnd(32, 40, seed=1) > 0.5).float(), (rnd(1, 32, 40, seed=2) > 0.5).float(),
                 (rnd(1, 1, 32, 40, seed=3) > 0.5).float(), (rnd(16, 20, seed=4) > 0.5).float()]
        for m in masks:
            a, b = enc.encode(img, vae, m)[0], oenc.encode(img, vae, m)[0]
            assert set(a) == set(b)
            for k in a:
                same(a[k], b[k], f"encode {k} 5d={five_d}")
            n += 1
            z = {"samples": a["samples"] + 0.1}
            for image in (None, img):
                for mk in (None, m if m is not None else (rnd(32, 40, seed=5) > 0.5).float()):
                    for ov in (1, 9, 21):
                        x, y = dec.decode(z, vae, image, mk, ov)[0], odec.decode(z, vae, image, mk, ov)[0]
                        same(x, y, f"decode ov={ov} 5d={five_d}")
                        n += 1
    print(f"   encode/decode: {n} cases equal to the original")


class StubSampling:
    sigma_max = 1.0

    def __init__(self, flow):
        self.flow = flow

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if self.flow:
            return sigma * noise + (1.0 - sigma) * latent_image
        return latent_image + noise * sigma

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class StubBase:
    def __init__(self, flow, mtype):
        self.model_type = mtype
        self.model_sampling = StubSampling(flow)


class StubPatcher:
    pass


class StubWrap:
    """What KSamplerX0Inpaint wraps: callable (x, sigma, model_options, seed) -> (denoised, denoised_big)."""
    def __init__(self, flow, mtype, patcher):
        self.inner_model = StubBase(flow, mtype)
        self.model_patcher = patcher

    def __call__(self, x, sigma, model_options={}, seed=None):
        s = sigma.reshape(-1, *([1] * (x.ndim - 1)))
        return x * 0.9 - 0.05 * s, x * 0.8 + 0.1


def _patcher(prefix, num_steps, mode_cfg):
    p = StubPatcher()
    setattr(p, prefix + "StepSize", 0.2)
    setattr(p, prefix + "Lambda", 5.0)
    setattr(p, prefix + "Beta", 1.)
    setattr(p, prefix + "NumSteps", num_steps)
    setattr(p, prefix + "MinStepFrac", 1.0)
    setattr(p, prefix + "Friction", 15.)
    setattr(p, prefix + "EarlyStop", 1)
    setattr(p, prefix + "InnerThreshold", 0.0)
    setattr(p, prefix + "InnerPatience", 1)
    setattr(p, prefix + "cfg_BIG", mode_cfg)
    return p


def _run_sampler(kind, flow, mtype, sampler_fn, steps, num_steps, cfg_big, seed):
    """Run the carried KSAMPLER.sample (kind 'port') or the original's (kind 'orig') on a masked latent."""
    if kind == "port":
        cls, prefix = sampling_port.KSAMPLER, "MpiLanPaint_"
    else:
        cls, prefix = nodes_orig.KSAMPLER, "LanPaint_"
    sigmas = (torch.linspace(0.95, 0.0, steps + 1) if flow else torch.linspace(8.0, 0.0, steps + 1))
    latent = rnd(1, 4, 8, 8, seed=11)
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., 2:6, 2:6] = 1.0
    wrap = StubWrap(flow, mtype, _patcher(prefix, num_steps, cfg_big))
    torch.manual_seed(seed)
    noise = torch.randn(latent.shape)
    return cls(sampler_fn).sample(wrap, sigmas.clone(), {"seed": seed}, None, noise, latent_image=latent.clone(),
                                  denoise_mask=mask, disable_pbar=True), wrap


def test_sampler_equivalence():
    if not orig:
        return
    n = 0
    for flow, mtype in ((True, ModelType.FLUX), (True, ModelType.FLOW), (False, ModelType.EPS)):
        for sfn in (kds.sample_euler, kds.sample_dpmpp_2m):
            for num_steps in (0, 3, 5):
                for cfg_big in (5.0, -0.5):
                    a, wa = _run_sampler("port", flow, mtype, sfn, 6, num_steps, cfg_big, 7)
                    b, wb = _run_sampler("orig", flow, mtype, sfn, 6, num_steps, cfg_big, 7)
                    same(a, b, f"sampler flow={flow} {mtype} {sfn.__name__} n={num_steps}")
                    assert wa.cfg_BIG == wb.cfg_BIG
                    n += 1
    print(f"   whole sampler loop: {n} runs bitwise equal to the original")


def test_inner_step_equivalence():
    """One KSamplerX0Inpaint call (the inner paint step) on its own, x updated in place as well."""
    if not orig:
        return
    for flow, mtype in ((True, ModelType.FLUX), (False, ModelType.EPS)):
        outs = []
        for kind, sm, lm, prefix in (("port", sampling_port, lang_port, "MpiLanPaint_"),
                                     ("orig", nodes_orig, lang_orig, "LanPaint_")):
            wrap = StubWrap(flow, mtype, _patcher(prefix, 4, 5.0))
            sigmas = torch.tensor([0.8, 0.5, 0.2, 0.0]) if flow else torch.tensor([6.0, 3.0, 1.0, 0.0])
            k = sm.KSamplerX0Inpaint(wrap, sigmas)
            latent = rnd(1, 4, 8, 8, seed=3)
            k.latent_image = latent.clone()
            k.noise = rnd(1, 4, 8, 8, seed=4)
            k.PaintMethod = lm.LanPaint(wrap, 4, 15., 5.0, 1., 0.2, IS_FLUX=(mtype == ModelType.FLUX),
                                        IS_FLOW=flow, MinStepFrac=1.0)
            k.LanPaint_early_stop = 1
            k.LanPaint_min_step_frac = 1.0
            mask = torch.zeros(1, 1, 8, 8)
            mask[..., 1:5, 3:7] = 1.0
            x = rnd(1, 4, 8, 8, seed=5)
            torch.manual_seed(123)
            out = k(x, sigmas[1:2], mask, model_options={}, seed=1)
            outs.append((out, x))
        same(outs[0][0], outs[1][0], f"inner out flow={flow}")
        same(outs[0][1], outs[1][1], f"inner x flow={flow}")
    print("   inner paint step: output and in-place x equal to the original")


def test_reshape_mask_and_override():
    if orig:
        for shape, video in (((1, 4, 8, 8), False), ((2, 4, 8, 8), False), ((1, 16, 5, 8, 8), True),
                             ((1, 16, 5, 8, 8), False)):
            for m in (rnd(32, 32, seed=1), rnd(1, 32, 32, seed=2), rnd(5, 32, 32, seed=3), rnd(1, 1, 32, 32, seed=4)):
                try:
                    a = sampling_port.reshape_mask(m, shape, video)
                except Exception as e:  # the original must fail the same way
                    try:
                        nodes_orig.reshape_mask(m, shape, video)
                    except type(e):
                        continue
                    raise
                same(a, nodes_orig.reshape_mask(m, shape, video), f"reshape {shape} {video}")
    # the hooks are in place only inside the context and everything is restored afterwards
    before = (comfy.samplers.CFGGuider.outer_sample, comfy.samplers.CFGGuider.predict_noise,
              comfy.samplers.KSAMPLER.sample, comfy.sampler_helpers.prepare_mask)
    with sampling_port.override_sample_function():
        assert comfy.samplers.KSAMPLER.sample is sampling_port.KSAMPLER.sample
        assert comfy.samplers.CFGGuider.outer_sample is sampling_port.CFGGuider_LanPaint.outer_sample
    after = (comfy.samplers.CFGGuider.outer_sample, comfy.samplers.CFGGuider.predict_noise,
             comfy.samplers.KSAMPLER.sample, comfy.sampler_helpers.prepare_mask)
    assert before == after
    print("   reshape_mask equal; hooks restored after the context")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all lanpaint tests passed" + ("" if orig else " (original pack absent: equivalence skipped)"))
