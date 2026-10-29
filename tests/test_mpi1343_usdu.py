"""CPU checks for usdu/ MpiUltimateSDUpscale (MPI-1343), tiny tensors, no weights. Where the original
ComfyUI_UltimateSDUpscale pack is on disk the port is compared with the original node: same
INPUT_TYPES, and identical output tensors with a stub VAE / upscale model / sampler. Run with:

    G:/ComfyUi/python_embeded/python.exe tests/test_mpi1343_usdu.py
"""
import importlib
import importlib.util
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "G:/ComfyUi/ComfyUI"
ORIG_CANDIDATES = [
    "C:/AI/Mpi/Cubric-Vision/engine/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui_ultimatesdupscale",
    COMFY + "/custom_nodes/comfyui_ultimatesdupscale",
]
sys.path.insert(0, COMFY)

import comfy.model_management as mm  # noqa: E402
mm.get_torch_device = lambda: torch.device("cpu")


def _load_pkg(name, path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- import the port first and prove it leaves sys.path / sys.modules alone
path_before = list(sys.path)
mods_before = set(sys.modules)
port = _load_pkg("mpi_usdu_port", os.path.join(ROOT, "usdu"))
new_mods = set(sys.modules) - mods_before
PATH_AFTER = list(sys.path)
USDU_DIR = os.path.normcase(os.path.join(ROOT, "usdu"))
# any module NOT named mpi_usdu_port.* whose file lives in usdu/ was injected under a top-level name
# (ComfyUI has its own top-level utils package; that one is not ours and lives in the engine)
INJECTED = [m for m in set(sys.modules) - mods_before if not m.startswith("mpi_usdu_port")
            and os.path.normcase(getattr(sys.modules[m], "__file__", None) or "").startswith(USDU_DIR)]
node_mod = sys.modules["mpi_usdu_port.node"]
proc_mod = sys.modules["mpi_usdu_port.modules.processing"]
shared_mod = sys.modules["mpi_usdu_port.modules.shared"]


def test_import_has_no_side_effects():
    assert PATH_AFTER == path_before, "usdu import changed sys.path"
    assert not INJECTED, "top-level module injected: %s" % INJECTED
    print("   sys.path unchanged; no top-level module sourced from usdu/ (%d modules loaded in all)" % len(new_mods))


def test_mapping():
    assert set(port.NODE_CLASS_MAPPINGS) == {"MpiUltimateSDUpscale"}
    assert port.NODE_DISPLAY_NAME_MAPPINGS == {"MpiUltimateSDUpscale": "Mpi Ultimate SD Upscale"}
    cls = port.NODE_CLASS_MAPPINGS["MpiUltimateSDUpscale"]
    assert cls.CATEGORY == "MpiNodes/Upscale" and cls.FUNCTION == "upscale" and cls.RETURN_TYPES == ("IMAGE",)


def _find_orig():
    for p in ORIG_CANDIDATES:
        if os.path.isfile(os.path.join(p, "usdu_nodes.py")):
            return p
    return None


ORIG_PATH = _find_orig()
orig_cls = None
if ORIG_PATH:
    # the original pack inserts its own dir on sys.path while loading and cleans up afterwards
    orig_pkg = _load_pkg("usdu_orig_pkg", ORIG_PATH)
    orig_cls = orig_pkg.NODE_CLASS_MAPPINGS["UltimateSDUpscale"]
    sys.modules.setdefault("usdu_orig_pkg", orig_pkg)
mine = port.NODE_CLASS_MAPPINGS["MpiUltimateSDUpscale"]


def test_inputs_match_original():
    if not orig_cls:
        return
    a, b = mine.INPUT_TYPES(), orig_cls.INPUT_TYPES()
    assert list(a) == list(b) == ["required"], (list(a), list(b))
    assert list(a["required"]) == list(b["required"]), set(a["required"]) ^ set(b["required"])
    for k in a["required"]:
        assert a["required"][k] == b["required"][k], (k, a["required"][k], b["required"][k])
    for attr in ("RETURN_TYPES", "FUNCTION", "OUTPUT_TOOLTIPS", "DESCRIPTION"):
        assert getattr(mine, attr) == getattr(orig_cls, attr), attr
    print("   %d inputs identical (names, order, types, defaults, choices, tooltips)" % len(a["required"]))


# ---- stubs shared by both nodes
class StubVAE:
    def encode(self, pixels):
        x = pixels[..., :3].movedim(-1, 1)
        return torch.nn.functional.avg_pool2d(x, 8)

    def decode(self, samples):
        return torch.nn.functional.interpolate(samples, scale_factor=8, mode="nearest").movedim(1, -1).clamp(0, 1)

    def temporal_compression_decode(self):
        return None

    def spacial_compression_decode(self):
        return 8

    def decode_tiled(self, samples, *a, **k):
        return self.decode(samples) * 0.97


class StubModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        return StubModel()


def stub_upscale(cls, upscale_model, image):
    x = image.movedim(-1, 1)
    x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    return (x.movedim(1, -1),)


import comfy_extras.nodes_upscale_model as num  # noqa: E402
num.ImageUpscaleWithModel.execute = classmethod(stub_upscale)

CALLS = []


def make_sampler(tag):
    def ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
        s = latent["samples"]
        CALLS.append((tag, tuple(s.shape), seed, steps, cfg, sampler_name, scheduler, denoise, len(positive)))
        out = s * 0.8 + 0.1 * torch.sin(s * 5 + denoise) + 0.002 * (seed % 5)
        return ({"samples": out},)
    return ksampler


def _orig_globals():
    # the module dict of the original pack's modules.processing, reached through a function it owns
    return orig_cls.upscale.__globals__["StableDiffusionProcessing"].__init__.__globals__


def run(cls, tag, img, **kw):
    cond = [[torch.zeros(1, 4, 8), {}]]
    args = dict(image=img, model=StubModel(), positive=cond, negative=cond, vae=StubVAE(), upscale_by=2.0,
                seed=7, steps=8, cfg=1.0, sampler_name="euler", scheduler="simple", denoise=0.35,
                upscale_model=object(), mode_type="Linear", tile_width=64, tile_height=64, mask_blur=4,
                tile_padding=16, seam_fix_mode="None", seam_fix_denoise=0.6, seam_fix_width=32,
                seam_fix_mask_blur=4, seam_fix_padding=16, force_uniform_tiles=True, tiled_decode=False,
                batch_size=1)
    args.update(kw)
    return cls().upscale(**args)[0]


def test_equivalent_to_original():
    if not orig_cls:
        return
    g = torch.Generator().manual_seed(3)
    img = torch.rand(1, 64, 96, 3, generator=g)
    img2 = torch.rand(2, 64, 96, 3, generator=g)
    proc_mod.common_ksampler = make_sampler("port")
    og = _orig_globals()
    og["common_ksampler"] = make_sampler("orig")
    cases = [
        ("workflow default: Linear, no seam fix, uniform", img, {}),
        ("workflow: non-uniform tiles", img, dict(force_uniform_tiles=False)),
        ("workflow: seam_fix_denoise 1.0, cfg 1.4, lcm", img, dict(seam_fix_denoise=1.0, cfg=1.4, sampler_name="lcm", steps=2)),
        ("Chess", img, dict(mode_type="Chess")),
        ("None redraw + Band Pass", img, dict(mode_type="None", seam_fix_mode="Band Pass")),
        ("Linear + Band Pass", img, dict(seam_fix_mode="Band Pass")),
        ("Linear + Half Tile", img, dict(seam_fix_mode="Half Tile")),
        ("Chess + Half Tile + Intersections", img, dict(mode_type="Chess", seam_fix_mode="Half Tile + Intersections")),
        ("batch_size 2 Linear", img, dict(batch_size=2)),
        ("batch_size 3 Chess, 2 images", img2, dict(batch_size=3, mode_type="Chess")),
        ("tiled_decode", img, dict(tiled_decode=True)),
        ("upscale_by 1.5, mask_blur 0", img, dict(upscale_by=1.5, mask_blur=0, tile_padding=0)),
    ]
    for name, im, kw in cases:
        CALLS.clear()
        a = run(mine, "port", im, **kw)
        b = run(orig_cls, "orig", im, **kw)
        assert a.shape == b.shape and a.dtype == b.dtype, (name, a.shape, b.shape)
        assert torch.equal(a, b), (name, float((a - b).abs().max()))
        pa = [c[1:] for c in CALLS if c[0] == "port"]
        pb = [c[1:] for c in CALLS if c[0] == "orig"]
        assert pa == pb and pa, (name, pa, pb)
        print("   equal: %-46s out %s, %d sampler calls" % (name, tuple(a.shape), len(pa)))


def test_no_banned_tokens():
    words = ["ev" + "al", "ex" + "ec", "comp" + "ile(", "__imp" + "ort__", "import_" + "module", "sub" + "process",
             "url" + "lib", "req" + "uests", "sys." + "path", "sys." + "modules", "os." + "system"]
    pat = re.compile(r"(?:" + "|".join(re.escape(w) for w in words) + r")(?![A-Za-z_])")
    for d, _, files in os.walk(os.path.join(ROOT, "usdu")):
        for f in files:
            if f.endswith(".py"):
                src = open(os.path.join(d, f), encoding="utf8").read()
                assert not pat.search(src), (f, pat.search(src).group(0))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all usdu tests passed" + ("" if orig_cls else " (original pack absent: equivalence skipped)"))
