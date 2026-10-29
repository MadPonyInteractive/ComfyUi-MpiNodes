"""CPU checks for pano/ (MPI-1342): the five ported Mickmumpitz pano nodes. No GPU, no weights.
Where the original pack is on disk the port is compared to it on fixed inputs. Run with:

    G:/ComfyUi/python_embeded/python.exe tests/test_pano.py
"""
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "G:/ComfyUi/ComfyUI"
ORIG = COMFY + "/custom_nodes/ComfyUI-Mickmumpitz-Nodes/nodes/panorama_tools"


def _load_pkg(name, path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


port = _load_pkg("mpi_pano_port", os.path.join(ROOT, "pano"))
orig = _load_pkg("mick_pano_orig", ORIG) if os.path.isdir(ORIG) else None
if os.path.isdir(COMFY):
    sys.path.insert(0, COMFY)

PAIRS = {
    "MpiPanoWarp": "MickmumpitzPanoWarp",
    "MpiPanoHarmonizeBoundary": "MickmumpitzPanoHarmonizeBoundary",
    "MpiPanoKrea2Reference": "MickmumpitzPanoKrea2Reference",
    "MpiPanoRollHorizontal": "MickmumpitzPanoRollHorizontal",
    "MpiPanoSeamMask": "MickmumpitzPanoSeamMask",
}


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y), "port differs from original"


def test_mappings():
    assert set(port.NODE_CLASS_MAPPINGS) == set(PAIRS)
    assert set(port.NODE_DISPLAY_NAME_MAPPINGS) == set(PAIRS)
    for k, v in port.NODE_DISPLAY_NAME_MAPPINGS.items():
        assert v.startswith("Mpi "), v
    for k, cls in port.NODE_CLASS_MAPPINGS.items():
        cls.INPUT_TYPES()


def test_inputs_match_original():
    if orig is None:
        return
    for new, old in PAIRS.items():
        a = port.NODE_CLASS_MAPPINGS[new]
        b = orig.NODE_CLASS_MAPPINGS[old]
        assert a.INPUT_TYPES() == b.INPUT_TYPES(), new
        assert a.RETURN_TYPES == b.RETURN_TYPES, new
        assert getattr(a, "RETURN_NAMES", None) == getattr(b, "RETURN_NAMES", None), new


def test_warp():
    torch.manual_seed(0)
    img = torch.rand(1, 40, 60, 3)
    kw = dict(width=128, height=64, length=1, h_fov_deg=70, yaw_deg=0, pitch_deg=0, mask_feather=0,
              supersample=2, grid_spacing_deg=10, guide_opacity=0.55, fill_color="#000000")
    out = port.NODE_CLASS_MAPPINGS["MpiPanoWarp"]().build(img, **kw)
    assert out[0].shape == (1, 64, 128, 3) and out[2].shape == (1, 64, 128) and out[3].shape == (1, 64, 128, 3)
    assert 0 < float(out[2].mean()) < 1  # a known patch and a hole
    if orig:
        _same(out, orig.NODE_CLASS_MAPPINGS["MickmumpitzPanoWarp"]().build(img, **kw))
        kw.update(yaw_deg=30, pitch_deg=-10, mask_feather=4, length=5, fill_color="128,128,128")
        a = port.NODE_CLASS_MAPPINGS["MpiPanoWarp"]().build(img, **kw)
        _same(a, orig.NODE_CLASS_MAPPINGS["MickmumpitzPanoWarp"]().build(img, **kw))


def test_roll():
    img = torch.arange(2 * 4 * 8 * 3, dtype=torch.float32).view(2, 4, 8, 3)
    n = port.NODE_CLASS_MAPPINGS["MpiPanoRollHorizontal"]()
    once = n.run(img, 0.5)[0]
    assert not torch.equal(once, img)
    assert torch.equal(n.run(once, 0.5)[0], img)  # its own inverse
    if orig:
        o = orig.NODE_CLASS_MAPPINGS["MickmumpitzPanoRollHorizontal"]()
        for f in (0.5, 0.25, -0.3, 0.0):
            assert torch.equal(n.run(img, f)[0], o.run(img, f)[0])


def test_seam_mask():
    n = port.NODE_CLASS_MAPPINGS["MpiPanoSeamMask"]()
    m = n.run(256, 32, 96, 24)[0]
    assert m.shape == (1, 32, 256)
    assert m[0, 0, 128] == 1.0 and m[0, 0, 0] == 0.0
    assert m[0, 0, 80] == 0.0 or m[0, 0, 80] > 0  # strip starts at x=80
    if orig:
        o = orig.NODE_CLASS_MAPPINGS["MickmumpitzPanoSeamMask"]()
        for args in ((256, 32, 96, 24), (512, 16, 160, 32), (256, 8, 16, 0), (256, 8, 16, 40)):
            assert torch.equal(n.run(*args)[0], o.run(*args)[0])


def test_harmonize():
    torch.manual_seed(1)
    H, W = 32, 64
    plate = torch.rand(1, H, W, 3)
    gen = (plate * 0.7 + 0.2).clamp(0, 1)
    mask = torch.zeros(1, H, W)
    mask[:, :, 32:] = 1.0  # right half = hole
    n = port.NODE_CLASS_MAPPINGS["MpiPanoHarmonizeBoundary"]()
    kw = dict(strength=1, solve_scale=2, iterations=60, wrap_horizontal=True)
    out = n.run(gen, plate, mask, **kw)
    assert out[0].shape == gen.shape
    # the correction pulls the known side back onto the plate
    assert (out[0][:, :, :32] - plate[:, :, :32]).abs().mean() < (gen[:, :, :32] - plate[:, :, :32]).abs().mean()
    if orig:
        o = orig.NODE_CLASS_MAPPINGS["MickmumpitzPanoHarmonizeBoundary"]()
        _same(out, o.run(gen, plate, mask, **kw))
        kw["wrap_horizontal"] = False
        _same(n.run(gen, plate, mask, **kw), o.run(gen, plate, mask, **kw))


class _FakeVae:
    def encode(self, x):
        return x.mean(dim=-1, keepdim=True)


def test_krea2_reference():
    try:
        import comfy.utils  # noqa: F401
        import node_helpers  # noqa: F401
    except Exception as e:  # ComfyUI not on disk: the code is a verbatim port, nothing to run
        print("skip krea2 (no ComfyUI):", e)
        return
    torch.manual_seed(2)
    img = torch.rand(1, 40, 72, 3)
    cond = [[torch.zeros(1, 2, 4), {"pooled_output": None}]]
    n = port.NODE_CLASS_MAPPINGS["MpiPanoKrea2Reference"]()
    for mp in (8, 0.002):  # uncapped, and capped (resize path)
        out = n.run(cond, _FakeVae(), img, max_megapixels=mp)[0]
        ref = out[0][1]["reference_latents"][0]
        assert ref.shape[1] % 1 == 0 and ref.ndim == 4
        if orig:
            o = orig.NODE_CLASS_MAPPINGS["MickmumpitzPanoKrea2Reference"]()
            o_out = o.run(cond, _FakeVae(), img, max_megapixels=mp)[0]
            assert torch.equal(ref, o_out[0][1]["reference_latents"][0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all pano tests passed" + ("" if orig else " (original pack absent: equivalence skipped)"))
