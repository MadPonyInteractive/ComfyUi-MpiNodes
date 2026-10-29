"""CPU checks for cropstitch/ and krea2edit/ (MPI-1343), tiny tensors, no weights. Where the pinned
originals (comfyui-inpaint-cropandstitch, comfyui-krea2edit, ComfyUI-Krea2-Ostris-Edit) are on disk,
each port's code is proven equal to the original's apart from the renames, and the crop/stitch pair is
run against the original on the settings Cubric Studio's workflows use. Run with:

    G:/ComfyUi/python_embeded/python.exe tests/test_mpi1343_small.py
"""
import importlib.util
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "G:/ComfyUi/ComfyUI"
ORIG = "C:/AI/Mpi/Cubric-Vision/engine/ComfyUI_windows_portable/ComfyUI/custom_nodes"
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


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


path_before = list(sys.path)
crop = _load_pkg("mpi_cropstitch_port", os.path.join(ROOT, "cropstitch"))
k2 = _load_pkg("mpi_krea2edit_port", os.path.join(ROOT, "krea2edit"))
assert sys.path == path_before

PAIRS = {
    "cropstitch/nodes.py": ("comfyui-inpaint-cropandstitch/inpaint_cropandstitch.py",
                            [("InpaintCropImproved", "MpiInpaintCrop"), ("InpaintStitchImproved", "MpiInpaintStitch")]),
    "krea2edit/edit.py": ("comfyui-krea2edit/__init__.py",
                          [("Krea2EditModelPatch", "MpiKrea2EditModelPatch"),
                           ("Krea2EditGroundedEncode", "MpiKrea2EditGroundedEncode"),
                           ('"krea2_edit", wrapper', '"mpi_krea2_edit", wrapper'),
                           ('CATEGORY = "krea2edit"', 'CATEGORY = "MpiNodes/Krea2"')]),
    "krea2edit/ostris.py": ("ComfyUI-Krea2-Ostris-Edit/nodes.py",
                            [("TextEncodeKrea2OstrisEdit", "MpiKrea2OstrisEditTextEncode"),
                             ("Krea2OstrisEditModelPatch", "MpiKrea2OstrisEditModelPatch")]),
}


def _code(text):
    """The node code: everything before NODE_CLASS_MAPPINGS, minus comment-only lines and the docstring
    header the port adds."""
    text = text[:text.index("NODE_CLASS_MAPPINGS = {")]
    lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    return "\n".join(lines)


def test_code_is_the_original():
    for port_rel, (orig_rel, renames) in PAIRS.items():
        orig_path = os.path.join(ORIG, orig_rel)
        if not os.path.isfile(orig_path):
            print("  skip (original not on disk):", orig_rel)
            continue
        with open(orig_path, encoding="utf-8") as f:
            orig = f.read()
        for old, new in renames:
            orig = re.sub(r"\b%s\b" % re.escape(old) if old.isidentifier() else re.escape(old), new, orig)
        with open(os.path.join(ROOT, port_rel), encoding="utf-8") as f:
            port = f.read()
        if port_rel == "cropstitch/nodes.py":  # the port adds a module docstring
            port = port[port.index('"""', 3) + 3:]
        assert _code(port) == _code(orig), port_rel


def test_mapping():
    assert set(crop.NODE_CLASS_MAPPINGS) == {"MpiInpaintCrop", "MpiInpaintStitch"}
    assert set(k2.NODE_CLASS_MAPPINGS) == {"MpiKrea2EditModelPatch", "MpiKrea2EditGroundedEncode",
                                           "MpiKrea2OstrisEditTextEncode", "MpiKrea2OstrisEditModelPatch"}
    for m in (crop, k2):
        assert set(m.NODE_DISPLAY_NAME_MAPPINGS) == set(m.NODE_CLASS_MAPPINGS)
        assert all(v.startswith("Mpi ") for v in m.NODE_DISPLAY_NAME_MAPPINGS.values())


# The settings the shipped workflows use (comfy_workflows/*.json, 2026-10-29).
BASE = {"downscale_algorithm": "bilinear", "upscale_algorithm": "bicubic", "preresize": False,
        "preresize_mode": "ensure minimum resolution", "preresize_min_width": 1024, "preresize_min_height": 1024,
        "preresize_max_width": 16384, "preresize_max_height": 16384, "mask_fill_holes": False,
        "mask_expand_pixels": 6, "mask_invert": False, "mask_blend_pixels": 32, "mask_hipass_filter": 0.1,
        "extend_for_outpainting": False, "extend_up_factor": 1, "extend_down_factor": 1,
        "extend_left_factor": 1, "extend_right_factor": 1, "context_from_mask_extend_factor": 1,
        "output_resize_to_target_size": True, "output_target_width": 1024, "output_target_height": 1024,
        "output_padding": "32"}
CASES = [dict(BASE, device_mode="gpu (much faster)"),
         dict(BASE, device_mode="gpu (much faster)", mask_expand_pixels=0, mask_blend_pixels=16, output_padding="16"),
         dict(BASE, device_mode="cpu (compatible)", context_from_mask_extend_factor=1.25, mask_expand_pixels=0,
              mask_blend_pixels=24)]


def _fill(cls, kw):
    """Every required/optional widget the case does not name gets the node's own default."""
    d = cls.INPUT_TYPES()
    out = dict(kw)
    for sect in ("required", "optional"):
        for k, v in d.get(sect, {}).items():
            if k in out or not isinstance(v, tuple) or len(v) < 2 or not isinstance(v[1], dict):
                continue
            if "default" in v[1]:
                out[k] = v[1]["default"]
            elif isinstance(v[0], list):
                out[k] = v[0][0]
    return out


def test_crop_stitch_match_original():
    orig_path = os.path.join(ORIG, "comfyui-inpaint-cropandstitch/inpaint_cropandstitch.py")
    if not os.path.isfile(orig_path):
        print("  skip (original not on disk)")
        return
    orig = _load_file("orig_cropstitch", orig_path)
    assert crop.NODE_CLASS_MAPPINGS["MpiInpaintCrop"].INPUT_TYPES() == orig.InpaintCropImproved.INPUT_TYPES()
    assert crop.NODE_CLASS_MAPPINGS["MpiInpaintStitch"].INPUT_TYPES() == orig.InpaintStitchImproved.INPUT_TYPES()
    g = torch.Generator().manual_seed(0)
    image = torch.rand(1, 200, 300, 3, generator=g)
    mask = torch.zeros(1, 200, 300)
    mask[:, 60:140, 90:200] = 1.0
    P, O = crop.NODE_CLASS_MAPPINGS["MpiInpaintCrop"](), orig.InpaintCropImproved()
    fn = P.FUNCTION
    for case in CASES:
        kw = _fill(orig.InpaintCropImproved, dict(case, image=image, mask=mask))
        a, b = getattr(P, fn)(**kw), getattr(O, fn)(**kw)
        assert len(a) == len(b)
        for x, y in zip(a[1:], b[1:]):
            assert torch.equal(x, y), case["device_mode"]
        sp =crop.NODE_CLASS_MAPPINGS["MpiInpaintStitch"]()
        so = orig.InpaintStitchImproved()
        inpainted = torch.rand(a[1].shape, generator=torch.Generator().manual_seed(1))
        sa = getattr(sp, sp.FUNCTION)(stitcher=a[0], inpainted_image=inpainted)[0]
        sb = getattr(so, so.FUNCTION)(stitcher=b[0], inpainted_image=inpainted)[0]
        assert torch.equal(sa, sb) and sa.shape == image.shape, case["device_mode"]


def test_krea2_inputs_match_original():
    for rel, pairs in (("comfyui-krea2edit/__init__.py", [("Krea2EditModelPatch", "MpiKrea2EditModelPatch"),
                                                          ("Krea2EditGroundedEncode", "MpiKrea2EditGroundedEncode")]),
                       ("ComfyUI-Krea2-Ostris-Edit/nodes.py", [("TextEncodeKrea2OstrisEdit", "MpiKrea2OstrisEditTextEncode"),
                                                               ("Krea2OstrisEditModelPatch", "MpiKrea2OstrisEditModelPatch")])):
        p = os.path.join(ORIG, rel)
        if not os.path.isfile(p):
            print("  skip (original not on disk):", rel)
            continue
        o = _load_file("orig_" + rel.split("/")[0].replace("-", "_"), p)
        for old, new in pairs:
            assert getattr(o, old).INPUT_TYPES() == k2.NODE_CLASS_MAPPINGS[new].INPUT_TYPES(), new
            assert getattr(o, old).RETURN_TYPES == k2.NODE_CLASS_MAPPINGS[new].RETURN_TYPES, new


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
