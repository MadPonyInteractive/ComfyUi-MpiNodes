"""CPU checks for kj/ image, batch and mask nodes (MPI-1341), tiny tensors, no weights. Where the
original comfyui-kjnodes pack (and color-matcher / kornia) is on disk each port is compared with
the original node on fixed inputs. Run with:

    G:/ComfyUi/python_embeded/python.exe tests/test_kj_image.py
"""
import importlib
import importlib.util
import itertools
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "G:/ComfyUi/ComfyUI"
ORIG = COMFY + "/custom_nodes/comfyui-kjnodes"
sys.path.insert(0, COMFY)

# CPU for everything, whatever the box has
import comfy.model_management as mm  # noqa: E402
mm.get_torch_device = lambda: torch.device("cpu")


def _load_pkg(name, path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


port = _load_pkg("mpi_kj_port", os.path.join(ROOT, "kj"))
img = sys.modules["mpi_kj_port.image"]

orig_img = orig_mask = None
if os.path.isdir(ORIG):
    stub = types.ModuleType("kjorig")  # the pack's __init__ loads every node; the two modules we need do not
    stub.__path__ = [ORIG]
    sys.modules["kjorig"] = stub
    def _load_orig(modname):
        spec = importlib.util.spec_from_file_location("kjorig.nodes." + modname, f"{ORIG}/nodes/{modname}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    orig_img = _load_orig("image_nodes")
    orig_mask = _load_orig("mask_nodes")


def rnd(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g)


def same(a, b, what=""):
    assert a.shape == b.shape and a.dtype == b.dtype, (what, a.shape, b.shape, a.dtype, b.dtype)
    assert torch.equal(a, b), (what, float((a - b).abs().max()))


def test_mapping():
    expected = {"MpiColorMatch", "MpiImageRange", "MpiImageBatchExtend", "MpiImageBatchMulti", "MpiImageConcat",
                "MpiImageResize", "MpiGrowMaskBlur", "MpiLTX2NAG", "MpiLTXAudioVideoMask", "MpiVAELoader",
                "MpiLoraExtract"}
    assert set(port.NODE_CLASS_MAPPINGS) == expected
    assert set(port.NODE_DISPLAY_NAME_MAPPINGS) == expected
    assert all(v.startswith("Mpi ") for v in port.NODE_DISPLAY_NAME_MAPPINGS.values())
    for n, cls in port.NODE_CLASS_MAPPINGS.items():
        assert hasattr(cls, "FUNCTION") and hasattr(cls, "RETURN_TYPES") and hasattr(cls, "CATEGORY"), n


def _inputs(cls):
    d = cls.INPUT_TYPES()
    return {k: v for sect in ("required", "optional") for k, v in d.get(sect, {}).items()}


def test_inputs_match_original():
    if not orig_img:
        return
    pairs = [
        (img.MpiColorMatch, orig_img.ColorMatch, {"method"}),
        (img.MpiImageRange, orig_img.GetImageRangeFromBatch, set()),
        (img.MpiImageBatchExtend, orig_img.ImageBatchExtendWithOverlap, {"overlap_mode"}),
        (img.MpiImageBatchMulti, orig_img.ImageBatchMulti, set()),
        (img.MpiImageResize, orig_img.ImageResizeKJv2, {"upscale_method", "keep_proportion"}),
        (img.MpiGrowMaskBlur, orig_mask.GrowMaskWithBlur, set()),
    ]
    for mine, theirs, narrowed in pairs:
        a, b = _inputs(mine), _inputs(theirs)
        extra = {"unique_id"} if mine is img.MpiImageResize else set()  # hidden input in the original
        assert set(a) == set(b) - extra, (mine.__name__, set(a) ^ set(b))
        for k in a:
            if k in narrowed:
                # a narrowed list may only drop values the workflows never set
                assert set(a[k][0]) <= set(b[k][0]), (mine.__name__, k)
            else:
                assert a[k] == b[k] or (k == "keep_proportion"), (mine.__name__, k, a[k], b[k])
    # the values the workflows set are all accepted
    assert "mkl" in _inputs(img.MpiColorMatch)["method"][0]
    ks = _inputs(img.MpiImageResize)
    for v in ("crop", "pad", "resize"):
        assert v in ks["keep_proportion"][0]
    assert "lanczos" in ks["upscale_method"][0]
    assert _inputs(img.MpiImageBatchExtend)["overlap_mode"][0].count("linear_blend") == 1
    assert _inputs(img.MpiImageConcat)["direction"][0] == ["right", "down", "left", "up"]
    # the concat original is a v3 node: compare its option list with its schema
    sch = orig_img.ImageConcanate.define_schema()
    names = [i.id for i in sch.inputs]
    assert names == ["image1", "image2", "direction", "match_image_size"]


def test_color_match():
    ref, tgt = rnd(1, 16, 24, 3, seed=1), rnd(3, 16, 24, 3, seed=2) * 0.5
    node = img.MpiColorMatch()
    out = node.colormatch(ref, tgt, "mkl")[0]
    assert out.shape == tgt.shape and out.dtype == torch.float32 and 0 <= out.min() and out.max() <= 1
    assert not torch.allclose(out, tgt)
    # strength 0 hands the target back untouched
    assert node.colormatch(ref, tgt, "mkl", strength=0)[0] is tgt
    # the matched result is closer to the reference mean than the target was
    assert (out.mean((0, 1, 2)) - ref.mean((0, 1, 2))).abs().sum() < (tgt.mean((0, 1, 2)) - ref.mean((0, 1, 2))).abs().sum()
    # alpha on one side only is dropped from it, as color-matcher does
    a4 = node.colormatch(torch.cat([ref, torch.ones(1, 16, 24, 1)], -1), tgt, "mkl")[0]
    assert a4.shape == tgt.shape
    if orig_img:
        o = orig_img.ColorMatch()
        refs3 = rnd(3, 16, 24, 3, seed=3)
        for ref_in, strength, mt in itertools.product((ref, refs3), (1.0, 0.4, 2.0), (True, False)):
            theirs = o.colormatch(ref_in, tgt, "mkl", strength=strength, multithread=mt)[0]
            same(node.colormatch(ref_in, tgt, "mkl", strength=strength, multithread=mt)[0], theirs, f"{strength} {mt}")
        # a single frame target (the original squeezes the batch away)
        same(node.colormatch(ref, tgt[:1], "mkl")[0], o.colormatch(ref, tgt[:1], "mkl")[0], "single")


def test_image_range():
    node, images, masks = img.MpiImageRange(), rnd(10, 4, 4, 3), rnd(10, 4, 4, seed=5)
    assert node.imagesfrombatch(-1, 3, images=images)[0].shape[0] == 3
    assert torch.equal(node.imagesfrombatch(-1, 3, images=images)[0], images[7:])
    assert torch.equal(node.imagesfrombatch(0, 4096, images=images)[0], images)  # h3_extend asks for 4096
    assert node.imagesfrombatch(0, 3, images=images)[1] is None
    for bad in (10, 99):
        try:
            node.imagesfrombatch(bad, 2, images=images)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    if orig_img:
        o = orig_img.GetImageRangeFromBatch()
        for start, n in ((0, 1), (0, 73), (3, 4), (9, 5), (-1, 1), (-1, 4096), (39 % 10, 4096)):
            a, b = node.imagesfrombatch(start, n, images=images, masks=masks), o.imagesfrombatch(start, n, images=images, masks=masks)
            same(a[0], b[0], f"img {start} {n}")
            same(a[1], b[1], f"mask {start} {n}")


def test_batch_extend():
    node = img.MpiImageBatchExtend()
    src, new = rnd(9, 8, 8, 3, seed=1), rnd(7, 8, 8, 3, seed=2)
    out = node.imagesfrombatch(src, 3, "source", "linear_blend", new_images=new)
    assert out[2].shape[0] == 9 - 3 + 7 and torch.equal(out[1], src[-3:])
    # no new images: a 1x64x64x3 placeholder
    assert node.imagesfrombatch(src, 3, "source", "linear_blend")[2].shape == (1, 64, 64, 3)
    # overlap longer than the source hands the source back three times
    assert node.imagesfrombatch(src, 12, "source", "cut", new_images=new)[2] is src
    try:
        node.imagesfrombatch(src, 3, "source", "cut", new_images=rnd(7, 4, 4, 3))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    if orig_img:
        o = orig_img.ImageBatchExtendWithOverlap()
        for side, mode, ov in itertools.product(("source", "new_images"), ("cut", "linear_blend", "ease_in_out", "filmic_crossfade"), (1, 3, 7)):
            a = node.imagesfrombatch(src, ov, side, mode, new_images=new)
            b = o.imagesfrombatch(src, ov, side, mode, new_images=new)
            for i in range(3):
                same(a[i], b[i], f"{side} {mode} {ov} out{i}")
        a, b = node.imagesfrombatch(src, 4, "source", "linear_blend"), o.imagesfrombatch(src, 4, "source", "linear_blend")
        for i in range(3):
            same(a[i], b[i], "no new")


def test_batch_multi():
    node = img.MpiImageBatchMulti()
    a, b = rnd(2, 6, 8, 3, seed=1), rnd(3, 6, 8, 3, seed=2)
    assert node.combine(2, image_1=a, image_2=b)[0].shape == (5, 6, 8, 3)
    # a missing input is a black stand-in of the first image's length
    out = node.combine(3, image_1=a, image_2=b)[0]
    assert out.shape == (2 + 3 + 2, 6, 8, 3) and float(out[5:].abs().max()) == 0
    if orig_img:
        o = orig_img.ImageBatchMulti()
        cases = [
            dict(inputcount=2, image_1=a, image_2=b),
            dict(inputcount=3, image_1=a, image_2=b),
            dict(inputcount=3, image_1=a, image_2=rnd(1, 3, 4, 4, seed=3), image_3=b),  # resized + padded alpha
            dict(inputcount=2, image_1=a),
        ]
        for kw in cases:
            same(node.combine(**kw)[0], o.combine(**kw)[0], str(sorted(kw)))


def test_concat():
    node = img.MpiImageConcat()
    a, b = rnd(2, 6, 8, 3, seed=1), rnd(2, 10, 4, 3, seed=2)
    assert node.execute(a, b, "right", True)[0].shape == (2, 6, 8 + int(6 * 4 / 10), 3)
    assert node.execute(a, b, "down", True)[0].shape == (2, 6 + int(8 / (4 / 10)), 8, 3)
    assert node.execute(a, b, "right", False)[0].shape == (2, 10, 12, 3)
    if orig_img:
        o = orig_img.ImageConcanate
        cases = [(a, b), (a, rnd(1, 5, 9, 3, seed=3)), (rnd(1, 5, 9, 3, seed=4), a), (a, rnd(2, 6, 8, 4, seed=5))]
        for (x, y), d, m in itertools.product(cases, ("right", "down", "left", "up"), (True, False)):
            same(node.execute(x, y, d, m)[0], o.concatenate(x, y, d, m), f"{tuple(x.shape)} {tuple(y.shape)} {d} {m}")


def test_resize():
    node = img.MpiImageResize()
    im = rnd(2, 30, 50, 3, seed=1)
    # the dims the workflows ask for
    out, w, h, m = node.resize(im, 32, 32, "lanczos", "crop", "0, 0, 0", "center", 16)
    assert (out.shape, w, h) == ((2, 32, 32, 3), 32, 32) and m.shape == (1, 64, 64)
    out, w, h, m = node.resize(im, 64, 64, "lanczos", "pad", "0, 0, 0", "center", 16)
    assert out.shape == (2, 64, 64, 3) and m.shape == (2, 64, 64) and float(m[:, 20:40, 10:50].max()) == 0 and float(m[:, 0, 0].min()) == 1
    # a pad that leaves the size indivisible grows the pad to a multiple (the 2px-divisible resize flows)
    out, w, h, m = node.resize(im, 50, 51, "lanczos", "pad", "255, 105, 180", "center", 2)
    assert out.shape[1] % 2 == 0 and out.shape[2] % 2 == 0
    # string colours
    assert img.string_to_color("255, 105, 180") == [255, 105, 180]
    assert img.string_to_color("1.0, 0.5, 0.0") == [255, 127, 0]
    assert img.string_to_color("#ff0000") == [255, 0, 0]
    assert img.string_to_color("red") == [255, 0, 0]
    assert img.string_to_color("0.5") == [127, 127, 127]
    try:
        node.resize(im, 32, 32, "lanczos", "stretch", "0, 0, 0", "center", 2, device="gpu")
        raise AssertionError("expected ValueError for lanczos on gpu")
    except ValueError:
        pass
    if orig_img:
        o = orig_img.ImageResizeKJv2()
        mask = (rnd(2, 30, 50, seed=7) > 0.5).float()
        modes = ["stretch", "resize", "pad", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur", "total_pixels"]
        n = 0
        for mode, pos, up in itertools.product(modes, ("center", "top", "bottom", "left", "right"), ("lanczos", "bilinear")):
            for (w_, h_, div) in ((64, 40, 16), (24, 48, 2), (0, 40, 8), (64, 0, 8), (0, 0, 1)):
                kw = dict(width=w_, height=h_, keep_proportion=mode, upscale_method=up, divisible_by=div,
                          pad_color="10, 200, 30", crop_position=pos)
                for with_mask in (False, True):
                    try:
                        b = o.resize(im, mask=mask if with_mask else None, unique_id=None, **kw)
                    except Exception as e:  # e.g. total_pixels with a 0 size: the port must fail the same way
                        try:
                            node.resize(im, mask=mask if with_mask else None, **kw)
                            raise AssertionError(f"original raised {e!r}, port did not: {kw}")
                        except type(e):
                            continue
                    a = node.resize(im, mask=mask if with_mask else None, **kw)
                    for i in range(4):
                        if isinstance(a[i], int):
                            assert a[i] == b[i], (kw, i)
                        else:
                            same(a[i], b[i], f"{kw} mask={with_mask} out{i}")
                    n += 1
        for up in ("nearest-exact", "area", "bicubic"):
            kw = dict(width=33, height=21, keep_proportion="stretch", upscale_method=up, divisible_by=3,
                      pad_color="0, 0, 0", crop_position="center")
            for i, (x, y) in enumerate(zip(node.resize(im, mask=mask, **kw), o.resize(im, mask=mask, unique_id=None, **kw))):
                if not isinstance(x, int):
                    same(x, y, f"{up} out{i}")
        # the 64x64 placeholder mask means "no mask"
        ph = torch.zeros(1, 64, 64)
        kw = dict(width=32, height=32, upscale_method="lanczos", keep_proportion="pad", pad_color="0, 0, 0",
                  crop_position="center", divisible_by=2, mask=ph)
        same(node.resize(im, **kw)[3], o.resize(im, unique_id=None, **kw)[3], "placeholder")
        print(f"   resize: {n} combinations equal to the original")


def _blobs(n=3, h=24, w=32, seed=9):
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(n, h, w, generator=g) > 0.985).float()
    m = torch.nn.functional.max_pool2d(m.unsqueeze(1), 5, 1, 2).squeeze(1)  # blobs
    m[:, 10:14, 12:20] = 1.0
    m[:, 11:13, 14:16] = 0.0  # a hole
    return m * (0.5 + 0.5 * torch.rand(n, h, w, generator=g))


def test_grow_mask():
    node = img.MpiGrowMaskBlur()
    m = _blobs()
    grown, inv = node.expand_mask(m, 3, True, False, 0.0, 0.0, 1.0, 1.0)
    assert grown.shape == m.shape and torch.equal(inv, 1.0 - grown) and grown.sum() > m.sum()
    shrunk = node.expand_mask(m, -2, False, False, 0.0, 0.0, 1.0, 1.0)[0]
    assert shrunk.sum() < m.sum()
    if orig_mask:
        o = orig_mask.GrowMaskWithBlur()
        n = 0
        # head swap: (12, 0, tapered, no flip, blur 12, 1, 1, no fill); img_auto_mask: (+-4, 0, ..., blur 0, fill holes)
        for expand, inc, tap, flip, blur, lerp, decay, fill in [
            (12, 0, True, False, 12, 1, 1, False),
            (4, 0, True, False, 0, 1, 1, True),
            (-4, 0, True, False, 0, 1, 1, False),
            (3, 0, False, False, 0, 1, 1, False),
            (2, 1.5, True, False, 0, 1, 1, False),
            (-2, 0.7, False, False, 0, 1, 1, False),
            (0, 0, True, True, 0, 1, 1, False),
            (2, 0, True, False, 2.5, 0.6, 1, False),
            (2, 0, True, False, 0, 1, 0.5, False),
            (0, 0, True, False, 3.0, 0.5, 0.8, True),
        ]:
            args = (m, expand, tap, flip, blur, inc, lerp, decay, fill)
            a, b = node.expand_mask(*args), o.expand_mask(*args)
            same(a[0], b[0], f"mask {args[1:]}")
            same(a[1], b[1], f"inverted {args[1:]}")
            n += 1
        # a single 2-D-batch mask and an all-zero mask
        same(node.expand_mask(m[:1], 2, True, False, 0, 0, 1, 1, False)[0], o.expand_mask(m[:1], 2, True, False, 0, 0, 1, 1, False)[0], "single")
        same(node.expand_mask(torch.zeros(2, 8, 8), 3, True, False, 0, 0, 1, 1, False)[0],
             o.expand_mask(torch.zeros(2, 8, 8), 3, True, False, 0, 0, 1, 1, False)[0], "zeros")
        print(f"   grow mask: {n} parameter sets equal to the original")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all kj image tests passed" + ("" if orig_img else " (original pack absent: equivalence skipped)"))
