"""The port in detailer/ against the ORIGINAL Impact-Pack / Impact-Subpack code, on fixed tiny inputs.

The originals import ComfyUI at module top, so this file loads their source files with stub modules for
comfy / nodes / folder_paths / server (a stub answers any attribute with a MagicMock) and calls the real
pure functions. Skipped (prints "skip") when the G:/ComfyUi bench is not on this machine. Run with the
engine's python (it has torch, numpy, cv2, torchvision):

    G:/ComfyUi/python_embeded/python.exe tests/test_detailer_ref.py
"""
import importlib.util
import os
import sys
import types
from unittest import mock

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMPACT_DIR = "G:/ComfyUi/ComfyUI/custom_nodes/comfyui-impact-pack/modules/impact"
SUBPACK_DIR = "G:/ComfyUi/ComfyUI/custom_nodes/ComfyUI-Impact-Subpack/modules"


class _Stub(types.ModuleType):
    """A module whose every public attribute is a MagicMock."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = mock.MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


def _stub(name):
    module = _Stub(name)
    module.__path__ = []
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    sys.modules[name] = module
    return module


def _load(package, directory, module):
    spec = importlib.util.spec_from_file_location(f"{package}.{module}", os.path.join(directory, f"{module}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{package}.{module}"] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules[package], module, mod)
    return mod


_REF = None


def reference():
    """-> a namespace with .utils .core .segs_nodes .wildcards .pack_nodes .subcore, or None without the bench."""
    global _REF
    if _REF is not None:
        return _REF or None
    if not (os.path.isdir(IMPACT_DIR) and os.path.isdir(SUBPACK_DIR)):
        _REF = False
        return None

    for name in ("folder_paths", "nodes", "server", "comfy", "comfy.model_management", "comfy.samplers", "comfy.sd",
                 "comfy.cli_args", "comfy.k_diffusion", "comfy.k_diffusion.sampling", "comfy.utils", "comfy.sample",
                 "comfy_extras", "comfy_extras.nodes_custom_sampler", "comfy_extras.nodes_differential_diffusion",
                 "latent_preview", "node_helpers", "segment_anything", "piexif", "impact.impact_server", "impact.defs",
                 "impact.segs_upscaler", "impact.hooks", "cm_global", "sam2"):
        _stub(name)
    # the Subpack's subcore.py imports ultralytics / dill / numpy.core at its top: the real ones when they
    # import (a real YOLO model keeps working), stubs when they do not
    try:
        import dill._dill  # noqa: F401
        import ultralytics  # noqa: F401
        import ultralytics.nn.modules.block  # noqa: F401
        import ultralytics.utils.loss  # noqa: F401
        import ultralytics.utils.tal  # noqa: F401
        from numpy.core.multiarray import scalar  # noqa: F401
    except ImportError:
        for name in ("dill", "dill._dill", "numpy.core.multiarray", "ultralytics", "ultralytics.nn", "ultralytics.nn.tasks",
                     "ultralytics.utils", "ultralytics.utils.tal", "ultralytics.utils.loss", "ultralytics.nn.modules",
                     "ultralytics.nn.modules.block"):
            _stub(name)
    sys.modules["sam2"] = None  # core.py: find_spec("sam2") -> None, SAM 2 left out
    sys.modules["comfy"].model_management.get_torch_device.return_value = torch.device("cpu")
    sys.modules["folder_paths"].get_user_directory.return_value = ""

    impact = types.ModuleType("impact")
    impact.__path__ = [IMPACT_DIR]
    sys.modules["impact"] = impact
    ns = types.SimpleNamespace()
    # use_gpu_opencv() reads the ini; the answer only picks UMat vs plain arrays
    ns.config = _load("impact", IMPACT_DIR, "config")
    ns.config.get_config = lambda: {"disable_gpu_opencv": True}
    ns.utils = _load("impact", IMPACT_DIR, "utils")
    ns.wildcards = _load("impact", IMPACT_DIR, "wildcards")
    ns.sampling = _load("impact", IMPACT_DIR, "impact_sampling")
    ns.core = _load("impact", IMPACT_DIR, "core")
    ns.segs_nodes = _load("impact", IMPACT_DIR, "segs_nodes")
    ns.pack = _load("impact", IMPACT_DIR, "impact_pack")
    _REF = ns
    return ns


def subpack_core():
    """The Subpack's subcore.py (imports ultralytics / dill at module top; the stubs let it load)."""
    reference()
    if "subpack.subcore" in sys.modules:
        return sys.modules["subpack.subcore"]
    sub = types.ModuleType("subpack")
    sub.__path__ = [SUBPACK_DIR]
    sys.modules["subpack"] = sub
    _load("subpack", SUBPACK_DIR, "utils")
    saved_load = torch.load
    subcore = _load("subpack", SUBPACK_DIR, "subcore")
    torch.load = saved_load  # subcore swaps torch.load for its own wrapper at import
    return subcore


# --- helpers ----------------------------------------------------------------------------------

def _rect_mask(h, w, boxes, seed=None):
    m = np.zeros((h, w), np.float32)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2, x1:x2] = 1.0
    if seed is not None:
        m = m * (0.5 + 0.5 * np.random.default_rng(seed).random((h, w)).astype(np.float32))
    return m


def _same_segs(a, b):
    assert tuple(a[0]) == tuple(b[0]), (a[0], b[0])
    assert len(a[1]) == len(b[1]), (len(a[1]), len(b[1]))
    for x, y in zip(a[1], b[1]):
        assert list(x.crop_region) == list(y.crop_region), (x.crop_region, y.crop_region)
        assert [int(v) for v in x.bbox] == [int(v) for v in y.bbox], (x.bbox, y.bbox)
        assert x.label == y.label
        assert np.array_equal(np.asarray(x.cropped_mask), np.asarray(y.cropped_mask))
        assert (x.cropped_image is None) == (y.cropped_image is None)


# --- tests ------------------------------------------------------------------------------------

def test_crop_region_and_morphology_match_impact():
    ref = reference()
    if ref is None:
        return print("skip (no G:/ComfyUi bench)")
    from detailer import segs as S

    for w, h, bbox, cf, mn in [(100, 80, (10, 10, 30, 40), 1.5, None), (100, 80, (0, 0, 100, 80), 3, None),
                               (100, 80, (90, 70, 99, 79), 3, None), (64, 64, (5.5, 7.25, 20.5, 33.75), 2.2, None),
                               (100, 80, (40, 30, 44, 34), 1.0, 20)]:
        assert S.make_crop_region(w, h, bbox, cf, mn) == ref.utils.make_crop_region(w, h, bbox, cf, mn)

    m = _rect_mask(40, 50, [(10, 8, 30, 25)], seed=1)
    for d in (-5, -1, 0, 1, 4, 11):
        assert np.array_equal(S.dilate_mask(m, d), ref.utils.dilate_mask(m, d)), d
        t = torch.from_numpy(m)[None]
        assert np.array_equal(S.dilate_mask(t.numpy(), d), ref.utils.dilate_mask(t.numpy(), d)), d
    sm = [((1, 2, 3, 4), m, 0.9), ((1, 2, 3, 4), m.T.copy(), 0.8)]
    for d in (-3, 0, 6):
        for x, y in zip(S.dilate_masks(sm, d), ref.utils.dilate_masks(sm, d)):
            assert np.array_equal(x[1], y[1])


def test_tensor_ops_match_impact():
    ref = reference()
    if ref is None:
        return print("skip (no G:/ComfyUi bench)")
    from detailer import segs as S

    torch.manual_seed(0)
    img = torch.rand(1, 24, 30, 3)
    for w, h in [(60, 48), (15, 12), (30, 24)]:
        assert torch.equal(S.tensor_resize(img, w, h), ref.utils.tensor_resize(img, w, h))
    gray = torch.rand(1, 24, 30, 1)
    assert torch.equal(S.tensor_resize(gray, 12, 10), ref.utils.tensor_resize(gray, 12, 10))

    for kernel in (0, 2, 5, 40):
        m = torch.rand(20, 25)
        assert torch.equal(S.tensor_gaussian_blur_mask(m, kernel), ref.utils.tensor_gaussian_blur_mask(m, kernel))
        assert torch.equal(S.tensor_gaussian_blur_mask(m.numpy(), kernel), ref.utils.tensor_gaussian_blur_mask(m.numpy(), kernel))

    for c1, c2 in [(3, 3), (4, 4), (4, 3), (3, 4)]:
        for pos in [(2, 3), (-0, 0), (20, 15), (28, 22), (40, 40)]:
            a, b = torch.rand(1, 24, 30, c1), torch.rand(1, 10, 12, c2)
            a2, b2 = a.clone(), b.clone()
            mask = torch.rand(1, 10, 12, 1)
            S.tensor_paste(a, b, pos, mask)
            ref.utils.tensor_paste(a2, b2, pos, mask)
            assert torch.equal(a, a2), (c1, c2, pos)
    # mask of another size is resized first
    a, b = torch.rand(1, 24, 30, 3), torch.rand(1, 10, 12, 3)
    a2 = a.clone()
    mask = torch.rand(1, 5, 6, 1)
    S.tensor_paste(a, b, (3, 3), mask)
    ref.utils.tensor_paste(a2, b, (3, 3), mask)
    assert torch.equal(a, a2)


def test_mask_to_segs_match_impact():
    ref = reference()
    if ref is None:
        return print("skip (no G:/ComfyUi bench)")
    from detailer import segs as S

    base = _rect_mask(60, 80, [(5, 5, 20, 30), (40, 10, 70, 50)])
    base[12:20, 8:15] = 0.0  # a hole in the first blob
    base[55:58, 70:72] = 1.0  # a small blob, below a drop_size
    soft = _rect_mask(60, 80, [(5, 5, 20, 30), (40, 10, 70, 50)], seed=3)
    batch = np.stack([base, _rect_mask(60, 80, [(30, 30, 50, 55)])])
    for mask in (base, soft, batch):
        for combined in (True, False):
            for crop_factor in (1.0, 1.8, 3.0):
                for bbox_fill in (False, True):
                    for drop_size in (1, 10):
                        for is_contour in (True, False):
                            a = S.mask_to_segs(mask.copy(), combined, crop_factor, bbox_fill, drop_size, is_contour=is_contour)
                            b = ref.core.mask_to_segs(mask.copy(), combined, crop_factor, bbox_fill, drop_size, is_contour=is_contour)
                            _same_segs(a, b)
    # tensors in, as MaskToSEGS passes them
    a = S.mask_to_segs(torch.from_numpy(base), True, 3.0, False, 1, is_contour=False)
    b = ref.core.mask_to_segs(torch.from_numpy(base), True, 3.0, False, 1, is_contour=False)
    _same_segs(a, b)


def test_segs_ops_match_impact():
    ref = reference()
    if ref is None:
        return print("skip (no G:/ComfyUi bench)")
    from detailer import segs as S

    base = _rect_mask(60, 80, [(5, 5, 20, 30), (40, 10, 70, 50)], seed=2)
    segs = S.mask_to_segs(base, False, 1.8, False, 1, is_contour=False)
    rsegs = ref.core.mask_to_segs(base, False, 1.8, False, 1, is_contour=False)

    assert torch.equal(S.segs_to_combined_mask(segs), ref.core.segs_to_combined_mask(rsegs))
    for x, y in zip(S.segs_to_masklist(segs), ref.core.segs_to_masklist(rsegs)):
        assert torch.equal(x, y)
    empty = ((60, 80), [])
    assert torch.equal(S.segs_to_masklist(empty)[0], ref.core.segs_to_masklist(empty)[0])

    other = torch.from_numpy(_rect_mask(60, 80, [(0, 0, 45, 60)], seed=4))
    _same_segs(S.segs_bitwise_and_mask(segs, other), ref.core.segs_bitwise_and_mask(rsegs, other))
    _same_segs(S.segs_bitwise_and_mask(segs, other[None]), ref.core.segs_bitwise_and_mask(rsegs, other[None]))

    for f in (-3, 0, 5):
        _same_segs(S.dilate_segs(segs, f), ref.core.dilate_segs(rsegs, f))

    labels = ["a", " b "]
    _same_segs(S.label_assign(segs, labels), ref.segs_nodes.SEGSLabelAssign.assign(rsegs, labels)[0])
    assert [s.label for s in S.label_assign(segs, labels)[1]] == ["a", "b"]
    labelled = S.label_assign(segs, ["face", "hand"])
    rlabelled = ref.segs_nodes.SEGSLabelAssign.assign(rsegs, ["face", "hand"])[0]
    for want in (["all"], ["face"], ["hand", "x"], ["eyes"]):
        k1, r1 = S.label_filter(labelled, want)
        k2, r2 = ref.segs_nodes.SEGSLabelFilter.filter(rlabelled, want)
        _same_segs(k1, k2)
        _same_segs(r1, r2)

    # the same image size is a no-op, a 2x image scales regions and masks
    assert S.segs_scale_match(segs, (1, 60, 80, 3)) is segs
    _same_segs(S.segs_scale_match(segs, (1, 120, 160, 3)), ref.core.segs_scale_match(rsegs, (1, 120, 160, 3)))


def test_tile_segs_match_impact():
    ref = reference()
    if ref is None:
        return print("skip (no G:/ComfyUi bench)")
    from detailer import segs as S

    maker = ref.segs_nodes.MakeTileSEGS.doit
    cases = [((1, 96, 128, 3), 64, 1.5, 16, 0), ((1, 100, 100, 3), 64, 1.5, 20, 7), ((1, 64, 64, 3), 128, 3.0, 5, 0),
             ((1, 200, 130, 3), 80, 1.0, 40, -5), ((1, 90, 300, 3), 1024, 1.5, 205, 30)]
    for shape, bbox_size, crop_factor, overlap, fdil in cases:
        img = torch.zeros(shape)
        _same_segs(S.make_tile_segs(img, bbox_size, crop_factor, overlap, fdil), maker(img, bbox_size, crop_factor, overlap, fdil)[0])

    # filter in / out by SEGS
    img = torch.zeros((1, 100, 120, 3))
    f_in = S.mask_to_segs(_rect_mask(100, 120, [(10, 10, 90, 80)]), True, 1.0, False, 1)
    f_out = S.mask_to_segs(_rect_mask(100, 120, [(0, 0, 40, 40)]), True, 1.0, False, 1)
    rin = ref.core.mask_to_segs(_rect_mask(100, 120, [(10, 10, 90, 80)]), True, 1.0, False, 1)
    rout = ref.core.mask_to_segs(_rect_mask(100, 120, [(0, 0, 40, 40)]), True, 1.0, False, 1)
    _same_segs(S.make_tile_segs(img, 48, 1.5, 8, 6, filter_in_segs_opt=f_in, filter_out_segs_opt=f_out),
               maker(img, 48, 1.5, 8, 6, filter_in_segs_opt=rin, filter_out_segs_opt=rout)[0])

    # irregular tile masks: random, so seed both and run the original's 16 threads one after another
    class _Sync:
        def __init__(self, max_workers=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, *args):
            fn(*args)
            return mock.MagicMock()

    img = torch.zeros((1, 100, 130, 3))
    np.random.seed(5)
    a = S.make_tile_segs(img, 64, 1.5, 10, 30, 0.7, "Reuse fast")
    np.random.seed(5)
    with mock.patch.object(ref.core, "ThreadPoolExecutor", _Sync):
        b = maker(img, 64, 1.5, 10, 30, 0.7, "Reuse fast")[0]
    _same_segs(a, b)
    assert len(a[1]) > 0 and any((s.cropped_mask != 1).any() for s in a[1])  # the masks really are irregular

    # the 512 px modes take minutes in the original's per-pixel loops: check the port alone
    for mode in ("Reuse quality", "All random fast"):
        got = S.make_tile_segs(img, 64, 1.5, 10, 30, 0.7, mode)
        assert len(got[1]) > 0
        for s in got[1]:
            assert s.cropped_mask.min() >= 0 and s.cropped_mask.max() <= 1 and s.cropped_mask.max() > 0


def test_do_detail_matches_impact_with_a_fake_sampler():
    ref = reference()
    if ref is None:
        return print("skip (no G:/ComfyUi bench)")
    from detailer import detail
    from detailer import segs as S

    calls = []

    def fake_enhance(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                     sampler_name, scheduler, positive, negative, denoise, noise_mask, force_inpaint,
                     wildcard_opt=None, wildcard_opt_concat_mode=None, **kw):
        calls.append((seed, wildcard_opt, wildcard_opt_concat_mode, None if noise_mask is None else tuple(noise_mask.shape)))
        out = image.clone() * 0.5 + 0.25
        return out.cpu()

    class _Model:
        model_options = {"denoise_mask_function": 1}

    torch.manual_seed(1)
    image = torch.rand(1, 90, 120, 3)
    base = _rect_mask(90, 120, [(5, 5, 30, 30), (60, 20, 100, 70)], seed=7)
    for wildcard in ("", "[CONCAT][LAB][t0]a red door[t1]blue sky", "plain text", "[LAB][t1]only second"):
        for noise_mask in (True, False):
            segs = S.label_assign(S.mask_to_segs(base, False, 1.8, False, 1, is_contour=False), ["t0", "t1"])
            rsegs = ref.core.mask_to_segs(base, False, 1.8, False, 1, is_contour=False)
            rsegs = ref.segs_nodes.SEGSLabelAssign.assign(rsegs, ["t0", "t1"])[0]
            args = (_Model(), None, None, 1024, True, 1536, 5, 4, 1.0, "euler", "simple", [[torch.zeros(1, 2), {}]],
                    [[torch.zeros(1, 2), {}]], 0.3, 10, noise_mask, True, wildcard)
            calls.clear()
            with mock.patch.object(detail, "enhance_detail", lambda *a, wildcard_opt=None, **kw: fake_enhance(*a, wildcard_opt=wildcard_opt, **kw)):
                mine = detail.do_detail(image, segs, *args, cycle=1, noise_mask_feather=0)
            mine_calls = list(calls)
            calls.clear()
            with mock.patch.object(ref.core, "enhance_detail", lambda *a, **kw: (fake_enhance(*a, **kw), None)):
                theirs = ref.pack.DetailerForEach.do_detail(image, rsegs, *args)
            assert mine_calls == list(calls), (wildcard, noise_mask, mine_calls, calls)
            assert torch.equal(mine[0], theirs[0])
            for x, y in zip(mine[1], theirs[1]):
                assert torch.equal(x, y)
            for x, y in zip(mine[2], theirs[2]):
                assert torch.equal(x, y)
            for x, y in zip(mine[3], theirs[3]):
                assert torch.equal(x, y)
            _same_segs(mine[4], theirs[5])


def test_sampling_schedule_matches_impact():
    ref = reference()
    if ref is None:
        return print("skip (no G:/ComfyUi bench)")
    from detailer import detail

    seen = []

    def sigmas_for(steps):
        return torch.linspace(1.0, 0.0, steps + 1)

    def record(tag):
        def fn(model, seed, cfg, positive, negative, sampler, sigmas, latent, **kw):
            seen.append((tag, seed, cfg, sigmas.clone(), latent))
            return {"samples": latent["samples"] + 1}, {"samples": latent["samples"] + 2}
        return fn

    def theirs_sample(model, add_noise, seed, cfg, positive, negative, sampler, sigmas, latent, noise=None, callback=None):
        assert add_noise is True and noise is None
        seen.append(("theirs", seed, cfg, sigmas.clone(), latent))
        return {"samples": latent["samples"] + 1}, {"samples": latent["samples"] + 2}

    def mine_sample(model, seed, cfg, positive, negative, sampler, sigmas, latent):
        seen.append(("mine", seed, cfg, sigmas.clone(), latent))
        return {"samples": latent["samples"] + 1}, {"samples": latent["samples"] + 2}

    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    for steps, denoise, sampler in [(8, 1.0, "euler"), (8, 0.25, "euler"), (4, 0.3, "lcm"), (15, 0.5, "heun"), (2, 0.9, "euler")]:
        seen.clear()
        with mock.patch.object(detail, "calculate_sigmas", lambda m, s, sch, n: sigmas_for(n)), \
                mock.patch.object(detail, "sample_with_custom_noise", mine_sample):
            mine = detail.ksampler_wrapper("model", 7, steps, 1.4, sampler, "simple", "p", "n", latent, denoise)
        with mock.patch.object(ref.sampling, "calculate_sigmas", lambda m, s, sch, n: sigmas_for(n)), \
                mock.patch.object(ref.sampling, "sample_with_custom_noise", theirs_sample):
            theirs = ref.sampling.ksampler_wrapper("model", 7, steps, 1.4, sampler, "simple", "p", "n", latent, denoise)
        assert [s[0] for s in seen] == ["mine", "theirs"], seen
        assert seen[0][1:3] == seen[1][1:3] and torch.equal(seen[0][3], seen[1][3]), (steps, denoise)
        assert torch.equal(mine["samples"], theirs["samples"])
        assert len(seen[0][3]) == steps + 1 and seen[0][3][-1] == 0  # `steps` real steps, ending at sigma 0


def test_wildcard_parsing_matches_impact():
    ref = reference()
    if ref is None:
        return print("skip (no G:/ComfyUi bench)")
    from detailer import segs as S

    text = "[LAB]a header[ALL] shared [t0]red door [t1]  [t10]blue sky"
    mode, chooser = ref.wildcards.process_wildcard_for_segs(text)
    mine_mode, mine = S.process_wildcard_for_segs(text)
    assert mode == mine_mode == "LAB"
    for label in ("t0", "t1", "t2", "t10", "ALL", ""):
        seg = S.SEG(None, None, 1.0, [0, 0, 1, 1], (0, 0, 1, 1), label, None)
        assert chooser.get(seg) == mine.get(seg), label
    mode, chooser = ref.wildcards.process_wildcard_for_segs("a plain prompt")
    mine_mode, mine = S.process_wildcard_for_segs("a plain prompt")
    assert mode is None and mine_mode is None
    assert chooser.get(None)[1] == mine.get(None) == "a plain prompt"
    for text in ("a\n# gone\nb", "# only\nx", "a\nb\n#c\n#d\ne"):
        assert S.strip_comment_lines(text) == ref.wildcards.process_comment_out(text), text


def test_bbox_detection_to_segs_matches_subpack():
    ref = reference()
    if ref is None:
        return print("skip (no G:/ComfyUi bench)")
    from detailer import segs as S

    from detailer.yolo import UltralyticsBBoxDetector

    subcore = subpack_core()

    class _Boxes:
        def __init__(self, xyxy, cls, conf):
            self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
            self._cls, self._conf = cls, conf

        def __getitem__(self, i):
            return types.SimpleNamespace(cls=torch.tensor([self._cls[i]]), conf=torch.tensor([self._conf[i]]))

    class _FakeYolo:
        """What ultralytics' YOLO(...) call returns: pred[0].boxes / pred[0].names."""

        def __init__(self, xyxy, cls, conf):
            self.pred = [types.SimpleNamespace(boxes=_Boxes(xyxy, cls, conf), names={0: "face", 1: "person"})]

        def __call__(self, image, conf=0.3, device="", **kw):
            return self.pred

    torch.manual_seed(2)
    image = torch.rand(1, 80, 110, 3)
    # three boxes: a normal one, a tiny one touching the border (dropped by drop_size), one reaching the right edge
    xyxy = [[10.4, 12.7, 52.2, 60.9], [100.0, 70.0, 109.0, 79.0], [60.0, 5.0, 108.0, 40.0]]
    model = _FakeYolo(xyxy, [0, 0, 1], [0.9, 0.8, 0.7])
    theirs_det = subcore.UltraBBoxDetector(model)
    mine_det = UltralyticsBBoxDetector(model)
    for dilation in (0, 4, 10):
        for crop_factor in (1.5, 3.0):
            for drop_size in (1, 10):
                theirs = theirs_det.detect(image, 0.5, dilation, crop_factor, drop_size)
                mine = mine_det.detect(image, 0.5, dilation, crop_factor, drop_size)
                _same_segs(mine, theirs)
                for x, y in zip(mine[1], theirs[1]):
                    assert torch.equal(x.cropped_image, y.cropped_image)
                    assert np.array_equal(x.confidence, y.confidence)
    assert len(mine[1]) == 2 and [s.label for s in mine[1]] == ["face", "person"]  # the tiny box was dropped
    # no detections at all
    none = _FakeYolo([], [], [])
    _same_segs(UltralyticsBBoxDetector(none).detect(image, 0.5, 4, 1.5), subcore.UltraBBoxDetector(none).detect(image, 0.5, 4, 1.5))


def test_segs_preview_and_pipe_helpers():
    ref = reference()
    if ref is None:
        return print("skip (no G:/ComfyUi bench)")
    from detailer import segs as S

    torch.manual_seed(3)
    t = torch.rand(1, 6, 7, 4)
    assert torch.equal(S.tensor_convert_rgba(t), ref.utils.tensor_convert_rgba(t))
    rgb = torch.rand(1, 6, 7, 3)
    assert torch.equal(S.tensor_convert_rgba(rgb), ref.utils.tensor_convert_rgba(rgb))
    assert torch.equal(S.tensor_convert_rgb(rgb), ref.utils.tensor_convert_rgb(rgb))
    assert torch.equal(S.empty_pil_tensor(), ref.utils.empty_pil_tensor())
    m = torch.rand(4, 5)
    for f in ("make_2d_mask", "make_3d_mask", "make_4d_mask"):
        assert torch.equal(getattr(S, f)(m), getattr(ref.utils, f)(m))
        assert torch.equal(getattr(S, f)(m[None, None]), getattr(ref.utils, f)(m[None, None]))
    assert S.center_of_bbox((2, 4, 10, 8)) == ref.utils.center_of_bbox((2, 4, 10, 8))
    mask = torch.rand(1, 10, 12)
    assert torch.equal(S.crop_condition_mask(mask, torch.zeros(1, 20, 24, 3), [4, 6, 18, 14]),
                       ref.core.crop_condition_mask(mask, torch.zeros(1, 20, 24, 3), [4, 6, 18, 14]))


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok", name, flush=True)
    print(f"{len(tests)} passed")
