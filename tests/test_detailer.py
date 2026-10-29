"""CPU checks for detailer/ (the Impact-Pack / Subpack port) on tiny tensors. No ComfyUI, no GPU, no weights.
Run with the engine's python (it has torch, numpy, cv2, torchvision):

    G:/ComfyUi/python_embeded/python.exe tests/test_detailer.py

Against the original code: tests/test_detailer_ref.py. Against real weights: tests/test_detailer_models.py.
"""
import glob
import inspect
import json
import os
import sys
import types

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detailer import segs as S  # noqa: E402

WORKFLOWS = "C:/AI/Mpi/Cubric-Vision/comfy_workflows"

OLD_TO_NEW = {
    "BboxDetectorSEGS": "MpiBboxDetectorSEGS", "DetailerForEachPipe": "MpiDetailerForEachPipe",
    "FromBasicPipe": "MpiFromBasicPipe", "ToBasicPipe": "MpiToBasicPipe", "ImpactDilateMask": "MpiDilateMask",
    "ImpactMakeTileSEGS": "MpiMakeTileSEGS", "ImpactSEGSLabelAssign": "MpiSEGSLabelAssign",
    "ImpactSEGSToMaskList": "MpiSEGSToMaskList", "ImpactSegsAndMask": "MpiSegsAndMask",
    "ImpactSimpleDetectorSEGS": "MpiSimpleDetectorSEGS", "MaskDetailerPipe": "MpiMaskDetailerPipe",
    "MaskToSEGS": "MpiMaskToSEGS", "SAMLoader": "MpiSAMLoader", "SEGSPreview": "MpiSEGSPreview",
    "SegsToCombinedMask": "MpiSegsToCombinedMask", "UltralyticsDetectorProvider": "MpiUltralyticsDetectorProvider",
}


def _rect(h, w, boxes):
    m = np.zeros((h, w), np.float32)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2, x1:x2] = 1.0
    return m


def _fake_comfy():
    """Just enough comfy.samplers / folder_paths for INPUT_TYPES; returns the names to remove again."""
    comfy = types.ModuleType("comfy")
    samplers = types.ModuleType("comfy.samplers")
    samplers.KSampler = types.SimpleNamespace(SAMPLERS=["euler", "heun", "lcm", "dpmpp_2m"])
    samplers.SCHEDULER_HANDLERS = {"simple": 1, "beta": 2, "normal": 3, "karras": 4}
    comfy.samplers = samplers
    fp = types.ModuleType("folder_paths")
    fp.models_dir = os.path.join(os.path.dirname(__file__), "_no_models")
    fp.folder_names_and_paths = {}
    fp.supported_pt_extensions = {".pt", ".pth", ".safetensors"}
    fp.add_model_folder_path = lambda name, path: fp.folder_names_and_paths.setdefault(name, ([path], set()))
    fp.get_filename_list = lambda name: []
    sys.modules.update({"comfy": comfy, "comfy.samplers": samplers, "folder_paths": fp})
    return ["comfy", "comfy.samplers", "folder_paths"]


# --- crop regions, morphology -----------------------------------------------------------------

def test_crop_region_is_centred_clamped_and_integer():
    assert S.make_crop_region(100, 80, (40, 30, 60, 50), 2.0) == [30, 20, 70, 60]
    assert S.make_crop_region(100, 80, (0, 0, 10, 10), 3.0) == [0, 0, 30, 30]  # pushed inside, not shrunk
    assert S.make_crop_region(100, 80, (90, 70, 100, 80), 3.0) == [70, 50, 100, 80]
    assert S.make_crop_region(100, 80, (0, 0, 100, 80), 3.0) == [0, 0, 100, 80]  # bigger than the image: whole image
    assert S.make_crop_region(100, 80, (45, 35, 55, 45), 1.0, crop_min_size=30) == [35, 25, 65, 55]
    x1, y1, x2, y2 = S.make_crop_region(64, 64, (5.5, 7.25, 20.5, 33.75), 2.2)
    assert all(isinstance(v, int) for v in (x1, y1, x2, y2)) and 0 <= x1 < x2 <= 64 and 0 <= y1 < y2 <= 64


def test_dilate_grows_and_negative_erodes():
    m = _rect(30, 30, [(10, 10, 20, 20)])
    assert S.dilate_mask(m, 0).sum() == m.sum()
    grown = S.dilate_mask(m, 5)
    assert grown.sum() > m.sum() and grown[10:20, 10:20].min() == 1.0
    eroded = S.dilate_mask(m, -5)
    assert 0 < eroded.sum() < m.sum()
    assert S.dilate_mask(m[None], 3).shape == (30, 30)  # a (1,H,W) mask comes back 2-D
    masks = [((0, 0, 1, 1), m, 0.5)]
    assert S.dilate_masks(masks, 0) is masks
    assert S.dilate_masks(masks, 3)[0][1].sum() > m.sum()


# --- mask -> SEGS -> mask ---------------------------------------------------------------------

def test_mask_to_segs_combined_makes_one_seg_around_everything():
    m = _rect(60, 80, [(5, 5, 20, 30), (40, 10, 70, 50)])
    shape, segs = S.mask_to_segs(m, True, 1.0, False, 1)
    assert shape == (60, 80) and len(segs) == 1
    assert [int(v) for v in segs[0].bbox] == [5, 5, 69, 49]  # inclusive max pixel, as upstream
    assert segs[0].label == "A" and segs[0].confidence == 1.0


def test_mask_to_segs_contours_split_skip_holes_and_drop_small():
    m = _rect(60, 80, [(5, 5, 25, 35), (40, 10, 70, 50)])
    m[12:20, 10:18] = 0.0  # a hole: not a SEG of its own
    m[55:57, 70:72] = 1.0  # 2x2: below drop_size 10
    _, segs = S.mask_to_segs(m, False, 1.5, False, 10)
    assert len(segs) == 2
    _, segs_all = S.mask_to_segs(m, False, 1.5, False, 1)
    assert len(segs_all) == 3
    # contour fill is a hard 0/1 around the blob (hole filled), is_contour=False keeps the mask's own values
    holed = lambda mode: [s for s in S.mask_to_segs(m, False, 1.0, False, 10, is_contour=mode)[1] if s.bbox[0] == 5][0]  # noqa: E731
    filled, kept = holed(True), holed(False)
    assert filled.cropped_mask.sum() > kept.cropped_mask.sum()  # the hole is filled only in contour mode


def test_mask_to_segs_batch_and_empty():
    batch = np.stack([_rect(40, 40, [(5, 5, 15, 15)]), np.zeros((40, 40), np.float32), _rect(40, 40, [(20, 20, 30, 35)])])
    shape, segs = S.mask_to_segs(torch.from_numpy(batch), True, 1.0, False, 1)
    assert shape == (40, 40) and len(segs) == 2
    assert S.mask_to_segs(np.zeros((40, 40), np.float32), False, 1.0, False, 1)[1] == []


def test_segs_to_combined_mask_and_masklist_round_trip():
    m = _rect(60, 80, [(5, 5, 25, 35), (40, 10, 70, 50)])
    segs = S.mask_to_segs(m, False, 1.5, False, 1)
    combined = S.segs_to_combined_mask(segs)
    assert combined.shape == (60, 80) and combined.dtype == torch.float32
    assert torch.equal(combined, torch.from_numpy(m))
    lst = S.segs_to_masklist(segs)
    assert len(lst) == 2 and all(t.shape == (60, 80) for t in lst)
    assert torch.equal(torch.stack(lst).max(0).values, combined)
    empty = S.segs_to_masklist(((7, 9), []))
    assert len(empty) == 1 and empty[0].shape == (7, 9) and empty[0].sum() == 0


def test_segs_bitwise_and_mask_keeps_the_overlap_only():
    m = _rect(60, 80, [(5, 5, 45, 35)])
    segs = S.mask_to_segs(m, False, 1.0, False, 1)
    cut = torch.from_numpy(_rect(60, 80, [(0, 0, 25, 60)]))
    out = S.segs_bitwise_and_mask(segs, cut)
    assert torch.equal(S.segs_to_combined_mask(out), torch.from_numpy(_rect(60, 80, [(5, 5, 25, 35)])))
    assert out[1][0].cropped_image is None and out[1][0].crop_region == segs[1][0].crop_region


def test_labels_assign_and_filter():
    segs = S.mask_to_segs(_rect(60, 80, [(5, 5, 20, 30), (40, 10, 70, 50)]), False, 1.0, False, 1)
    lab = S.label_assign(segs, ["t0", " t1 ", "t2"])
    assert [s.label for s in lab[1]] == ["t0", "t1"]
    assert [s.label for s in S.label_assign(segs, ["only"])[1]] == ["only", "A"]
    kept, rest = S.label_filter(lab, ["t1"])
    assert [s.label for s in kept[1]] == ["t1"] and [s.label for s in rest[1]] == ["t0"]
    assert S.label_filter(lab, ["all"])[0] is lab
    assert kept[0] == (60, 80)


# --- tiles ------------------------------------------------------------------------------------

def test_tile_segs_cover_the_image_in_reading_order():
    img = torch.zeros((1, 96, 128, 3))
    shape, segs = S.make_tile_segs(img, 64, 1.5, 16, 0)
    assert shape == (96, 128) and len(segs) == 6
    cover = np.zeros((96, 128), bool)
    last = (-1, -1)
    for s in segs:
        x1, y1, x2, y2 = s.bbox
        assert (x2 - x1, y2 - y1) == (64, 64) and 0 <= x1 and x2 <= 128 and 0 <= y1 and y2 <= 96
        assert (y1, x1) > last  # row by row, left to right
        last = (y1, x1)
        cover[y1:y2, x1:x2] = True
        cr = s.crop_region
        assert s.cropped_mask.shape == (cr[3] - cr[1], cr[2] - cr[0])
        assert s.cropped_mask[y1 - cr[1]:y2 - cr[1], x1 - cr[0]:x2 - cr[0]].min() == 1.0  # the tile itself is the mask
    assert cover.all()
    # the workflows' numbers (Input_Tile_Size 1024, Input_Tile_Overlap 205) on a 2000x1500 image: 3 x 2 tiles
    big = S.make_tile_segs(torch.zeros((1, 1500, 2000, 3)), 1024, 1.5, 205, 30)
    assert len(big[1]) == 6 and all(s.bbox[2] - s.bbox[0] == 1024 == s.bbox[3] - s.bbox[1] for s in big[1])
    assert big[1][0].bbox[:2] == (0, 0) and big[1][-1].bbox[2:] == (2000, 1500)
    # upstream quirk kept: an image smaller than the tile gets none when the overlap is bigger than the image
    assert S.make_tile_segs(torch.zeros((1, 90, 300, 3)), 1024, 1.5, 205, 30)[1] == []


def test_tile_segs_filter_out_drops_tiles():
    img = torch.zeros((1, 100, 120, 3))
    out = S.mask_to_segs(_rect(100, 120, [(0, 0, 60, 100)]), True, 1.0, False, 1)
    all_tiles = S.make_tile_segs(img, 48, 1.5, 8, 6)
    kept = S.make_tile_segs(img, 48, 1.5, 8, 6, filter_out_segs_opt=out)
    assert 0 < len(kept[1]) < len(all_tiles[1])
    for s in kept[1]:
        assert s.bbox[2] > 60  # nothing fully inside the excluded left half survives


def test_tile_segs_irregular_masks():
    np.random.seed(0)
    img = torch.zeros((1, 100, 130, 3))
    segs = S.make_tile_segs(img, 64, 1.5, 10, 30, 0.7, "Reuse fast")
    assert len(segs[1]) > 0
    for s in segs[1]:
        assert 0 < s.cropped_mask.sum() < s.cropped_mask.size  # a blob, not the full rectangle
        assert s.cropped_mask.min() >= 0 and s.cropped_mask.max() <= 1


def test_random_mask_raw_fills_the_inside():
    np.random.seed(1)
    m = np.zeros((128, 128), np.float32)
    S.random_mask_raw(m, (0, 0, 128, 128), 0.7)
    factor = max(6, int(128 * 0.7 / 4))
    assert m[factor:128 - factor, factor:128 - factor].min() == 1.0
    assert m[0, 0] == 0.0


# --- detections -> SEGS -----------------------------------------------------------------------

def test_detections_to_segs_box_masks_dilation_and_drop_size():
    image = torch.rand(1, 80, 110, 3)
    dets = [("face", np.array([10.4, 12.7, 52.2, 60.9], np.float32), np.array([0.9], np.float32)),
            ("face", np.array([100.0, 70.0, 109.0, 79.0], np.float32), np.array([0.8], np.float32))]
    shape, segs = S.detections_to_segs(image, dets, 0, 1.5, 10)
    assert shape == (80, 110) and [s.label for s in segs] == ["face"]  # the 9x9 box is under drop_size 10
    s = segs[0]
    full = np.zeros((80, 110), np.float32)
    cr = s.crop_region
    full[cr[1]:cr[3], cr[0]:cr[2]] = s.cropped_mask
    assert full[12:61, 10:53].min() == 1.0 and full[11, 30] == 0.0 and full.sum() == 49 * 43  # filled box, int coords
    assert s.cropped_image.shape == (1, cr[3] - cr[1], cr[2] - cr[0], 3)
    big = S.detections_to_segs(image, dets[:1], 6, 1.5, 10)[1][0]
    assert big.cropped_mask.sum() > s.cropped_mask.sum()
    assert S.detections_to_segs(image, [], 6, 1.5, 10) == ((80, 110), [])


# --- paste, feather, resize -------------------------------------------------------------------

def test_tensor_paste_blends_through_the_mask_and_clips():
    img = torch.zeros(1, 10, 10, 3)
    patch = torch.ones(1, 4, 4, 3)
    mask = torch.full((1, 4, 4, 1), 0.5)
    S.tensor_paste(img, patch, (2, 3), mask)
    assert img[0, 3, 2, 0] == 0.5 and img[0, 6, 5, 0] == 0.5 and img[0, 2, 2, 0] == 0 and img[0, 7, 5, 0] == 0
    S.tensor_paste(img, patch, (8, 8), torch.ones(1, 4, 4, 1))  # half outside: clipped
    assert img[0, 9, 9, 0] == 1.0
    before = img.clone()
    S.tensor_paste(img, patch, (50, 50), torch.ones(1, 4, 4, 1))  # fully outside: nothing
    assert torch.equal(img, before)
    rgba = torch.zeros(1, 4, 4, 4)
    S.tensor_paste(rgba, torch.ones(1, 2, 2, 3), (0, 0), torch.ones(1, 2, 2, 1))
    assert rgba[0, 0, 0, 3] == 1.0 and rgba[0, 3, 3, 3] == 0.0


def test_gaussian_feather_and_resize_shapes():
    m = torch.zeros(20, 20)
    m[8:12, 8:12] = 1.0
    out = S.tensor_gaussian_blur_mask(m, 3)
    assert out.shape == (1, 20, 20, 1) and 0 < out[0, 7, 8, 0] < 1
    assert torch.equal(S.tensor_gaussian_blur_mask(m, 0), m[None, ..., None])
    tiny = torch.ones(2, 2)
    assert S.tensor_gaussian_blur_mask(tiny, 5).shape == (1, 2, 2, 1)  # too small to feather: unchanged
    img = torch.rand(1, 8, 10, 3)
    assert S.tensor_resize(img, 20, 16).shape == (1, 16, 20, 3)
    assert S.tensor_resize(torch.rand(1, 8, 10, 1), 5, 4).shape == (1, 4, 5, 1)
    assert S.tensor_convert_rgb(torch.rand(1, 2, 2, 4)).shape == (1, 2, 2, 3)
    assert S.tensor_convert_rgba(torch.rand(1, 2, 2, 3))[..., 3].min() == 1.0


def test_segs_scale_match_scales_regions_and_masks():
    segs = S.mask_to_segs(_rect(40, 60, [(10, 10, 30, 30)]), False, 1.5, False, 1)
    assert S.segs_scale_match(segs, (1, 40, 60, 3)) is segs
    big = S.segs_scale_match(segs, (1, 80, 120, 3))
    assert big[0] == (80, 120)
    seg, src = big[1][0], segs[1][0]
    assert list(seg.crop_region) == [v * 2 for v in src.crop_region]
    assert seg.cropped_mask.shape == (seg.crop_region[3] - seg.crop_region[1], seg.crop_region[2] - seg.crop_region[0])
    # a non-uniform rescale uses the width ratio for x and the height ratio for y
    wide = S.segs_scale_match(segs, (1, 40, 120, 3))[1][0]
    assert wide.crop_region[1] == src.crop_region[1] and wide.crop_region[0] == src.crop_region[0] * 2
    assert wide.crop_region[3] == src.crop_region[3] and wide.crop_region[2] == src.crop_region[2] * 2


# --- per-SEG prompt text ----------------------------------------------------------------------

def test_wildcard_lab_chooser_and_plain_text():
    mode, ch = S.process_wildcard_for_segs("[LAB][ALL]golden hour [t0]a field[t1][t2]sky")
    seg = lambda label: S.SEG(None, None, 1.0, [0, 0, 1, 1], (0, 0, 1, 1), label, None)  # noqa: E731
    assert mode == "LAB"
    assert ch.get(seg("t0")) == "golden houra field"  # ALL first, then the label's own text
    assert ch.get(seg("t1")) == "golden hour"  # an empty entry is dropped
    assert ch.get(seg("t9")) == "golden hour"
    mode, ch = S.process_wildcard_for_segs("a plain prompt")
    assert mode is None and ch.get(seg("t0")) == "a plain prompt"
    mode, ch = S.process_wildcard_for_segs("")
    assert mode is None and ch.get(seg("t0")) == ""
    for text in ("[ASC]a[SEP]b", "[DSC]x", "[ASC-SIZE]x", "[DSC-SIZE]x", "[RND]x"):
        try:
            S.process_wildcard_for_segs(text)
        except ValueError as e:
            assert "not supported" in str(e)
        else:
            raise AssertionError(text)


def test_prompt_text_comment_lines_and_break():
    assert S.strip_comment_lines("x\n# note\ny") == "x y"
    assert S.strip_comment_lines("plain") == "plain"
    assert S.split_break("a BREAK  b BREAK ") == ["a", "b"]
    assert S.split_break("BREAK") == [""]
    assert S.split_break("one") == ["one"]


# --- the nodes --------------------------------------------------------------------------------

def test_package_loads_without_comfy_and_maps_16_nodes():
    sys.modules.pop("comfy", None)
    import detailer
    assert "comfy" not in sys.modules
    assert sorted(detailer.NODE_CLASS_MAPPINGS) == sorted(OLD_TO_NEW.values())
    assert len(detailer.NODE_CLASS_MAPPINGS) == 16 and len(set(detailer.NODE_CLASS_MAPPINGS.values())) == 16
    for name, cls in detailer.NODE_CLASS_MAPPINGS.items():
        assert name.startswith("Mpi") and cls.__name__ == name
        assert name not in OLD_TO_NEW  # never an original class name
        assert detailer.NODE_DISPLAY_NAME_MAPPINGS[name].startswith("Mpi ")
        assert cls.CATEGORY == "MpiNodes/Detailer" and callable(getattr(cls, cls.FUNCTION))
    assert not set(OLD_TO_NEW) & set(detailer.NODE_CLASS_MAPPINGS)


def test_pure_nodes_run_on_tiny_inputs():
    from detailer import nodes as N

    pipe = N.MpiToBasicPipe().doit("m", "c", "v", "p", "n")[0]
    assert pipe == ("m", "c", "v", "p", "n") and N.MpiFromBasicPipe().doit(pipe) == ("m", "c", "v", "p", "n")

    m = torch.from_numpy(_rect(60, 80, [(5, 5, 25, 35), (40, 10, 70, 50)]))[None]
    segs = N.MpiMaskToSEGS().doit(m, False, 1.5, False, 1, False)[0]
    assert len(segs[1]) == 2
    assert N.MpiSegsToCombinedMask().doit(segs)[0].shape == (1, 60, 80)
    masks = N.MpiSEGSToMaskList().doit(segs)[0]
    assert len(masks) == 2 and masks[0].shape == (1, 60, 80)
    grown = N.MpiDilateMask().doit(m, 4)[0]
    assert grown.shape == (1, 60, 80) and grown.sum() > m.sum()
    labelled = N.MpiSEGSLabelAssign().doit(segs, "t0,t1")[0]
    assert [s.label for s in labelled[1]] == ["t0", "t1"]
    cut = N.MpiSegsAndMask().doit(labelled, torch.from_numpy(_rect(60, 80, [(0, 0, 30, 60)]))[None])[0]
    assert sum(s.cropped_mask.sum() for s in cut[1]) < sum(s.cropped_mask.sum() for s in labelled[1])
    combined = N.MpiMaskToSEGS().doit(m, True, 3.0, False, 1, False)[0]
    assert len(combined[1]) == 1
    tiles = N.MpiMakeTileSEGS().doit(torch.zeros(1, 96, 128, 3), 64, 1.5, 16, 0, 0.0, "Reuse fast")[0]
    assert len(tiles[1]) == 6


class _FakeDetector:
    """A BBOX_DETECTOR: any object with detect(); here the real class around a fake model."""

    def __init__(self, xyxy, names):
        from detailer.yolo import UltralyticsBBoxDetector

        def boxes_item(i):
            return types.SimpleNamespace(cls=torch.tensor([i % 2]), conf=torch.tensor([0.9 - 0.1 * i]))

        class Boxes:
            def __init__(self):
                self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)

            def __getitem__(self, i):
                return boxes_item(i)

        pred = [types.SimpleNamespace(boxes=Boxes(), names=names)]
        self.inner = UltralyticsBBoxDetector(lambda image, conf=0.3, device="", **kw: pred)

    def detect(self, *args, **kwargs):
        return self.inner.detect(*args, **kwargs)


def test_bbox_and_simple_detector_nodes_with_a_fake_model_and_sam():
    from detailer import nodes as N
    from detailer import sam as SAM

    image = torch.rand(1, 80, 110, 3)
    det = _FakeDetector([[10, 12, 52, 60], [60, 5, 108, 40]], {0: "face", 1: "person"})

    all_segs = N.MpiBboxDetectorSEGS().doit(det, image, 0.5, 4, 1.5, 10, "all")[0]
    assert [s.label for s in all_segs[1]] == ["face", "person"]
    only = N.MpiBboxDetectorSEGS().doit(det, image, 0.5, 4, 1.5, 10, "person")[0]
    assert [s.label for s in only[1]] == ["person"]

    plain = N.MpiSimpleDetectorSEGS().doit(det, image, 0.5, 6, 1.8, 10, 0.5, 0, 0, 0.7, 0)[0]
    assert len(plain[1]) == 2

    # a SAM that answers a fixed 20x20 square for the first box and a half-box for the second
    class _Predictor:
        calls = []

        def __init__(self, model):
            pass

        def set_image(self, image, fmt):
            _Predictor.calls.append(("set_image", image.shape, fmt))

        def predict(self, point_coords, point_labels, box):
            _Predictor.calls.append(("predict", point_coords.tolist(), point_labels.tolist(), box.tolist()))
            m = np.zeros((80, 110), bool)
            x1, y1, x2, y2 = [int(v) for v in box[0]]
            m[y1:(y1 + y2) // 2, x1:x2] = True
            return np.stack([m, ~m]), np.array([0.9, 0.2]), None

    class _Sam:
        model = object()
        prepared = []

        def prepare_device(self):
            self.prepared.append("up")

        def release_device(self):
            self.prepared.append("down")

    real = SAM.SamPredictor
    SAM.SamPredictor = _Predictor
    try:
        sam = _Sam()
        refined = N.MpiSimpleDetectorSEGS().doit(det, image, 0.5, 6, 1.8, 10, 0.5, 0, 3, 0.7, 0, sam_model_opt=sam)[0]
    finally:
        SAM.SamPredictor = real
    assert sam.prepared == ["up", "down"]
    assert [c[0] for c in _Predictor.calls].count("set_image") == 1  # encoded once for both boxes
    first_predict = [c for c in _Predictor.calls if c[0] == "predict"][0]
    # box (10,12,52,60): centre point as the one foreground hint, box grown by sub_bbox_expansion=3
    assert first_predict[1] == [[31.0, 36.0]] and first_predict[2] == [1]
    assert first_predict[3] == [[7.0, 9.0, 55.0, 63.0]]
    assert len(refined[1]) == 2
    for a, b in zip(refined[1], plain[1]):
        assert 0 < a.cropped_mask.sum() < b.cropped_mask.sum()  # the box mask was cut down to the SAM mask

    # no SEGS: no SAM call, an empty (1,H,W) mask
    _Predictor.calls.clear()
    SAM.SamPredictor = _Predictor
    try:
        empty = SAM.make_sam_mask_center(_Sam(), ((80, 110), []), image, 0, 0.5, 0)
    finally:
        SAM.SamPredictor = real
    assert empty.shape == (1, 80, 110) and empty.sum() == 0 and _Predictor.calls == []


def test_sam_helpers():
    from detailer import sam as SAM

    assert SAM.sam_kind("sam_vit_b_01ec64.pth") == "vit_b" and SAM.sam_kind("sam_vit_h_4b8939.pth") == "vit_h"
    assert SAM.sam_kind("sam_vit_l_0b3195.pth") == "vit_l"
    assert SAM.combine_masks([]) is None
    both = SAM.combine_masks([np.array([[1, 0], [0, 0]], bool), np.array([[0, 0], [0, 1]], bool), np.zeros((3, 3), bool)])
    assert both.tolist() == [[1, 0], [0, 1]]  # the odd-sized mask is skipped

    class _P:
        def predict(self, point_coords, point_labels, box):
            return [np.array([1]), np.array([2]), np.array([3])], np.array([0.4, 0.95, 0.9]), None

    assert [int(m[0]) for m in SAM.sam_predict(_P(), [(1, 1)], [1], [0, 0, 5, 5], 0.9)] == [2, 3]
    assert [int(m[0]) for m in SAM.sam_predict(_P(), [(1, 1)], [1], [0, 0, 5, 5], 0.99)] == [2]  # best one when none passes


def test_detail_helpers_without_comfy():
    from detailer import detail

    assert detail.SDE_SAMPLERS[0] == "dpmpp_sde"
    # Impact's floor(steps / denoise) tail schedule: 4 steps at 0.25 run as the last 4 of 16
    steps, denoise = 4, 0.25
    import math
    advanced = math.floor(steps / denoise)
    assert (advanced, advanced - steps) == (16, 12)


def test_every_workflow_input_is_accepted_by_the_ported_node():
    """Each class_type the workflows use (and the values they set) must fit the Mpi node that replaces it."""
    files = sorted(glob.glob(f"{WORKFLOWS}/*.json") + glob.glob(f"{WORKFLOWS}/raw/*.json"))
    if not files:
        return print("skip (no Cubric-Vision workflows on this machine)")
    added = _fake_comfy()
    try:
        import detailer
        seen = {}
        for path in files:
            data = json.load(open(path, encoding="utf8"))
            if isinstance(data.get("nodes"), list):
                continue  # a UI-format graph, not an API one
            for node in data.values():
                if not isinstance(node, dict) or node.get("class_type") not in OLD_TO_NEW:
                    continue
                cls = detailer.NODE_CLASS_MAPPINGS[OLD_TO_NEW[node["class_type"]]]
                spec = cls.INPUT_TYPES()
                known = {**spec["required"], **spec.get("optional", {})}
                params = inspect.signature(getattr(cls, cls.FUNCTION)).parameters
                for key, value in node["inputs"].items():
                    assert key in known, (node["class_type"], key)
                    assert key in params, (node["class_type"], key, "not a parameter of", cls.FUNCTION)
                    seen.setdefault(node["class_type"], set()).add(key)
                    if isinstance(value, list) and len(value) == 2:
                        continue  # a link
                    kind = known[key][0]
                    opts = known[key][1] if len(known[key]) > 1 else {}
                    if isinstance(kind, list):
                        if key != "model_name":  # model files are not on this machine
                            assert value in kind, (node["class_type"], key, value, kind)
                    elif kind == "BOOLEAN":
                        assert isinstance(value, bool), (node["class_type"], key, value)
                    elif kind in ("INT", "FLOAT"):
                        assert isinstance(value, (int, float)) and not isinstance(value, bool)
                        assert opts.get("min", -1e18) <= value <= opts.get("max", 1e18), (node["class_type"], key, value)
                    elif kind == "STRING":
                        assert isinstance(value, str)
                # every required input the node has is set or linked
                for key in spec["required"]:
                    assert key in node["inputs"], (node["class_type"], "missing required", key, path)
        assert set(seen) == set(OLD_TO_NEW), set(OLD_TO_NEW) - set(seen)
    finally:
        for name in added:
            sys.modules.pop(name, None)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok", name, flush=True)
    print(f"{len(tests)} passed")
