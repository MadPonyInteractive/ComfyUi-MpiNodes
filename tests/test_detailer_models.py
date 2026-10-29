"""detailer/ against the REAL weights: the vendored SAM vs the segment-anything pip package (and Impact's
SAM refinement on top of it), and the YOLO face / person models through ultralytics. CPU, a few seconds
each, skipped (prints "skip") when the weights or the packs are not on this machine. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_detailer_models.py
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # the embedded python does not add the script folder

# the real packages first: test_detailer_ref stubs them in sys.modules later
try:
    import segment_anything as pip_sam
except ImportError:
    pip_sam = None
try:
    import ultralytics  # noqa: F401
except ImportError:
    ultralytics = None

MODEL_ROOTS = ["M:/CubricModels", "C:/AI/Mpi/Cubric-Vision/engine/mpi_models"]
PICTURE = "C:/AI/Mpi/Cubric-Vision/.github/readme/gallery.webp"


def _weight(rel):
    for root in MODEL_ROOTS:
        if os.path.isfile(f"{root}/{rel}"):
            return f"{root}/{rel}"
    return None


def _picture(max_side=640):
    if not os.path.isfile(PICTURE):
        return None
    img = Image.open(PICTURE).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def _blob_mask(h, w, boxes):
    m = np.zeros((h, w), np.float32)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2, x1:x2] = 1.0
    return m


def test_vendored_sam_equals_the_pip_package_on_the_real_vit_b():
    path = _weight("sams/sam_vit_b_01ec64.pth")
    if pip_sam is None or path is None:
        return print("skip (no segment-anything pip package or sam_vit_b weights)")
    from detailer.segment_anything import SamPredictor, sam_model_registry

    mine = sam_model_registry["vit_b"](checkpoint=path)  # torch.load(weights_only=True)
    theirs = pip_sam.sam_model_registry["vit_b"](checkpoint=path)
    assert all(torch.equal(a, b) for a, b in zip(mine.state_dict().values(), theirs.state_dict().values()))

    rng = np.random.default_rng(0)
    image = (np.add.outer(np.linspace(0, 1, 120), np.linspace(0, 1, 160))[..., None] * 100 + rng.random((120, 160, 3)) * 50).astype(np.uint8)
    image[30:80, 40:110] = (200, 60, 60)
    box = np.array([[35, 25, 115, 85]])
    point = np.array([[75, 55]])
    a, b = SamPredictor(mine), pip_sam.SamPredictor(theirs)
    a.set_image(image, "RGB")
    b.set_image(image, "RGB")
    for kwargs in ({"point_coords": point, "point_labels": np.array([1]), "box": box}, {"box": box},
                   {"point_coords": point, "point_labels": np.array([1])}):
        ma, sa, _ = a.predict(**kwargs)
        mb, sb, _ = b.predict(**kwargs)
        assert np.array_equal(ma, mb) and np.array_equal(sa, sb), kwargs
    assert ma.shape == (3, 120, 160) and ma.any()


def test_sam_refinement_equals_impacts_make_sam_mask():
    path = _weight("sams/sam_vit_b_01ec64.pth")
    if pip_sam is None or path is None:
        return print("skip (no segment-anything pip package or sam_vit_b weights)")
    from test_detailer_ref import reference

    from detailer import sam as SAM
    from detailer import segs as S

    ref = reference()
    if ref is None:
        return print("skip (no G:/ComfyUi bench)")
    ref.core.SamPredictor = pip_sam.SamPredictor

    theirs_model = pip_sam.sam_model_registry["vit_b"](checkpoint=path)
    theirs_model.sam_wrapper = ref.core.SAMWrapper(theirs_model, is_auto_mode=False)
    mine = SAM.load_sam(path, "sam_vit_b_01ec64.pth", "CPU")

    rng = np.random.default_rng(1)
    image = torch.from_numpy((rng.random((1, 120, 160, 3)) * 0.4).astype(np.float32))
    image[:, 25:85, 35:90] = torch.tensor([0.8, 0.3, 0.3])
    image[:, 60:110, 100:150] = torch.tensor([0.2, 0.7, 0.4])
    mask = _blob_mask(120, 160, [(35, 25, 90, 85), (100, 60, 150, 110)])
    segs = S.mask_to_segs(mask, False, 1.5, False, 1)
    rsegs = ref.core.mask_to_segs(mask, False, 1.5, False, 1)

    for dilation, threshold, expansion in [(0, 0.5, 0), (4, 0.9, 6), (-2, 0.99, 0)]:
        got = SAM.make_sam_mask_center(mine, segs, image, dilation, threshold, expansion)
        want = ref.core.make_sam_mask(theirs_model, rsegs, image, "center-1", dilation, threshold, expansion, 0.7, "False")
        assert got.shape == want.shape == (1, 120, 160)
        assert torch.equal(got, want), (dilation, threshold, expansion)
        assert got.sum() > 0
    # the whole Simple Detector step (SAM mask cut into the box SEGS) is the same, too
    sub = SAM.make_sam_mask_center(mine, segs, image, 0, 0.5, 0)
    cut = S.segs_bitwise_and_mask(segs, sub)
    rcut = ref.core.segs_bitwise_and_mask(rsegs, ref.core.make_sam_mask(theirs_model, rsegs, image, "center-1", 0, 0.5, 0, 0.7, "False"))
    for x, y in zip(cut[1], rcut[1]):
        assert np.array_equal(x.cropped_mask, y.cropped_mask)


_COMFY_CHECK = r'''
import sys
sys.path.insert(0, "G:/ComfyUi/ComfyUI")
sys.path.append(sys.argv[1])  # the repo has a json.py that must not shadow the stdlib
sys.argv = ["comfy", "--cpu"]  # comfy.cli_args parses sys.argv
import folder_paths, nodes, latent_preview
import comfy.model_management as mm, comfy.samplers, comfy.sample, comfy.utils
from comfy.k_diffusion import sampling as kd
from comfy_extras import nodes_differential_diffusion as dd
# everything detail.py / nodes.py / sam.py call, at this ComfyUI version
for obj, names in [
    (comfy.samplers, ["SCHEDULER_HANDLERS", "calculate_sigmas", "sampler_object", "KSAMPLER"]),
    (comfy.sample, ["prepare_noise", "fix_empty_latent_channels", "sample_custom"]),
    (comfy.utils, ["PROGRESS_BAR_ENABLED"]),
    (mm, ["OOM_EXCEPTION", "get_torch_device", "intermediate_device", "free_memory", "get_free_memory"]),
    (latent_preview, ["prepare_callback"]),
    (kd, ["BrownianTreeNoiseSampler"] + ["sample_" + n for n in ("dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu")]),
    (nodes, ["CLIPTextEncode", "ConditioningConcat", "VAEEncode", "VAEEncodeTiled", "VAEDecodeTiled", "InpaintModelConditioning"]),
    (dd, ["DifferentialDiffusion"]),
    (folder_paths, ["add_model_folder_path", "folder_names_and_paths", "supported_pt_extensions", "get_full_path", "get_filename_list", "get_folder_paths", "get_temp_directory", "get_save_image_path", "models_dir"]),
]:
    for name in names:
        assert hasattr(obj, name), (obj.__name__, name)
import inspect
assert "noise_mask" in inspect.signature(nodes.InpaintModelConditioning().encode).parameters
assert hasattr(dd.DifferentialDiffusion, "execute")

import detailer
assert "comfy" in sys.modules
for name, cls in detailer.NODE_CLASS_MAPPINGS.items():
    spec = cls.INPUT_TYPES()  # real comfy.samplers / folder_paths
    assert spec["required"], name
    if "sampler_name" in spec["required"]:
        assert "euler" in spec["required"]["sampler_name"][0] and "beta" in spec["required"]["scheduler"][0], name
        assert {"normal", "simple", "beta"} <= set(spec["required"]["scheduler"][0])
        assert {"euler", "heun", "lcm"} <= set(spec["required"]["sampler_name"][0])
for name in ("sams", "ultralytics", "ultralytics_bbox", "ultralytics_segm"):
    assert name in folder_paths.folder_names_and_paths, name
print("COMFY API OK", len(detailer.NODE_CLASS_MAPPINGS))
'''


def test_ported_nodes_fit_the_comfyui_api_at_the_pinned_version():
    import subprocess

    if not os.path.isdir("G:/ComfyUi/ComfyUI"):
        return print("skip (no G:/ComfyUi bench)")
    exe = "G:/ComfyUi/python_embeded/python.exe"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run([exe, "-c", _COMFY_CHECK, repo], capture_output=True, text=True, timeout=180,
                          cwd="G:/ComfyUi/ComfyUI")
    assert "COMFY API OK 16" in done.stdout, done.stdout[-1500:] + done.stderr[-3000:]


class _Replay:
    """Stands in for a YOLO model: returns a prediction recorded earlier."""

    def __init__(self, pred):
        self.pred = pred

    def __call__(self, image, conf=0.3, device="", **kw):
        return self.pred


def test_yolo_models_load_and_detect_like_the_subpack():
    pic = _picture()
    models = {name: _weight(f"ultralytics/bbox/{name}") for name in ("face_yolov8n.pt", "person_yolov8n-seg.pt", "hand_yolov8n.pt")}
    if ultralytics is None or pic is None or not any(models.values()):
        return print("skip (no ultralytics package, YOLO weights or picture)")
    from detailer import segs as S
    from detailer.yolo import UltralyticsBBoxDetector, inference_bbox, load_yolo

    image = torch.from_numpy(np.asarray(pic).astype(np.float32) / 255.0)[None]
    recorded = {}
    for name, path in models.items():
        if path is None:
            continue
        model = load_yolo(path)  # real ultralytics.YOLO
        for conf in (0.1, 0.5):
            recorded[(name, conf)] = (model, model(pic, conf=conf, device="", verbose=False))
            got = inference_bbox(model, pic, conf)
            boxes = recorded[(name, conf)][1][0].boxes.xyxy.cpu().numpy()
            assert len(got) == len(boxes) and all(np.array_equal(g[1], b) for g, b in zip(got, boxes))
    found = sum(len(v[1][0].boxes.xyxy) for v in recorded.values())
    print(f"  YOLO found {found} boxes over {len(recorded)} runs on {os.path.basename(PICTURE)} {pic.size}")

    # the Subpack's own post-processing (imported with ultralytics stubbed) on the same recorded predictions
    from test_detailer_ref import subpack_core

    subcore = subpack_core()
    for (name, conf), (_, pred) in recorded.items():
        for dilation, crop_factor, drop_size in [(10, 3.0, 10), (4, 1.5, 1)]:
            theirs = subcore.UltraBBoxDetector(_Replay(pred)).detect(image, conf, dilation, crop_factor, drop_size)
            mine = UltralyticsBBoxDetector(_Replay(pred)).detect(image, conf, dilation, crop_factor, drop_size)
            assert theirs[0] == mine[0] and len(theirs[1]) == len(mine[1])
            for x, y in zip(mine[1], theirs[1]):
                assert list(x.crop_region) == list(y.crop_region) and x.label == y.label
                assert np.array_equal(x.cropped_mask, y.cropped_mask) and torch.equal(x.cropped_image, y.cropped_image)
            if mine[1]:
                assert S.segs_to_combined_mask(mine).sum() > 0


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok", name, flush=True)
    print(f"{len(tests)} passed")
