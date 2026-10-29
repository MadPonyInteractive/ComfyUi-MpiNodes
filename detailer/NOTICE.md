# detailer/ - provenance

This folder is a port of the nodes Cubric Studio uses from two ComfyUI packs, so they can be dropped
from the engine. It is part of ComfyUi-MpiNodes (AGPL-3.0); the sources are GPL-3.0 / AGPL-3.0 /
Apache-2.0, all of which may be copied into an AGPL-3.0 work with their notices kept.

| File | Upstream | Licence |
|---|---|---|
| `segs.py`, `detail.py`, `sam.py`, `nodes.py` | ComfyUI-Impact-Pack, (c) ltdrdata, `modules/impact/{utils,core,impact_pack,impact_sampling,segs_nodes,detectors,pipe,wildcards}.py` | GPL-3.0 |
| `yolo.py` | ComfyUI-Impact-Subpack, (c) ltdrdata, `modules/{subcore,subpack_nodes}.py` | AGPL-3.0 |
| `segment_anything/` | facebookresearch/segment-anything 1.0 (see `segment_anything/LICENSE`), (c) Meta Platforms, Inc. | Apache-2.0 |

Class map (original -> port): BboxDetectorSEGS -> MpiBboxDetectorSEGS, DetailerForEachPipe ->
MpiDetailerForEachPipe, FromBasicPipe -> MpiFromBasicPipe, ToBasicPipe -> MpiToBasicPipe, ImpactDilateMask ->
MpiDilateMask, ImpactMakeTileSEGS -> MpiMakeTileSEGS, ImpactSEGSLabelAssign -> MpiSEGSLabelAssign,
ImpactSEGSToMaskList -> MpiSEGSToMaskList, ImpactSegsAndMask -> MpiSegsAndMask, ImpactSimpleDetectorSEGS ->
MpiSimpleDetectorSEGS, MaskDetailerPipe -> MpiMaskDetailerPipe, MaskToSEGS -> MpiMaskToSEGS, SAMLoader ->
MpiSAMLoader, SEGSPreview -> MpiSEGSPreview, SegsToCombinedMask -> MpiSegsToCombinedMask,
UltralyticsDetectorProvider -> MpiUltralyticsDetectorProvider.

## What is not ported (nothing in Cubric Studio's workflows sets it)

- Detailer hooks, refiner model / `refiner_basic_pipe_opt`, ControlNet / IPAdapter SEG wrappers, the
  `DUMMY` model, `NegativePlaceholder`, `scheduler_func_opt`, AYS / GITS / LTXV / OSS scheduler extras.
- Wildcards: the per-SEG prompt is a plain prompt (`[CONCAT]`, `[LAB]` labels, `[SKIP]`, `[STOP]`, `#` comment
  lines and `BREAK` work). `{a|b}`, `__file__` and `<lora:...>` are NOT expanded, and the ordered modes
  `[ASC]` `[DSC]` `[ASC-SIZE]` `[DSC-SIZE]` `[RND]` raise.
- SAM 2, EfficientSAM, SAM hints other than `center-1`, negative points. SAM `vit_b` / `vit_l` / `vit_h` load.
- `segm/` ultralytics models and the SEGM_DETECTOR branch; `person_yolov8n-seg.pt` is read as a bbox model,
  as before.

## Deliberate differences from upstream

- SAM encodes the image once per call, not once per SEG (same masks).
- `random_mask_raw` (irregular tile masks) runs its 16 line chunks one after another instead of in 16 threads and
  draws each circle with array slicing. The radii are random and were already unordered in the threaded version.
- `segs_scale_match` scales y with the height ratio (upstream used the width ratio for y of x2/y2).
- `tensor_convert_rgb` / `_rgba` work for 1 and 4 channels (upstream raised on those).
- A decode that runs out of memory falls back to a tiled decode (as upstream, but only on the OOM exception).
- Weights: SAM checkpoints are read as plain tensors only (`torch.load` with `weights_only=True`). YOLO `.pt`
  files hold whole ultralytics model objects, so `ultralytics.YOLO` loads them itself; the files are the
  engine's sha256-pinned assets (`face-yolov8n`, `hand-yolov8n`, `person-yolov8n-seg` in assetDeps.js).
