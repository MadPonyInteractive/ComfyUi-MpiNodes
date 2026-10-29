# SAM (Segment Anything) model handle and the "center-1" SEGS refinement the Simple Detector uses.
#
# Ported from ComfyUI-Impact-Pack (GPL-3.0: modules/impact/core.py SAMWrapper, make_sam_mask,
# sam_predict, SafeToGPU) on top of the vendored Meta segment-anything predictor in ./segment_anything
# (Apache-2.0). SAM 2 and EfficientSAM, the other hint modes and the negative-point modes are not ported:
# the only model Cubric Studio ships is sam_vit_b and the only hint it sets is center-1.
# comfy is imported inside the functions that need it, so this module loads without ComfyUI.
import logging
import os

import numpy as np
import torch

from .segment_anything import SamPredictor, sam_model_registry
from .segs import center_of_bbox, dilate_mask, make_3d_mask


def sam_kind(model_name):
    """vit_h / vit_l from the file name, vit_b otherwise (Impact's rule)."""
    if "vit_h" in model_name:
        return "vit_h"
    if "vit_l" in model_name:
        return "vit_l"
    return "vit_b"


class SafeToGPU:
    """Move a model to the GPU only if it fits (frees other models first), else leave it where it is."""

    def __init__(self, size):
        self.size = size

    def to_device(self, obj, device):
        import comfy.model_management as mm

        if torch.device(device).type == "cpu":
            obj.to(device)
        elif obj.device.type == "cpu":
            mm.free_memory(self.size * 1.3, device)
            if mm.get_free_memory(device) > self.size * 1.3:
                try:
                    obj.to(device)
                except Exception:
                    logging.warning(f"[MpiDetailer] SAM was not moved to '{device}': not enough memory [1]")
            else:
                logging.warning(f"[MpiDetailer] SAM was not moved to '{device}': not enough memory [2]")


class SAMHandle:
    """The SAM_MODEL value: a Sam model plus where it lives. AUTO moves it to the GPU only while predicting."""

    def __init__(self, model, is_auto_mode, safe_to_gpu):
        self.model = model
        self.is_auto_mode = is_auto_mode
        self.safe_to_gpu = safe_to_gpu

    def prepare_device(self):
        if self.is_auto_mode:
            import comfy.model_management as mm

            self.safe_to_gpu.to_device(self.model, device=mm.get_torch_device())

    def release_device(self):
        if self.is_auto_mode:
            self.model.to(device="cpu")


def load_sam(path, model_name, device_mode):
    """Build the SAM for `path` (a file inside models/sams). device_mode: AUTO | Prefer GPU | CPU."""
    sam = sam_model_registry[sam_kind(model_name)](checkpoint=path)
    safe_to = SafeToGPU(os.path.getsize(path))

    # Unless the user explicitly wants the CPU, the GPU is used
    if device_mode == "Prefer GPU":
        import comfy.model_management as mm

        safe_to.to_device(sam, mm.get_torch_device())

    logging.info(f"[MpiDetailer] loaded SAM {path} (device_mode: {device_mode})")
    return SAMHandle(sam, is_auto_mode=device_mode == "AUTO", safe_to_gpu=safe_to)


def sam_predict(predictor, points, plabs, bbox, threshold):
    """Masks of one prompt with score >= threshold; the best one if none reaches it."""
    point_coords = None if not points else np.array(points)
    point_labels = None if not plabs else np.array(plabs)
    box = np.array([bbox]) if bbox is not None else None

    cur_masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box)

    total_masks = []
    selected = False
    max_score = 0
    max_mask = None
    for idx in range(len(scores)):
        if scores[idx] > max_score:
            max_score = scores[idx]
            max_mask = cur_masks[idx]

        if scores[idx] >= threshold:
            selected = True
            total_masks.append(cur_masks[idx])

    if not selected and max_mask is not None:
        total_masks.append(max_mask)

    return total_masks


def combine_masks(masks):
    """OR of equally sized bool/0-1 masks as a uint8 tensor; None for an empty list (odd shapes are skipped)."""
    if len(masks) == 0:
        return None

    combined = np.array(masks[0]).astype(np.uint8)
    for m in masks[1:]:
        m = np.array(m).astype(np.uint8)
        if combined.shape == m.shape:
            combined = np.bitwise_or(combined, m)
    return torch.from_numpy(combined)


def make_sam_mask_center(sam, segs, image, dilation, threshold, bbox_expansion):
    """One combined (1,H,W) float mask: SAM run on every SEG's bbox, prompted with the bbox centre (hint 'center-1')."""
    sam.prepare_device()
    try:
        image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        # The image is encoded once; upstream re-encodes it for every SEG (same result, slower).
        total_masks = []
        predictor = None
        for seg in segs[1]:
            if predictor is None:
                predictor = SamPredictor(sam.model)
                predictor.set_image(image, "RGB")

            bbox = seg.bbox
            center = center_of_bbox(bbox)

            x1 = max(bbox[0] - bbox_expansion, 0)
            y1 = max(bbox[1] - bbox_expansion, 0)
            x2 = min(bbox[2] + bbox_expansion, image.shape[1])
            y2 = min(bbox[3] + bbox_expansion, image.shape[0])

            total_masks += sam_predict(predictor, [center], [1], [x1, y1, x2, y2], threshold)

        mask = combine_masks(total_masks)
    finally:
        sam.release_device()

    if mask is not None:
        mask = mask.float()
        mask = torch.from_numpy(dilate_mask(mask.cpu().numpy(), dilation))
    else:
        mask = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.float32, device="cpu")  # empty mask

    return make_3d_mask(mask)
