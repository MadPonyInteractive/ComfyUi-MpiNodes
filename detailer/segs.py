# Pure maths behind the detailer nodes: SEGS (cropped detections), crop regions, mask dilation,
# contour/combined mask -> SEGS, tile SEGS, label assign, paste and feather. No ComfyUI import, so
# tests/test_detailer*.py can load it alone (torch, numpy, cv2, PIL and torchvision only).
#
# Ported from ComfyUI-Impact-Pack (ltdrdata, GPL-3.0: modules/impact/utils.py, core.py, segs_nodes.py)
# and ComfyUI-Impact-Subpack (AGPL-3.0: modules/subcore.py). Copying into this AGPL-3.0 pack is
# allowed; the notices stay. Only the code paths Cubric Studio's workflows reach are here; the
# differences from upstream are listed in NOTICE.md.
#
# SEGS is Impact's type: (shape, [SEG, ...]) with shape = (height, width) of the source image.
import logging
import math
import re
from collections import namedtuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import GaussianBlur

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
    defaults=[None],
)

LANCZOS = Image.Resampling.LANCZOS


# --- tensor / PIL helpers ---------------------------------------------------------------------

def _check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")


def _check_mask(mask):
    if mask.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions")
    if mask.shape[-1] != 1:
        raise ValueError(f"Expected 1 channel for mask, but found {mask.shape[-1]} channels")


def tensor2pil(image):
    _check_image(image)
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def to_tensor(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image)) / 255.0
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    raise ValueError(f"Cannot convert {type(image)} to torch.Tensor")


def tensor_resize(image, w: int, h: int):
    """Lanczos through PIL for RGB(A) (what Impact does), bilinear for a 1-channel tensor."""
    _check_image(image)
    if image.shape[3] >= 3:
        out = [pil2tensor(tensor2pil(single.unsqueeze(0)).resize((w, h), resample=LANCZOS)) for single in image]
        return torch.cat(out, dim=0)
    x = image.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(h, w), mode="bilinear")
    return x.permute(0, 2, 3, 1)


def tensor_get_size(image):
    _check_image(image)
    _, h, w, _ = image.shape
    return (w, h)


def tensor_convert_rgb(image):
    """NHWC with 1, 3 or 4 channels -> 3 channels."""
    _check_image(image)
    n = image.shape[-1]
    if n == 3:
        return image
    if n == 4:
        return image[..., :3].clone()
    return image.expand(-1, -1, -1, 3).clone()


def tensor_convert_rgba(image):
    _check_image(image)
    n = image.shape[-1]
    if n == 4:
        return image
    if n == 3:
        alpha = torch.ones((*image.shape[:-1], 1))
        return torch.cat((image, alpha), dim=-1)
    return torch.cat((image.expand(-1, -1, -1, 3), torch.ones((*image.shape[:-1], 1))), dim=-1)


def tensor_putalpha(image, mask):
    _check_image(image)
    _check_mask(mask)
    image[..., -1] = mask[..., 0]


def empty_pil_tensor(w=64, h=64):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def make_2d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)
    if len(mask.shape) == 3:
        return mask.squeeze(0)
    return mask


def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)
    if len(mask.shape) == 2:
        return mask.unsqueeze(0)
    return mask


def make_4d_mask(mask):
    if len(mask.shape) == 3:
        return mask.unsqueeze(0)
    if len(mask.shape) == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    return mask


def resize_mask(mask, size):
    mask = make_4d_mask(mask)
    resized = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def tensor_gaussian_blur_mask(mask, kernel_size, sigma=10.0):
    """NHWC mask out of a 2-D / 3-D / 4-D np.ndarray or tensor. Skips the feather when the mask is too small for it."""
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        mask = mask[None, ..., None]
    elif mask.ndim == 3:
        mask = mask[..., None]

    _check_mask(mask)

    if kernel_size <= 0:
        return mask

    kernel_size = kernel_size * 2 + 1

    shortest = min(mask.shape[1], mask.shape[2])
    if shortest <= kernel_size:
        kernel_size = int(shortest / 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return mask  # too small to feather

    blurred = GaussianBlur(kernel_size=kernel_size, sigma=sigma)(mask[:, None, ..., 0])
    return blurred[:, 0, ..., None]


def tensor_paste(image1, image2, left_top, mask):
    """Paste image2 onto image1 (in place) at left_top through mask. RGB and RGBA mixes."""
    _check_image(image1)
    _check_image(image2)
    _check_mask(mask)

    if image2.shape[1:3] != mask.shape[1:3]:
        mask = resize_mask(mask.squeeze(dim=3), image2.shape[1:3]).unsqueeze(dim=3)

    x, y = left_top
    _, h1, w1, c1 = image1.shape
    _, h2, w2, c2 = image2.shape

    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y
    if w <= 0 or h <= 0:
        return  # patch is out of bounds

    mask = mask[:, :h, :w, :]
    region1 = image1[:, y:y + h, x:x + w, :]
    region2 = image2[:, :h, :w, :]

    if c1 == 3 and c2 == 3:
        image1[:, y:y + h, x:x + w, :] = (1 - mask) * region1 + mask * region2
    elif c1 == 4 and c2 == 4:
        image1[:, y:y + h, x:x + w, :3] = (1 - mask) * region1[..., :3] + mask * region2[..., :3]
        a1 = region1[..., 3:4]
        a2 = region2[..., 3:4] * mask
        image1[:, y:y + h, x:x + w, 3:4] = a1 + a2 * (1 - a1)
    elif c1 == 4 and c2 == 3:
        image1[:, y:y + h, x:x + w, :3] = (1 - mask) * region1[..., :3] + mask * region2
        image1[:, y:y + h, x:x + w, 3:4] = region1[..., 3:4] * (1 - mask) + mask
    elif c1 == 3 and c2 == 4:
        effective = mask * region2[..., 3:4]
        image1[:, y:y + h, x:x + w, :] = (1 - effective) * region1 + effective * region2[..., :3]


# --- crop regions -----------------------------------------------------------------------------

def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp + size)
    return int(new_startp), int(new_endp)


def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor
    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)
    return [new_x1, new_y1, new_x2, new_y2]


def crop_ndarray4(npimg, crop_region):
    x1, y1, x2, y2 = crop_region[0], crop_region[1], crop_region[2], crop_region[3]
    return npimg[:, y1:y2, x1:x2, :]


def crop_ndarray3(npimg, crop_region):
    x1, y1, x2, y2 = crop_region[0], crop_region[1], crop_region[2], crop_region[3]
    return npimg[:, y1:y2, x1:x2]


def crop_ndarray2(npimg, crop_region):
    x1, y1, x2, y2 = crop_region[0], crop_region[1], crop_region[2], crop_region[3]
    return npimg[y1:y2, x1:x2]


def crop_condition_mask(mask, image, crop_region):
    """Crop a conditioning mask (B,H,W) with a crop region given in image pixels."""
    cond_scale = (mask.shape[1] / image.shape[1], mask.shape[2] / image.shape[2])
    mask_region = [round(v * cond_scale[i % 2]) for i, v in enumerate(crop_region)]
    return crop_ndarray3(mask, mask_region)


def center_of_bbox(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w / 2, bbox[1] + h / 2


# --- mask morphology --------------------------------------------------------------------------

def dilate_mask(mask, dilation_factor, iterations=1):
    """Square-kernel dilate (positive) or erode (negative) of a 2-D mask array/tensor."""
    if dilation_factor == 0:
        return make_2d_mask(mask)

    mask = make_2d_mask(mask)
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)
    if dilation_factor > 0:
        return cv2.dilate(mask, kernel, iterations)
    return cv2.erode(mask, kernel, iterations)


def dilate_masks(segmasks, dilation_factor, iterations=1):
    """Same, over [(bbox, mask, confidence), ...]."""
    if dilation_factor == 0:
        return segmasks

    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)
    out = []
    for bbox, cv2_mask, confidence in segmasks:
        if dilation_factor > 0:
            dilated = cv2.dilate(cv2_mask, kernel, iterations)
        else:
            dilated = cv2.erode(cv2_mask, kernel, iterations)
        out.append((bbox, dilated, confidence))
    return out


def dilate_segs(segs, factor):
    if factor == 0:
        return segs

    new_segs = []
    for seg in segs[1]:
        new_mask = dilate_mask(seg.cropped_mask, factor)
        new_segs.append(SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label,
                            seg.control_net_wrapper))
    return (segs[0], new_segs)


# --- SEGS <-> masks ---------------------------------------------------------------------------

def segs_scale_match(segs, target_shape):
    """Rescale SEGS made on one image size to an image of target_shape (NHWC)."""
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs

    rh = th / h
    rw = tw / w

    new_segs = []
    for seg in segs[1]:
        cropped_image = seg.cropped_image
        cropped_mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region
        bx1, by1, bx2, by2 = seg.bbox

        crop_region = int(x1 * rw), int(y1 * rh), int(x2 * rw), int(y2 * rh)
        bbox = int(bx1 * rw), int(by1 * rh), int(bx2 * rw), int(by2 * rh)
        new_w = crop_region[2] - crop_region[0]
        new_h = crop_region[3] - crop_region[1]

        if isinstance(cropped_mask, np.ndarray):
            cropped_mask = torch.from_numpy(cropped_mask)

        if isinstance(cropped_mask, torch.Tensor) and len(cropped_mask.shape) == 3:
            cropped_mask = F.interpolate(cropped_mask.unsqueeze(0), size=(new_h, new_w), mode="bilinear",
                                         align_corners=False)
            cropped_mask = cropped_mask.squeeze(0)
        else:
            cropped_mask = F.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w),
                                         mode="bilinear", align_corners=False)
            cropped_mask = cropped_mask.squeeze(0).squeeze(0).numpy()

        if cropped_image is not None:
            src = cropped_image if isinstance(cropped_image, torch.Tensor) else torch.from_numpy(cropped_image)
            cropped_image = tensor_resize(src, new_w, new_h).numpy()

        new_segs.append(SEG(cropped_image, cropped_mask, seg.confidence, crop_region, bbox, seg.label,
                            seg.control_net_wrapper))

    return (th, tw), new_segs


def segs_to_combined_mask(segs):
    h, w = segs[0][0], segs[0][1]
    mask = np.zeros((h, w), dtype=np.uint8)
    for seg in segs[1]:
        cr = seg.crop_region
        mask[cr[1]:cr[3], cr[0]:cr[2]] |= (seg.cropped_mask * 255).astype(np.uint8)
    return torch.from_numpy(mask.astype(np.float32) / 255.0)


def segs_to_masklist(segs):
    h, w = segs[0][0], segs[0][1]

    masks = []
    for seg in segs[1]:
        cropped_mask = seg.cropped_mask
        if isinstance(cropped_mask, np.ndarray):
            cropped_mask = torch.from_numpy(cropped_mask)
        if cropped_mask.ndim == 2:
            cropped_mask = cropped_mask.unsqueeze(0)

        mask = torch.zeros((len(cropped_mask), h, w), dtype=torch.uint8)
        cr = seg.crop_region
        mask[:, cr[1]:cr[3], cr[0]:cr[2]] |= (cropped_mask * 255).to(torch.uint8)
        mask = (mask / 255.0).to(torch.float32)
        masks.extend(mask)

    if len(masks) == 0:
        masks = [torch.zeros((h, w), dtype=torch.float32, device="cpu")]
    return masks


def segs_bitwise_and_mask(segs, mask):
    """Intersect every SEG's mask with a full-image mask (pixelwise, on 8-bit values)."""
    mask = make_2d_mask(mask)
    mask = (mask.cpu().numpy() * 255).astype(np.uint8)

    items = []
    for seg in segs[1]:
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        cr = seg.crop_region
        cropped_mask2 = mask[cr[1]:cr[3], cr[0]:cr[2]]

        new_mask = np.bitwise_and(cropped_mask, cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0
        items.append(SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None))

    return segs[0], items


def mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size=1, label="A", crop_min_size=None, is_contour=True):
    """Mask (H,W) / (B,H,W) tensor or array -> SEGS.

    combined=True: one SEG around everything non-zero per mask. Otherwise one SEG per outer contour
    (holes are skipped); is_contour=False keeps the mask's own values inside each contour instead of
    a hard 0/1 fill.
    """
    drop_size = max(drop_size, 1)

    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy()

    result = []

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)

    for i in range(mask.shape[0]):
        mask_i = mask[i]

        if combined:
            indices = np.nonzero(mask_i)
            if len(indices[0]) > 0 and len(indices[1]) > 0:
                bbox = (np.min(indices[1]), np.min(indices[0]), np.max(indices[1]), np.max(indices[0]))
                crop_region = make_crop_region(mask_i.shape[1], mask_i.shape[0], bbox, crop_factor)
                x1, y1, x2, y2 = crop_region

                if x2 - x1 > 0 and y2 - y1 > 0:
                    cropped_mask = mask_i[y1:y2, x1:x2]

                    if bbox_fill:
                        bx1, by1, bx2, by2 = bbox
                        cropped_mask = cropped_mask.copy()
                        cropped_mask[by1:by2, bx1:bx2] = 1.0

                    result.append(SEG(None, cropped_mask, 1.0, crop_region, bbox, label, None))
        else:
            mask_i_uint8 = (mask_i * 255.0).astype(np.uint8)
            contours, ctree = cv2.findContours(mask_i_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for j, contour in enumerate(contours):
                hierarchy = ctree[0][j]
                if hierarchy[3] != -1:
                    continue  # a hole inside another contour

                separated_mask = np.zeros_like(mask_i_uint8)
                cv2.drawContours(separated_mask, [contour], 0, 255, -1)
                separated_mask = np.array(separated_mask / 255.0).astype(np.float32)

                x, y, w, h = cv2.boundingRect(contour)
                bbox = x, y, x + w, y + h
                crop_region = make_crop_region(mask_i.shape[1], mask_i.shape[0], bbox, crop_factor, crop_min_size)

                if w > drop_size and h > drop_size:
                    mask_src = separated_mask if is_contour else mask_i * separated_mask

                    cropped_mask = np.array(mask_src[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]])

                    if bbox_fill:
                        cx1, cy1, _, _ = crop_region
                        cropped_mask[y - cy1:y + h - cy1, x - cx1:x + w - cx1] = 1.0

                    cropped_mask = np.clip(cropped_mask, 0, 1.0)
                    result.append(SEG(None, cropped_mask, 1.0, crop_region, bbox, label, None))

    if not result:
        logging.info("[MpiDetailer] mask_to_segs: empty mask")
    logging.info(f"[MpiDetailer] # of detected SEGS: {len(result)}")

    return (mask.shape[1], mask.shape[2]), result


def label_filter(segs, labels):
    """Keep SEGS whose label is in `labels` ('all' keeps everything). Returns (kept, remained)."""
    labels = set(label.strip() for label in labels)

    if "all" in labels:
        return segs, (segs[0], [])

    kept, remained = [], []
    for x in segs[1]:
        if x.label in labels:
            kept.append(x)
        elif "eyes" in labels and x.label in ["left_eye", "right_eye"]:
            kept.append(x)
        elif "eyebrows" in labels and x.label in ["left_eyebrow", "right_eyebrow"]:
            kept.append(x)
        elif "pupils" in labels and x.label in ["left_pupil", "right_pupil"]:
            kept.append(x)
        else:
            remained.append(x)
    return (segs[0], kept), (segs[0], remained)


def label_assign(segs, labels):
    """Give the SEGS the labels in order; extra SEGS keep their label, extra labels are ignored."""
    labels = [label.strip() for label in labels]

    if len(labels) != len(segs[1]):
        logging.warning(f"[MpiDetailer] label assign: {len(labels)} labels for {len(segs[1])} segs")

    labeled = []
    for idx, x in enumerate(segs[1]):
        if len(labels) > idx:
            x = x._replace(label=labels[idx])
        labeled.append(x)
    return (segs[0], labeled)


# --- detections -> SEGS (YOLO bbox detector) --------------------------------------------------

def rect_masks_from_bboxes(bboxes, height, width):
    """One filled-rectangle bool mask per xyxy box, like cv2.rectangle(..., 255, -1) on a blank image."""
    segms = []
    for x0, y0, x1, y1 in bboxes:
        m = np.zeros((height, width), np.uint8)
        cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        segms.append(m.astype(bool))
    return segms


def detections_to_segs(image, detections, dilation, crop_factor, drop_size=1):
    """Detections [(label, bbox_xyxy, confidence), ...] on an NHWC image -> SEGS of box masks.

    This is Impact-Subpack's UltraBBoxDetector.detect after the YOLO call: every box becomes a filled
    rectangle mask, dilated by `dilation`, with a crop region of `crop_factor` x the box.
    """
    drop_size = max(drop_size, 1)
    h = image.shape[1]
    w = image.shape[2]

    segmasks, labels = [], []
    bboxes = [d[1] for d in detections]
    for (label, bbox, confidence), seg_mask in zip(detections, rect_masks_from_bboxes(bboxes, h, w)):
        segmasks.append((bbox, seg_mask.astype(np.float32), confidence))
        labels.append(label)

    if dilation > 0:
        segmasks = dilate_masks(segmasks, dilation)

    items = []
    for (item_bbox, item_mask, confidence), label in zip(segmasks, labels):
        y1, x1, y2, x2 = item_bbox  # names as upstream; both sides of the test below are one axis each

        if x2 - x1 > drop_size and y2 - y1 > drop_size:
            crop_region = make_crop_region(w, h, item_bbox, crop_factor)
            cropped_image = crop_ndarray4(image, crop_region)
            cropped_mask = crop_ndarray2(item_mask, crop_region)
            items.append(SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None))

    return (image.shape[1], image.shape[2]), items


# --- tile SEGS --------------------------------------------------------------------------------

def random_mask_raw(mask, bbox, factor):
    """Draw an irregular blob outline-filled rectangle into `mask` (rows = axis 0), upstream's algorithm.

    Upstream runs 16 threads over the line segments; here they run one after the other. The pixels
    drawn per circle are the same, only the order of the random radii differs from a threaded run
    (which is unordered anyway).
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    factor = max(6, int(min(w, h) * factor / 4))

    def draw_random_circle(center, radius):
        i, j = center
        xs = np.arange(int(i - radius), int(i + radius))[:, None]
        ys = np.arange(int(j - radius), int(j + radius))[None, :]
        inside = (xs - i) ** 2 + (ys - j) ** 2 <= radius ** 2
        xs_all = np.broadcast_to(xs, inside.shape)[inside]
        ys_all = np.broadcast_to(ys, inside.shape)[inside]
        mask[xs_all, ys_all] = 1

    def draw_irregular_line(start, end, pivot, is_vertical):
        i = start
        while i < end:
            radius = int(np.random.randint(5, factor))
            if is_vertical:
                draw_random_circle((i, pivot), radius)
            else:
                draw_random_circle((pivot, i), radius)
            i += radius

    def draw_irregular_line_parallel(start, end, pivot, is_vertical):
        step = (end - start) // 16
        for i in range(start, end, step):
            draw_irregular_line(i, min(i + step, end), pivot, is_vertical)

    draw_irregular_line_parallel(y1 + factor, y2 - factor, x1 + factor, True)
    draw_irregular_line_parallel(y1 + factor, y2 - factor, x2 - factor, True)
    draw_irregular_line_parallel(x1 + factor, x2 - factor, y1 + factor, False)
    draw_irregular_line_parallel(x1 + factor, x2 - factor, y2 - factor, False)

    mask[y1 + factor:y2 - factor, x1 + factor:x2 - factor] = 1.0


def random_mask(mask, bbox, factor, size=128):
    small_mask = np.zeros((size, size)).astype(np.float32)
    random_mask_raw(small_mask, (0, 0, size, size), factor)

    x1, y1, x2, y2 = bbox
    small_mask = torch.tensor(small_mask).unsqueeze(0).unsqueeze(0)
    bbox_mask = F.interpolate(small_mask, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
    mask[y1:y2, x1:x2] = bbox_mask.squeeze(0).squeeze(0)


def adaptive_mask_paste(dest_mask, src_mask, bbox):
    x1, y1, x2, y2 = bbox
    small_mask = torch.tensor(src_mask).unsqueeze(0).unsqueeze(0)
    bbox_mask = F.interpolate(small_mask, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
    dest_mask[y1:y2, x1:x2] = bbox_mask.squeeze(0).squeeze(0)


def make_tile_segs(images, bbox_size, crop_factor, min_overlap, filter_segs_dilation, mask_irregularity=0,
                   irregular_mask_mode="Reuse fast", filter_in_segs_opt=None, filter_out_segs_opt=None):
    """Cut an image into overlapping square tiles, one SEG each, in reading order (row by row)."""
    if bbox_size <= 2 * min_overlap:
        new_min_overlap = bbox_size / 2
        logging.info(f"[MpiDetailer] tile: min_overlap too large ({min_overlap} => {new_min_overlap})")
        min_overlap = new_min_overlap

    _, ih, iw, _ = images.size()

    mask_cache = None
    mask_quality = 512
    if mask_irregularity > 0:
        if irregular_mask_mode == "Reuse fast":
            mask_quality = 128
            mask_cache = np.zeros((128, 128)).astype(np.float32)
            random_mask(mask_cache, (0, 0, 128, 128), factor=mask_irregularity, size=mask_quality)
        elif irregular_mask_mode == "Reuse quality":
            mask_quality = 512
            mask_cache = np.zeros((512, 512)).astype(np.float32)
            random_mask(mask_cache, (0, 0, 512, 512), factor=mask_irregularity, size=mask_quality)
        elif irregular_mask_mode == "All random fast":
            mask_quality = 512

    # compensate overlap/bbox_size for the irregular mask
    if mask_irregularity > 0:
        compensate = max(6, int(mask_quality * mask_irregularity / 4))
        min_overlap += compensate
        bbox_size += compensate * 2

    if filter_out_segs_opt is not None:
        exclusion_mask = segs_to_combined_mask(filter_out_segs_opt)
        exclusion_mask = make_3d_mask(exclusion_mask)
        exclusion_mask = resize_mask(exclusion_mask, (ih, iw))
        exclusion_mask = dilate_mask(exclusion_mask.cpu().numpy(), filter_segs_dilation)
    else:
        exclusion_mask = None

    if filter_in_segs_opt is not None:
        and_mask = segs_to_combined_mask(filter_in_segs_opt)
        and_mask = make_3d_mask(and_mask)
        and_mask = resize_mask(and_mask, (ih, iw))
        and_mask = dilate_mask(and_mask.cpu().numpy(), filter_segs_dilation)

        a, b = mask_to_segs(and_mask, True, 1.0, False, 0)
        if len(b) == 0:
            return (a, b)

        start_x, start_y, c, d = b[0].crop_region
        w = c - start_x
        h = d - start_y
    else:
        start_x = 0
        start_y = 0
        h, w = ih, iw
        and_mask = None

    if bbox_size > h or bbox_size > w:
        new_bbox_size = min(bbox_size, min(w, h))
        logging.info(f"[MpiDetailer] tile: bbox_size larger than the image ({bbox_size} => {new_bbox_size})")
        bbox_size = new_bbox_size

    n_horizontal = math.ceil(w / (bbox_size - min_overlap))
    n_vertical = math.ceil(h / (bbox_size - min_overlap))

    w_overlap_sum = (bbox_size * n_horizontal) - w
    if w_overlap_sum < 0:
        n_horizontal += 1
        w_overlap_sum = (bbox_size * n_horizontal) - w
    w_overlap_size = 0 if n_horizontal == 1 else int(w_overlap_sum / (n_horizontal - 1))

    h_overlap_sum = (bbox_size * n_vertical) - h
    if h_overlap_sum < 0:
        n_vertical += 1
        h_overlap_sum = (bbox_size * n_vertical) - h
    h_overlap_size = 0 if n_vertical == 1 else int(h_overlap_sum / (n_vertical - 1))

    new_segs = []

    if w_overlap_size == bbox_size:
        n_horizontal = 1
    if h_overlap_size == bbox_size:
        n_vertical = 1

    y = start_y
    for j in range(0, n_vertical):
        x = start_x
        for i in range(0, n_horizontal):
            x1 = x
            y1 = y

            if x + bbox_size < iw - 1:
                x2 = x + bbox_size
            else:
                x2 = iw
                x1 = iw - bbox_size

            if y + bbox_size < ih - 1:
                y2 = y + bbox_size
            else:
                y2 = ih
                y1 = ih - bbox_size

            bbox = x1, y1, x2, y2
            crop_region = make_crop_region(iw, ih, bbox, crop_factor)
            cx1, cy1, cx2, cy2 = crop_region

            mask = np.zeros((cy2 - cy1, cx2 - cx1)).astype(np.float32)

            rel_left = x1 - cx1
            rel_top = y1 - cy1
            rel_right = x2 - cx1
            rel_bot = y2 - cy1

            if mask_irregularity > 0:
                if mask_cache is not None:
                    adaptive_mask_paste(mask, mask_cache, (rel_left, rel_top, rel_right, rel_bot))
                else:
                    random_mask(mask, (rel_left, rel_top, rel_right, rel_bot), factor=mask_irregularity,
                                size=mask_quality)

                # corner filling
                if rel_left == 0:
                    pad = int((x2 - x1) / 8)
                    mask[rel_top:rel_bot, :pad] = 1.0
                if rel_top == 0:
                    pad = int((y2 - y1) / 8)
                    mask[:pad, rel_left:rel_right] = 1.0
                if rel_right == mask.shape[1]:
                    pad = int((x2 - x1) / 8)
                    mask[rel_top:rel_bot, -pad:] = 1.0
                if rel_bot == mask.shape[0]:
                    pad = int((y2 - y1) / 8)
                    mask[-pad:, rel_left:rel_right] = 1.0
            else:
                mask[rel_top:rel_bot, rel_left:rel_right] = 1.0

            mask = torch.tensor(mask)

            if exclusion_mask is not None:
                mask[exclusion_mask[cy1:cy2, cx1:cx2] != 0] = 0.0

            if and_mask is not None:
                mask[and_mask[cy1:cy2, cx1:cx2] == 0] = 0.0

            if not torch.all(mask == 0.0).item():
                new_segs.append(SEG(None, mask.numpy(), 1.0, crop_region, bbox, "", None))

            x += bbox_size - w_overlap_size
        y += bbox_size - h_overlap_size

    return (ih, iw), new_segs


# --- detailer wildcard text (per-SEG prompt) --------------------------------------------------

_LAB_PATTERN = r"\[([A-Za-z0-9_. ]+)\]([^\[]+)(?=\[|$)"


def split_wildcard_labels(text):
    """'[ALL]a[t0]b' -> {'ALL': 'a', 't0': 'b'} (Impact's [LAB] syntax)."""
    return {key: value.strip() for key, value in re.findall(_LAB_PATTERN, text)}


class WildcardChooser:
    """The text a SEG gets: one plain text for every SEG, or the [LAB] entry of its label."""

    def __init__(self, text=None, items=None):
        self.text = text
        self.items = items

    def get(self, seg):
        if self.items is None:
            return self.text
        text = ""
        if "ALL" in self.items:
            text = self.items["ALL"]
        if seg.label in self.items:
            text += self.items[seg.label]
        return text


def process_wildcard_for_segs(wildcard):
    """-> (mode, chooser). Modes: 'LAB' (per label) or None (one text for all). Impact's ordered
    modes ([ASC] [DSC] [ASC-SIZE] [DSC-SIZE] [RND] with [SEP]) are not ported: nothing in Cubric Studio sends them."""
    if wildcard.startswith("[LAB]"):
        items = {}
        for k, v in split_wildcard_labels(wildcard).items():
            v = v.strip()
            if v != "":
                items[k] = v
        return "LAB", WildcardChooser(items=items)

    if re.match(r"\[(ASC-SIZE|DSC-SIZE|ASC|DSC|RND)\]", wildcard):
        raise ValueError("MpiDetailer: the ordered wildcard modes [ASC] [DSC] [ASC-SIZE] [DSC-SIZE] [RND] are not supported; use [LAB]")
    return None, WildcardChooser(text=wildcard)


def strip_comment_lines(text):
    """Impact's wildcard comment-out: a line starting with '#' is dropped and the next line is appended to the previous."""
    lines0 = []
    flag = False
    for line in text.split("\n"):
        if line.lstrip().startswith("#"):
            flag = True
            continue

        if len(lines0) == 0:
            lines0.append(line)
        elif flag:
            lines0[-1] += " " + line
            flag = False
        else:
            lines0.append(line)
    return "\n".join(lines0)


def split_break(text):
    """'a BREAK b' -> ['a', 'b'] (blank parts dropped; '' when nothing is left)."""
    parts = [x.strip() for x in text.split("BREAK")]
    parts = [x for x in parts if x != ""]
    return parts if parts else [""]
