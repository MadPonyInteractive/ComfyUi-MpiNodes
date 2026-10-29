# The detailer chain as ComfyUI nodes: YOLO face/person detect -> SEGS -> per-SEG inpaint -> paste, the
# tile-upscale SEGS, and the SAM-refined auto mask. 16 nodes, ported from ComfyUI-Impact-Pack (GPL-3.0)
# and ComfyUI-Impact-Subpack (AGPL-3.0); see NOTICE.md next to this file.
#
# Type strings (SEGS, BASIC_PIPE, BBOX_DETECTOR, SEGM_DETECTOR, SAM_MODEL) are the originals', so the
# slots keep their colours; the VALUES are this port's own objects, so a graph must use the Mpi nodes
# throughout (Impact's SAM_MODEL / BBOX_DETECTOR objects do not work here and vice versa).
#
# comfy / folder_paths / nodes are imported inside the methods, so importing this module needs no ComfyUI.
import os

import torch

from . import segs as S
from .detail import do_detail
from .sam import load_sam, make_sam_mask_center
from .yolo import UltralyticsBBoxDetector, load_yolo

MAX_RESOLUTION = 16384  # nodes.MAX_RESOLUTION
CATEGORY = "MpiNodes/Detailer"

_MODEL_FOLDERS = (
    ("sams", "sams"),
    ("ultralytics", "ultralytics"),
    ("ultralytics_bbox", os.path.join("ultralytics", "bbox")),
    ("ultralytics_segm", os.path.join("ultralytics", "segm")),
)


def _ensure_model_folders():
    """Register models/sams and models/ultralytics[/bbox|/segm] (what Impact / its Subpack did at import)."""
    import folder_paths

    for name, rel in _MODEL_FOLDERS:
        folder_paths.add_model_folder_path(name, os.path.join(folder_paths.models_dir, rel))
        paths, extensions = folder_paths.folder_names_and_paths[name]
        folder_paths.folder_names_and_paths[name] = (paths, extensions | folder_paths.supported_pt_extensions)


def _samplers():
    import comfy.samplers

    return comfy.samplers.KSampler.SAMPLERS


def _schedulers():
    import comfy.samplers

    return list(comfy.samplers.SCHEDULER_HANDLERS)


# --- models -----------------------------------------------------------------------------------

class NoSegmDetector:
    """The SEGM_DETECTOR slot of a bbox model. Nothing here consumes it."""


class MpiUltralyticsDetectorProvider:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        _ensure_model_folders()
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        return {"required": {"model_name": (bboxs,)}}

    RETURN_TYPES = ("BBOX_DETECTOR", "SEGM_DETECTOR")
    FUNCTION = "doit"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load a YOLO detector from models/ultralytics/bbox (needs the ultralytics package)."

    def doit(self, model_name):
        import folder_paths

        _ensure_model_folders()
        model_path = folder_paths.get_full_path("ultralytics", model_name)
        if model_path is None and model_name.startswith("bbox/"):
            model_path = folder_paths.get_full_path("ultralytics_bbox", model_name[5:])

        if model_path is None:
            raise ValueError(f"[MpiDetailer] model file '{model_name}' not found in "
                             f"{folder_paths.get_folder_paths('ultralytics')}")

        return UltralyticsBBoxDetector(load_yolo(model_path)), NoSegmDetector()


class MpiSAMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        _ensure_model_folders()
        models = [x for x in folder_paths.get_filename_list("sams")
                  if "hq" not in x and "sam2" not in x and x.endswith((".pt", ".pth", ".safetensors"))]
        return {"required": {
            "model_name": (models, {"tooltip": "Segment Anything checkpoint from models/sams (vit_b, vit_l or vit_h; SAM 2 and HQ are not supported)."}),
            "device_mode": (["AUTO", "Prefer GPU", "CPU"], {"tooltip": "AUTO: on the GPU only while it predicts. Prefer GPU: keep it there if it fits. CPU: always CPU."}),
        }}

    RETURN_TYPES = ("SAM_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load a Segment Anything model for the Simple Detector."

    def load_model(self, model_name, device_mode="AUTO"):
        import folder_paths

        path = folder_paths.get_full_path("sams", model_name)
        return (load_sam(path, model_name, device_mode),)


# --- pipes ------------------------------------------------------------------------------------

class MpiToBasicPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
        }}

    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("basic_pipe",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(self, model, clip, vae, positive, negative):
        return ((model, clip, vae, positive, negative),)


class MpiFromBasicPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"basic_pipe": ("BASIC_PIPE",)}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "clip", "vae", "positive", "negative")
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(self, basic_pipe):
        model, clip, vae, positive, negative = basic_pipe
        return model, clip, vae, positive, negative


# --- detect -----------------------------------------------------------------------------------

class MpiBboxDetectorSEGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "bbox_detector": ("BBOX_DETECTOR",),
            "image": ("IMAGE",),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
            "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
            "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
            "labels": ("STRING", {"multiline": True, "default": "all", "placeholder": "List the types of segments to be allowed, separated by commas"}),
        }}

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(self, bbox_detector, image, threshold, dilation, crop_factor, drop_size, labels=None):
        if len(image) > 1:
            raise Exception("[MpiDetailer] the bbox detector does not take image batches; run the images one at a time")

        segs = bbox_detector.detect(image, threshold, dilation, crop_factor, drop_size)

        if labels is not None and labels != "":
            segs, _ = S.label_filter(segs, labels.split(","))

        return (segs,)


class MpiSimpleDetectorSEGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_detector": ("BBOX_DETECTOR",),
                "image": ("IMAGE",),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                "sub_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sub_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sub_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "post_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sam_model_opt": ("SAM_MODEL", {"tooltip": "Refines each detected box into a silhouette with SAM (centre point prompt)."}),
            },
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(self, bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size, sub_threshold,
             sub_dilation, sub_bbox_expansion, sam_mask_hint_threshold, post_dilation=0, sam_model_opt=None):
        if len(image) > 1:
            raise Exception("[MpiDetailer] the simple detector does not take image batches; run the images one at a time")

        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, crop_factor, drop_size)

        if sam_model_opt is not None:
            mask = make_sam_mask_center(sam_model_opt, segs, image, sub_dilation, sub_threshold, sub_bbox_expansion)
            segs = S.segs_bitwise_and_mask(segs, mask)

        return (S.dilate_segs(segs, post_dilation),)


class MpiMaskToSEGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "combined": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
            "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
            "bbox_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
            "contour_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
        }}

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(self, mask, combined, crop_factor, bbox_fill, drop_size, contour_fill=False):
        mask = S.make_2d_mask(mask)
        return (S.mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size, is_contour=contour_fill),)


class MpiMakeTileSEGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "bbox_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.01}),
                "min_overlap": ("INT", {"default": 5, "min": 0, "max": 512, "step": 1}),
                "filter_segs_dilation": ("INT", {"default": 20, "min": -255, "max": 255, "step": 1}),
                "mask_irregularity": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                "irregular_mask_mode": (["Reuse fast", "Reuse quality", "All random fast", "All random quality"],),
            },
            "optional": {
                "filter_in_segs_opt": ("SEGS",),
                "filter_out_segs_opt": ("SEGS",),
            },
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(self, images, bbox_size, crop_factor, min_overlap, filter_segs_dilation, mask_irregularity=0,
             irregular_mask_mode="Reuse fast", filter_in_segs_opt=None, filter_out_segs_opt=None):
        return (S.make_tile_segs(images, bbox_size, crop_factor, min_overlap, filter_segs_dilation, mask_irregularity,
                                 irregular_mask_mode, filter_in_segs_opt, filter_out_segs_opt),)


# --- SEGS tools -------------------------------------------------------------------------------

class MpiSEGSLabelAssign:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "segs": ("SEGS",),
            "labels": ("STRING", {"multiline": True, "placeholder": "List the label to be assigned in order of segs, separated by commas"}),
        }}

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(self, segs, labels):
        return (S.label_assign(segs, labels.split(",")),)


class MpiSegsAndMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segs": ("SEGS",), "mask": ("MASK",)}}

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(self, segs, mask):
        return (S.segs_bitwise_and_mask(segs, mask),)


class MpiSegsToCombinedMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segs": ("SEGS",)}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(self, segs):
        return (S.make_3d_mask(S.segs_to_combined_mask(segs)),)


class MpiSEGSToMaskList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segs": ("SEGS",)}}

    RETURN_TYPES = ("MASK",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(self, segs):
        masks = S.segs_to_masklist(segs)
        return ([S.make_3d_mask(mask) for mask in masks],)


class MpiDilateMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"
    CATEGORY = CATEGORY
    DESCRIPTION = "Grow (positive) or shrink (negative) a mask with a square kernel."

    def doit(self, mask, dilation):
        mask = S.dilate_mask(mask.numpy(), dilation)
        return (S.make_3d_mask(torch.from_numpy(mask)),)


class MpiSEGSPreview:
    def __init__(self):
        import folder_paths

        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "alpha_mode": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "min_alpha": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"fallback_image_opt": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def doit(self, segs, alpha_mode=True, min_alpha=0.0, fallback_image_opt=None):
        import folder_paths
        import numpy as np
        from PIL import Image

        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path("impact_seg_preview", self.output_dir, segs[0][1], segs[0][0])

        results = []
        result_image_list = []

        if fallback_image_opt is not None:
            segs = S.segs_scale_match(segs, fallback_image_opt.shape)

        if min_alpha != 0:
            min_alpha = int(255 * min_alpha)

        if len(segs[1]) > 0:
            if segs[1][0].cropped_image is not None:
                batch_count = len(segs[1][0].cropped_image)
            elif fallback_image_opt is not None:
                batch_count = len(fallback_image_opt)
            else:
                return {"ui": {"images": results}}

            for seg in segs[1]:
                result_image_batch = None
                cached_mask = None

                def get_combined_mask():
                    nonlocal cached_mask
                    if cached_mask is not None:
                        return cached_mask
                    masks = torch.tensor(seg.cropped_mask) if isinstance(seg.cropped_mask, np.ndarray) else seg.cropped_mask
                    cached_mask = (masks[0] * 255).to(torch.uint8)
                    for x in masks[1:]:
                        cached_mask |= (x * 255).to(torch.uint8)
                    cached_mask = (cached_mask / 255.0).to(torch.float32)
                    cached_mask = _to_binary_mask(cached_mask, 0.1).numpy()
                    return cached_mask

                def stack_image(image, mask=None):
                    nonlocal result_image_batch
                    if isinstance(image, np.ndarray):
                        image = torch.from_numpy(image)
                    if mask is not None:
                        image *= torch.tensor(mask)[None, ..., None]
                    result_image_batch = image if result_image_batch is None else torch.concat((result_image_batch, image), dim=0)

                for i in range(batch_count):
                    cropped_image = None

                    if seg.cropped_image is not None:
                        cropped_image = seg.cropped_image[i, None]
                    elif fallback_image_opt is not None:
                        # take from the original image
                        cropped_image = S.crop_ndarray4(fallback_image_opt[i].unsqueeze(0), seg.crop_region)

                    if cropped_image is not None:
                        if isinstance(cropped_image, np.ndarray):
                            cropped_image = torch.from_numpy(cropped_image)

                        cropped_image = cropped_image.clone()
                        cropped_pil = S.tensor2pil(cropped_image)

                        if alpha_mode:
                            if isinstance(seg.cropped_mask, np.ndarray):
                                cropped_mask = seg.cropped_mask
                            elif seg.cropped_image is not None and len(seg.cropped_image) != len(seg.cropped_mask):
                                cropped_mask = get_combined_mask()
                            else:
                                cropped_mask = seg.cropped_mask[i].numpy()

                            mask_array = (cropped_mask * 255).astype(np.uint8)
                            if min_alpha != 0:
                                mask_array[mask_array < min_alpha] = min_alpha

                            mask_pil = Image.fromarray(mask_array).resize(cropped_pil.size)
                            cropped_pil.putalpha(mask_pil)
                            stack_image(cropped_image, cropped_mask)
                        else:
                            stack_image(cropped_image)

                        file = f"{filename}_{counter:05}_.webp"
                        cropped_pil.save(os.path.join(full_output_folder, file))
                        results.append({"filename": file, "subfolder": subfolder, "type": self.type})
                        counter += 1

                if result_image_batch is not None:
                    result_image_list.append(result_image_batch)

        return {"ui": {"images": results}, "result": (result_image_list,)}


def _to_binary_mask(mask, threshold=0):
    mask = S.make_3d_mask(mask).clone().cpu()
    mask[mask > threshold] = 1.0
    mask[mask <= threshold] = 0.0
    return mask


# --- detailers --------------------------------------------------------------------------------

class MpiDetailerForEachPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (_samplers(),),
                "scheduler": (_schedulers(),),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "basic_pipe": ("BASIC_PIPE",),
                "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "tiled_encode": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "tiled_decode": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "SEGS", "BASIC_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "segs", "basic_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "doit"
    CATEGORY = CATEGORY
    DESCRIPTION = "Inpaint each detected region (SEGS) after enlarging it to the guide size, then paste it back."

    def doit(self, image, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             denoise, feather, noise_mask, force_inpaint, basic_pipe, wildcard, refiner_ratio=None, cycle=1,
             inpaint_model=False, noise_mask_feather=0, tiled_encode=False, tiled_decode=False):
        model, clip, vae, positive, negative = basic_pipe

        enhanced_img, _, _, _, new_segs = do_detail(
            image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
            scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint, wildcard, cycle=cycle,
            inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, tiled_encode=tiled_encode,
            tiled_decode=tiled_decode)

        # ControlNet pictures: none here, so the fallback picture
        return enhanced_img, new_segs, basic_pipe, [S.empty_pil_tensor()]


class MpiMaskDetailerPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "basic_pipe": ("BASIC_PIPE",),
                "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "mask bbox", "label_off": "crop region"}),
                "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "mask_mode": ("BOOLEAN", {"default": True, "label_on": "masked only", "label_off": "whole"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (_samplers(),),
                "scheduler": (_schedulers(),),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "bbox_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "contour_fill": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "BASIC_PIPE", "BASIC_PIPE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "basic_pipe", "refiner_basic_pipe_opt")
    OUTPUT_IS_LIST = (False, True, True, False, False)
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(self, image, mask, basic_pipe, guide_size, guide_size_for, max_size, mask_mode, seed, steps, cfg,
             sampler_name, scheduler, denoise, feather, crop_factor, drop_size, refiner_ratio, batch_size, cycle=1,
             inpaint_model=False, noise_mask_feather=0, bbox_fill=False, contour_fill=True):
        if len(image) > 1:
            raise Exception("[MpiDetailer] the mask detailer does not take image batches; run the images one at a time")

        model, clip, vae, positive, negative = basic_pipe

        mask = S.make_2d_mask(mask)
        segs = S.mask_to_segs(mask, False, crop_factor, bbox_fill, drop_size, is_contour=contour_fill)

        enhanced_img_batch = None
        cropped_enhanced_list = []
        cropped_enhanced_alpha_list = []

        for i in range(batch_size):
            enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, _ = do_detail(
                image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed + i, steps, cfg,
                sampler_name, scheduler, positive, negative, denoise, feather, mask_mode, True, wildcard_opt=None,
                cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)

            enhanced_img_batch = enhanced_img if enhanced_img_batch is None else torch.cat((enhanced_img_batch, enhanced_img), dim=0)
            cropped_enhanced_list += cropped_enhanced
            cropped_enhanced_alpha_list += cropped_enhanced_alpha

        # set fallback image
        if len(cropped_enhanced_list) == 0:
            cropped_enhanced_list = [S.empty_pil_tensor()]
        if len(cropped_enhanced_alpha_list) == 0:
            cropped_enhanced_alpha_list = [S.empty_pil_tensor()]

        # the last output was Impact's refiner pipe; there is no refiner here
        return enhanced_img_batch, cropped_enhanced_list, cropped_enhanced_alpha_list, basic_pipe, None


NODE_CLASS_MAPPINGS = {
    "MpiBboxDetectorSEGS": MpiBboxDetectorSEGS,
    "MpiDetailerForEachPipe": MpiDetailerForEachPipe,
    "MpiFromBasicPipe": MpiFromBasicPipe,
    "MpiToBasicPipe": MpiToBasicPipe,
    "MpiDilateMask": MpiDilateMask,
    "MpiMakeTileSEGS": MpiMakeTileSEGS,
    "MpiSEGSLabelAssign": MpiSEGSLabelAssign,
    "MpiSEGSToMaskList": MpiSEGSToMaskList,
    "MpiSegsAndMask": MpiSegsAndMask,
    "MpiSimpleDetectorSEGS": MpiSimpleDetectorSEGS,
    "MpiMaskDetailerPipe": MpiMaskDetailerPipe,
    "MpiMaskToSEGS": MpiMaskToSEGS,
    "MpiSAMLoader": MpiSAMLoader,
    "MpiSEGSPreview": MpiSEGSPreview,
    "MpiSegsToCombinedMask": MpiSegsToCombinedMask,
    "MpiUltralyticsDetectorProvider": MpiUltralyticsDetectorProvider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiBboxDetectorSEGS": "Mpi Bbox Detector SEGS",
    "MpiDetailerForEachPipe": "Mpi Detailer For Each Pipe",
    "MpiFromBasicPipe": "Mpi From Basic Pipe",
    "MpiToBasicPipe": "Mpi To Basic Pipe",
    "MpiDilateMask": "Mpi Dilate Mask",
    "MpiMakeTileSEGS": "Mpi Make Tile SEGS",
    "MpiSEGSLabelAssign": "Mpi SEGS Label Assign",
    "MpiSEGSToMaskList": "Mpi SEGS To Mask List",
    "MpiSegsAndMask": "Mpi Segs And Mask",
    "MpiSimpleDetectorSEGS": "Mpi Simple Detector SEGS",
    "MpiMaskDetailerPipe": "Mpi Mask Detailer Pipe",
    "MpiMaskToSEGS": "Mpi Mask To SEGS",
    "MpiSAMLoader": "Mpi SAM Loader",
    "MpiSEGSPreview": "Mpi SEGS Preview",
    "MpiSegsToCombinedMask": "Mpi Segs To Combined Mask",
    "MpiUltralyticsDetectorProvider": "Mpi Ultralytics Detector Provider",
}
