# YOLO bbox detector for the detailer chain.
#
# Ported from ComfyUI-Impact-Subpack (AGPL-3.0: modules/subcore.py, load_yolo / inference_bbox /
# UltraBBoxDetector.detect). The detection itself is still run by the `ultralytics` pip package: the
# .pt files (face_yolov8n, hand_yolov8n, person_yolov8n-seg) hold whole ultralytics model objects, so
# loading them needs the ultralytics classes. `ultralytics` is imported inside load_yolo(), so this
# module loads without it.
#
# Upstream also runs a segm branch (UltraSegmDetector) for models under `segm/`; no workflow uses it
# and the person_yolov8n-seg model is loaded as a bbox model, so only boxes are read here.
from .segs import detections_to_segs, tensor2pil


def load_yolo(model_path):
    from ultralytics import YOLO

    return YOLO(model_path)


def inference_bbox(model, image, confidence=0.3):
    """PIL image -> [(label, xyxy box, confidence array), ...] in detection order."""
    pred = model(image, conf=confidence, device="", verbose=False)

    boxes = pred[0].boxes
    bboxes = boxes.xyxy.cpu().numpy()

    results = []
    for i in range(len(bboxes)):
        label = pred[0].names[int(boxes[i].cls.item())]
        results.append((label, bboxes[i], boxes[i].conf.cpu().numpy()))
    return results


class UltralyticsBBoxDetector:
    """The BBOX_DETECTOR value."""

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1):
        detections = inference_bbox(self.bbox_model, tensor2pil(image), threshold)
        return detections_to_segs(image, detections, dilation, crop_factor, drop_size)
