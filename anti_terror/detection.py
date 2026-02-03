"""Object detection module with optimized bag detection.

Uses YOLO11x (latest, most accurate) with separate confidence thresholds
for persons and bags to improve bag detection.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from loguru import logger
from ultralytics import YOLO

from .config import DetectionConfig, select_device


@dataclass
class DetectionResult:
    boxes: np.ndarray  # (N,4) xyxy
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,)


class Detector:
    """YOLO-based object detector optimized for persons and bags.

    Key improvements for bag detection:
    1. Uses YOLO11x (most accurate model)
    2. Tuned confidence thresholds for production stability
    3. Multi-scale detection with augmentation
    4. Strict bag classes only (no false positives from umbrellas, etc.)
    """

    # COCO class IDs for bags ONLY - strict filtering
    # Removed: umbrella, tie, frisbee, skis, snowboard, sports ball, etc.
    # These were causing too many false positives
    BAG_CLASSES = {
        24: "backpack",
        26: "handbag",
        28: "suitcase",
    }

    def __init__(self, cfg: DetectionConfig):
        device = select_device(cfg.device)
        model_path = cfg.model_path

        logger.info(f"Loading YOLO model {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.cfg = cfg

        # Separate thresholds
        self.person_conf = cfg.conf_threshold
        self.bag_conf = cfg.bag_conf_threshold  # Lower threshold for bags

        logger.info(f"Detection thresholds - Person: {self.person_conf}, Bag: {self.bag_conf}")

    def __call__(self, frame: np.ndarray) -> DetectionResult:
        """Detect objects in frame with optimized bag detection."""

        # Run detection with lower threshold to catch more bags
        min_conf = min(self.person_conf, self.bag_conf)

        results = self.model.predict(
            frame,
            conf=min_conf,
            iou=self.cfg.iou_threshold,
            device=self.model.device,
            verbose=False,
            imgsz=self.cfg.imgsz,  # Larger input = better small object detection
            augment=self.cfg.augment,  # Test-time augmentation for better accuracy
        )[0]

        if len(results.boxes) == 0:
            return DetectionResult(
                boxes=np.array([]).reshape(0, 4),
                scores=np.array([]),
                classes=np.array([])
            )

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # Filter by class-specific thresholds
        keep_mask = np.zeros(len(boxes), dtype=bool)

        for i, (score, cls) in enumerate(zip(scores, classes)):
            if cls in self.cfg.classes_person:
                # Person class - use person threshold
                if score >= self.person_conf:
                    keep_mask[i] = True
            elif cls in self.cfg.classes_bag:
                # Bag class - use lower bag threshold
                if score >= self.bag_conf:
                    keep_mask[i] = True

        return DetectionResult(
            boxes=boxes[keep_mask],
            scores=scores[keep_mask],
            classes=classes[keep_mask]
        )
