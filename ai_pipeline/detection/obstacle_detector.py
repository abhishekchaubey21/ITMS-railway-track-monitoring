"""Railway track obstacle and hazard detection module for ITMS pipeline.

Wraps Ultralytics YOLO inference bounded strictly within the deterministic track ROI corridor,
translating detections back to full-frame coordinates and emitting validated VisionDetectionEvent
schemas for the downstream multimodal fusion engine.

NOTE: This module implements the obstacle inference pipeline foundation. It does NOT claim
certified railway-specific accuracy out-of-the-box without targeted fine-tuning on track hazards.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from ai_pipeline.detection.track_roi import TrackROIExtractor, TrackROIConfig
from ai_pipeline.models.vision_events import (
    DetectionEvent,
    SeverityLevel,
    VisionDetectionEvent,
)


def _get_default_device() -> str:
    """Detects available hardware accelerator (CUDA/CPU)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except (ImportError, Exception):
        return "cpu"


@dataclass
class ObstacleDetectorConfig:
    """Configuration options for ObstacleDetector."""
    model_path: str = "yolo26n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: Optional[str] = None
    to_global_coords: bool = True
    source_name: str = "yolo_obstacle_detector"


class ObstacleDetector:
    """Inference detector for railway track obstructions and foreign hazards."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
        roi_extractor: Optional[TrackROIExtractor] = None,
        model: Optional[Any] = None,
        to_global_coords: bool = True,
        source_name: str = "yolo_obstacle_detector",
    ) -> None:
        """Initializes the ObstacleDetector.

        Args:
            model_path: Path or identifier for the YOLO model weights (default: yolo26n.pt).
            confidence_threshold: Minimum confidence score to accept detection (0.0 - 1.0).
            iou_threshold: NMS IoU threshold (0.0 - 1.0).
            imgsz: Inference input dimension.
            device: Inference device ('cuda', 'cpu', or None for auto-detect).
            roi_extractor: Pre-configured TrackROIExtractor instance.
            model: Optional injected mock/preloaded YOLO model for testing and custom execution.
            to_global_coords: Whether to map ROI bounding boxes back to full image coordinates.
            source_name: Identifier recorded in DetectionEvent.source.
        """
        self.model_path = model_path
        self.confidence_threshold = max(0.0, min(confidence_threshold, 1.0))
        self.iou_threshold = max(0.0, min(iou_threshold, 1.0))
        self.imgsz = imgsz
        self.device = device if device is not None else _get_default_device()
        self.roi_extractor = roi_extractor or TrackROIExtractor()
        self.model = model
        self.to_global_coords = to_global_coords
        self.source_name = source_name

    def _load_model(self) -> Any:
        """Lazily loads the Ultralytics YOLO model if not injected."""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
            except ImportError as e:
                raise ImportError(
                    "Ultralytics is required for live YOLO inference. Install with 'pip install ultralytics' "
                    "or inject a mock model for unit testing."
                ) from e
        return self.model

    @staticmethod
    def _map_severity(confidence: float, class_name: str) -> SeverityLevel:
        """Map detection confidence and object class to severity."""
        name_lower = class_name.lower().strip()

        critical_classes = {
            "person",
            "human",
            "vehicle",
            "car",
            "truck",
            "motorcycle",
            "bicycle",
            "fallen_tree",
            "boulder",
        }

        if name_lower in critical_classes and confidence >= 0.60:
            return SeverityLevel.CRITICAL

        if confidence >= 0.80:
            return SeverityLevel.HIGH

        if confidence >= 0.50:
            return SeverityLevel.MEDIUM

        return SeverityLevel.LOW

    def detect(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> List[DetectionEvent]:
        """Runs obstacle inference on the track ROI of the supplied image.

        Args:
            image: Full-frame input BGR image (e.g. 1920x1080).
            frame_id: Frame index to preserve in detection events.
            timestamp: Video timestamp in seconds to preserve in detection events.

        Returns:
            List of validated DetectionEvent objects.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image passed to ObstacleDetector.detect()")

        if frame_id < 0:
            raise ValueError(f"frame_id must be non-negative, got {frame_id}")
        if timestamp < 0.0:
            raise ValueError(f"timestamp must be non-negative, got {timestamp}")

        h, w = image.shape[:2]

        # 1. Extract Track Bed ROI
        roi_crop = self.roi_extractor.crop(image)
        rh, rw = roi_crop.shape[:2]
        ymin, ymax, xmin, xmax = self.roi_extractor.get_pixel_bounds(h, w)

        # 2. Run Inference
        model = self._load_model()
        results = model.predict(
            source=roi_crop,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[DetectionEvent] = []
        if not results:
            return detections

        first_result = results[0]
        boxes = getattr(first_result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        # Extract coordinates, confidences, and class indices
        xyxy_data = boxes.xyxy
        conf_data = boxes.conf
        cls_data = boxes.cls

        # Convert tensors / arrays to standard python floats/ints
        if hasattr(xyxy_data, "cpu"):
            xyxy_data = xyxy_data.cpu().numpy()
        if hasattr(conf_data, "cpu"):
            conf_data = conf_data.cpu().numpy()
        if hasattr(cls_data, "cpu"):
            cls_data = cls_data.cpu().numpy()

        names = getattr(first_result, "names", getattr(model, "names", {}))

        for i in range(len(xyxy_data)):
            conf = float(conf_data[i])
            if conf < self.confidence_threshold:
                continue

            cls_id = int(cls_data[i])
            class_name = names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict) else str(cls_id)

            bx1, by1, bx2, by2 = [float(v) for v in xyxy_data[i]]

            # Clip box to ROI boundaries
            bx1 = max(0.0, min(bx1, float(rw)))
            by1 = max(0.0, min(by1, float(rh)))
            bx2 = max(bx1, min(bx2, float(rw)))
            by2 = max(by1, min(by2, float(rh)))

            # Reject zero-area or invalid boxes
            if (bx2 - bx1) < 1.0 or (by2 - by1) < 1.0:
                continue

            # Translate to full image coordinate space if requested
            if self.to_global_coords:
                gx1 = max(0.0, min(bx1 + xmin, float(w)))
                gy1 = max(0.0, min(by1 + ymin, float(h)))
                gx2 = max(gx1, min(bx2 + xmin, float(w)))
                gy2 = max(gy1, min(by2 + ymin, float(h)))
                final_bbox = [round(gx1, 2), round(gy1, 2), round(gx2, 2), round(gy2, 2)]
            else:
                final_bbox = [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)]

            severity = self._map_severity(conf, class_name)

            event = DetectionEvent(
                frame_id=frame_id,
                timestamp=timestamp,
                class_name=class_name,
                confidence=round(conf, 4),
                bbox=final_bbox,
                severity=severity,
                source=self.source_name,
            )
            detections.append(event)

        return detections
