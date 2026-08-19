"""End-to-End Vision Pipeline for ITMS railway track monitoring.

Orchestrates frame validation, deterministic track bed ROI extraction, obstacle detection,
and placeholder hooks for vegetation and track geometry assessments into a unified
VisionResult schema.
"""

from typing import Any, List, Optional
import numpy as np

from ai_pipeline.detection.obstacle_detector import ObstacleDetector
from ai_pipeline.detection.track_roi import TrackROIExtractor
from ai_pipeline.models.vision_events import (
    DetectionEvent,
    GeometryAssessment,
    VegetationAssessment,
    VisionResult,
)


class VisionPipeline:
    """Orchestrates computer vision inference across track ROI for a single frame."""

    def __init__(
        self,
        obstacle_detector: Optional[ObstacleDetector] = None,
        roi_extractor: Optional[TrackROIExtractor] = None,
        vegetation_assessor: Optional[Any] = None,
        geometry_assessor: Optional[Any] = None,
    ) -> None:
        """Initializes the VisionPipeline with dependency-injectable submodules.

        Args:
            obstacle_detector: Pre-configured or mock ObstacleDetector instance.
            roi_extractor: Pre-configured TrackROIExtractor instance.
            vegetation_assessor: Optional callable/module for vegetation analysis (Stage 8 placeholder).
            geometry_assessor: Optional callable/module for track geometry analysis (Stage 8 placeholder).
        """
        self.roi_extractor = roi_extractor or TrackROIExtractor()
        self.obstacle_detector = obstacle_detector or ObstacleDetector(
            roi_extractor=self.roi_extractor
        )
        self.vegetation_assessor = vegetation_assessor
        self.geometry_assessor = geometry_assessor

    def _assess_vegetation(
        self, image: np.ndarray, frame_id: int, timestamp: float
    ) -> Optional[VegetationAssessment]:
        """Placeholder hook for vegetation assessment module."""
        if self.vegetation_assessor is not None:
            if callable(self.vegetation_assessor):
                return self.vegetation_assessor(image, frame_id=frame_id, timestamp=timestamp)
            if hasattr(self.vegetation_assessor, "assess"):
                return self.vegetation_assessor.assess(image, frame_id=frame_id, timestamp=timestamp)
        return None

    def _assess_geometry(
        self, image: np.ndarray, frame_id: int, timestamp: float
    ) -> Optional[GeometryAssessment]:
        """Placeholder hook for track geometry assessment module."""
        if self.geometry_assessor is not None:
            if callable(self.geometry_assessor):
                return self.geometry_assessor(image, frame_id=frame_id, timestamp=timestamp)
            if hasattr(self.geometry_assessor, "assess"):
                return self.geometry_assessor.assess(image, frame_id=frame_id, timestamp=timestamp)
        return None

    def process_frame(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        source_frame: Optional[int] = None,
        timestamp: float = 0.0,
    ) -> VisionResult:
        """Executes full vision pipeline on a single frame.

        Args:
            image: Full-frame BGR image array (e.g. 1920x1080).
            frame_id: Sequential extracted frame index.
            source_frame: Source video frame index (defaults to frame_id if None).
            timestamp: Video timestamp in seconds.

        Returns:
            VisionResult containing detections and assessments.
        """
        # Validate image array
        if image is None:
            raise ValueError("Input image cannot be None.")
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Input image must be a numpy ndarray, got {type(image)}")
        if image.size == 0 or len(image.shape) < 2 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Input image array is empty or has invalid shape.")

        # Validate indexing
        if frame_id < 0:
            raise ValueError(f"frame_id must be non-negative, got {frame_id}")
        src_frame = source_frame if source_frame is not None else frame_id
        if src_frame < 0:
            raise ValueError(f"source_frame must be non-negative, got {src_frame}")
        if timestamp < 0.0:
            raise ValueError(f"timestamp must be non-negative, got {timestamp}")

        # 1. Run Obstacle / Defect Detection
        detections: List[DetectionEvent] = self.obstacle_detector.detect(
            image=image,
            frame_id=frame_id,
            timestamp=timestamp,
        )

        # 2. Run Vegetation Assessment (Hook)
        vegetation = self._assess_vegetation(image, frame_id, timestamp)

        # 3. Run Geometry Assessment (Hook)
        geometry = self._assess_geometry(image, frame_id, timestamp)

        # 4. Construct composite result
        return VisionResult(
            frame_id=frame_id,
            source_frame=src_frame,
            timestamp=timestamp,
            detections=detections,
            vegetation=vegetation,
            geometry=geometry,
        )
