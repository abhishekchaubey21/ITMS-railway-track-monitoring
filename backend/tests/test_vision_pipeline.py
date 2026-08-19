"""Unit tests for End-to-End Vision Pipeline."""

from typing import List, Optional
import numpy as np
import pytest

from ai_pipeline.detection.obstacle_detector import ObstacleDetector
from ai_pipeline.detection.track_roi import TrackROIExtractor, TrackROIConfig
from ai_pipeline.detection.vision_pipeline import VisionPipeline
from ai_pipeline.models.vision_events import (
    DetectionEvent,
    GeometryAssessment,
    GeometryStatus,
    SeverityLevel,
    VegetationAssessment,
    VisionResult,
)


class MockObstacleDetector:
    """Mock ObstacleDetector returning predetermined detections."""

    def __init__(self, detections: Optional[List[DetectionEvent]] = None):
        self.mock_detections = detections

    def detect(self, image: np.ndarray, frame_id: int = 0, timestamp: float = 0.0) -> List[DetectionEvent]:
        if self.mock_detections is not None:
            return self.mock_detections
        return [
            DetectionEvent(
                frame_id=frame_id,
                timestamp=timestamp,
                class_name="rock_debris",
                confidence=0.89,
                bbox=[200.0, 520.0, 320.0, 640.0],
                severity=SeverityLevel.HIGH,
                source="mock_obstacle_detector",
            )
        ]


def test_vision_pipeline_initialization():
    """Verify default and customized initialization of VisionPipeline."""
    pipeline = VisionPipeline()
    assert isinstance(pipeline.roi_extractor, TrackROIExtractor)
    assert isinstance(pipeline.obstacle_detector, ObstacleDetector)
    assert pipeline.vegetation_assessor is None
    assert pipeline.geometry_assessor is None


def test_vision_pipeline_invalid_frame_rejection():
    """Verify error handling on invalid image inputs and negative indices."""
    pipeline = VisionPipeline(obstacle_detector=MockObstacleDetector())

    # None image
    with pytest.raises(ValueError, match="Input image cannot be None"):
        pipeline.process_frame(None)

    # Non-ndarray
    with pytest.raises(TypeError, match="Input image must be a numpy ndarray"):
        pipeline.process_frame("invalid_image_type")

    # Empty array
    with pytest.raises(ValueError, match="empty or has invalid shape"):
        pipeline.process_frame(np.array([]))

    # Zero dimension array
    with pytest.raises(ValueError, match="empty or has invalid shape"):
        pipeline.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))

    # Negative indices
    valid_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="frame_id must be non-negative"):
        pipeline.process_frame(valid_img, frame_id=-1)

    with pytest.raises(ValueError, match="source_frame must be non-negative"):
        pipeline.process_frame(valid_img, frame_id=1, source_frame=-5)

    with pytest.raises(ValueError, match="timestamp must be non-negative"):
        pipeline.process_frame(valid_img, frame_id=1, timestamp=-0.1)


def test_vision_pipeline_mocked_inference_and_result():
    """Verify execution with injected mock detector producing valid VisionResult."""
    mock_detector = MockObstacleDetector()
    pipeline = VisionPipeline(obstacle_detector=mock_detector)

    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = pipeline.process_frame(img, frame_id=10, source_frame=50, timestamp=1.6667)

    assert isinstance(result, VisionResult)
    assert result.frame_id == 10
    assert result.source_frame == 50
    assert result.timestamp == 1.6667
    assert len(result.detections) == 1

    det = result.detections[0]
    assert det.class_name == "rock_debris"
    assert det.confidence == 0.89
    assert det.bbox == [200.0, 520.0, 320.0, 640.0]
    assert det.severity == SeverityLevel.HIGH
    assert result.vegetation is None
    assert result.geometry is None


def test_vision_pipeline_coordinate_preservation():
    """Verify that bounding box coordinates from detector are preserved without corruption."""
    expected_bbox = [154.5, 486.2, 350.0, 620.8]
    custom_detection = DetectionEvent(
        frame_id=2,
        timestamp=0.33,
        class_name="fallen_tree",
        confidence=0.95,
        bbox=expected_bbox,
        severity=SeverityLevel.CRITICAL,
        source="mock_detector",
    )
    pipeline = VisionPipeline(obstacle_detector=MockObstacleDetector(detections=[custom_detection]))

    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = pipeline.process_frame(img, frame_id=2, timestamp=0.33)

    assert result.detections[0].bbox == expected_bbox


def test_vision_pipeline_empty_detections():
    """Verify clean handling when no obstacles or defects are detected."""
    pipeline = VisionPipeline(obstacle_detector=MockObstacleDetector(detections=[]))
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = pipeline.process_frame(img, frame_id=1, timestamp=0.0)

    assert result.detections == []
    assert result.vegetation is None
    assert result.geometry is None


def test_vision_pipeline_with_assessor_hooks():
    """Verify integration hooks for vegetation and geometry assessment modules."""
    def mock_veg_assessor(image: np.ndarray, frame_id: int, timestamp: float) -> VegetationAssessment:
        return VegetationAssessment(score=0.35, intrusion_detected=False)

    class MockGeometryAssessor:
        def assess(self, image: np.ndarray, frame_id: int, timestamp: float) -> GeometryAssessment:
            return GeometryAssessment(alignment_score=0.97, status=GeometryStatus.NORMAL)

    pipeline = VisionPipeline(
        obstacle_detector=MockObstacleDetector(detections=[]),
        vegetation_assessor=mock_veg_assessor,
        geometry_assessor=MockGeometryAssessor(),
    )

    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = pipeline.process_frame(img, frame_id=3, timestamp=0.5)

    assert result.vegetation is not None
    assert result.vegetation.score == 0.35
    assert result.vegetation.intrusion_detected is False

    assert result.geometry is not None
    assert result.geometry.alignment_score == 0.97
    assert result.geometry.status == GeometryStatus.NORMAL


def test_vision_pipeline_serialization_round_trip():
    """Verify VisionResult serialization and deserialization across full pipeline outputs."""
    pipeline = VisionPipeline(obstacle_detector=MockObstacleDetector())
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = pipeline.process_frame(img, frame_id=25, source_frame=125, timestamp=4.1667)

    json_data = result.model_dump_json()
    assert isinstance(json_data, str)

    restored = VisionResult.model_validate_json(json_data)
    assert restored == result
    assert restored.detections[0].class_name == "rock_debris"
