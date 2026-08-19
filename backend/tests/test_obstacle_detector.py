"""Unit tests for ObstacleDetector using dependency-injected mock YOLO models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from ai_pipeline.detection.obstacle_detector import ObstacleDetector, _get_default_device
from ai_pipeline.detection.track_roi import TrackROIExtractor, TrackROIConfig
from ai_pipeline.models.vision_events import (
    DetectionEvent,
    SeverityLevel,
    VisionDetectionEvent,
)


@dataclass
class MockBoxes:
    """Mock container for YOLO boxes tensor/array data."""
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return len(self.xyxy)


class MockYOLOResult:
    """Mock YOLO prediction result object."""
    def __init__(self, boxes: MockBoxes, names: Dict[int, str]):
        self.boxes = boxes
        self.names = names


class MockYOLOModel:
    """Mock YOLO model implementing predict() interface."""
    def __init__(self, results: Optional[List[MockYOLOResult]] = None, names: Optional[Dict[int, str]] = None):
        self.names = names or {0: "rock_debris", 1: "fallen_tree", 2: "person"}
        self.mock_results = results

    def predict(self, source: Any, **kwargs: Any) -> List[MockYOLOResult]:
        if self.mock_results is not None:
            return self.mock_results

        # Default standard detection
        boxes = MockBoxes(
            xyxy=np.array([[50.0, 60.0, 150.0, 180.0]]),
            conf=np.array([0.88]),
            cls=np.array([0]),
        )
        return [MockYOLOResult(boxes=boxes, names=self.names)]


def test_obstacle_detector_initialization():
    """Verify default configurations and parameter assignment."""
    detector = ObstacleDetector(
        model_path="yolo26n.pt",
        confidence_threshold=0.35,
        iou_threshold=0.50,
        imgsz=512,
        device="cpu",
    )

    assert detector.model_path == "yolo26n.pt"
    assert detector.confidence_threshold == 0.35
    assert detector.iou_threshold == 0.50
    assert detector.imgsz == 512
    assert detector.device == "cpu"
    assert isinstance(detector.roi_extractor, TrackROIExtractor)


def test_obstacle_detector_device_selection():
    """Verify explicit and default hardware device selection."""
    detector_cpu = ObstacleDetector(device="cpu")
    assert detector_cpu.device == "cpu"

    detector_cuda = ObstacleDetector(device="cuda")
    assert detector_cuda.device == "cuda"

    default_dev = _get_default_device()
    assert default_dev in ("cuda", "cpu")


def test_obstacle_detector_mock_inference():
    """Verify mock model inference, ROI integration, and DetectionEvent conversion."""
    mock_model = MockYOLOModel()
    config = TrackROIConfig(top_ratio=0.45, bottom_ratio=0.98, left_ratio=0.08, right_ratio=0.92)
    roi_ext = TrackROIExtractor(config=config)

    detector = ObstacleDetector(
        model=mock_model,
        roi_extractor=roi_ext,
        confidence_threshold=0.25,
        to_global_coords=True,
    )

    # 1080p synthetic image
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    events = detector.detect(img, frame_id=5, timestamp=1.5)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DetectionEvent)
    assert isinstance(event, VisionDetectionEvent)
    assert event.frame_id == 5
    assert event.timestamp == 1.5
    assert event.class_name == "rock_debris"
    assert event.confidence == 0.88
    assert event.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)

    # Verify global coordinate translation (ROI offset xmin=154, ymin=486)
    # Local box: [50, 60, 150, 180] -> Global box: [50+154, 60+486, 150+154, 180+486]
    assert event.bbox == [204.0, 546.0, 304.0, 666.0]


def test_obstacle_detector_confidence_filtering():
    """Verify that detections below confidence_threshold are rejected."""
    boxes = MockBoxes(
        xyxy=np.array([
            [10.0, 10.0, 50.0, 50.0],
            [60.0, 60.0, 100.0, 100.0],
        ]),
        conf=np.array([0.90, 0.20]),  # One high, one below threshold
        cls=np.array([0, 1]),
    )
    mock_model = MockYOLOModel(results=[MockYOLOResult(boxes=boxes, names={0: "rock", 1: "debris"})])

    detector = ObstacleDetector(model=mock_model, confidence_threshold=0.50)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    events = detector.detect(img, frame_id=1, timestamp=0.1)

    assert len(events) == 1
    assert events[0].class_name == "rock"
    assert events[0].confidence == 0.90


def test_obstacle_detector_invalid_and_zero_area_boxes():
    """Verify zero-area or out-of-bounds boxes are filtered out."""
    boxes = MockBoxes(
        xyxy=np.array([
            [50.0, 50.0, 50.0, 50.0],   # Zero area
            [20.0, 30.0, 80.0, 90.0],   # Valid box
        ]),
        conf=np.array([0.80, 0.85]),
        cls=np.array([0, 0]),
    )
    mock_model = MockYOLOModel(results=[MockYOLOResult(boxes=boxes, names={0: "rock"})])

    detector = ObstacleDetector(model=mock_model)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    events = detector.detect(img, frame_id=2, timestamp=0.2)

    assert len(events) == 1
    assert events[0].confidence == 0.85


def test_obstacle_detector_empty_results():
    """Verify handling when no obstacles are detected."""
    empty_boxes = MockBoxes(
        xyxy=np.empty((0, 4)),
        conf=np.empty((0,)),
        cls=np.empty((0,)),
    )
    mock_model = MockYOLOModel(results=[MockYOLOResult(boxes=empty_boxes, names={})])

    detector = ObstacleDetector(model=mock_model)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    events = detector.detect(img, frame_id=10, timestamp=2.0)

    assert events == []


def test_obstacle_detector_local_coordinates_option():
    """Verify to_global_coords=False retains local ROI coordinates."""
    boxes = MockBoxes(
        xyxy=np.array([[25.0, 35.0, 75.0, 85.0]]),
        conf=np.array([0.77]),
        cls=np.array([0]),
    )
    mock_model = MockYOLOModel(results=[MockYOLOResult(boxes=boxes, names={0: "obstacle"})])

    detector = ObstacleDetector(model=mock_model, to_global_coords=False)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    events = detector.detect(img, frame_id=1, timestamp=0.0)

    assert len(events) == 1
    assert events[0].bbox == [25.0, 35.0, 75.0, 85.0]


def test_obstacle_detector_invalid_input_parameters():
    """Verify validation exceptions for empty images or negative indices."""
    mock_model = MockYOLOModel()
    detector = ObstacleDetector(model=mock_model)

    # Empty image
    with pytest.raises(ValueError):
        detector.detect(None)

    with pytest.raises(ValueError):
        detector.detect(np.array([]))

    # Negative frame_id
    valid_img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detector.detect(valid_img, frame_id=-1)

    # Negative timestamp
    with pytest.raises(ValueError):
        detector.detect(valid_img, frame_id=0, timestamp=-0.5)
