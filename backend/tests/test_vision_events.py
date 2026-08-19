"""Comprehensive unit tests for Vision Event Pydantic schemas."""

import json
from pydantic import ValidationError
import pytest

from ai_pipeline.models.vision_events import (
    DetectionEvent,
    GeometryAssessment,
    GeometryStatus,
    SeverityLevel,
    VegetationAssessment,
    VisionResult,
)


def test_valid_detection_event():
    """Verify instantiation of a valid DetectionEvent with enum and string severity."""
    event = DetectionEvent(
        frame_id=1,
        timestamp=0.1667,
        class_name="rock_debris",
        confidence=0.92,
        bbox=[150.0, 500.0, 280.0, 620.0],
        severity=SeverityLevel.HIGH,
        source="yolov8n",
    )

    assert event.frame_id == 1
    assert event.timestamp == 0.1667
    assert event.class_name == "rock_debris"
    assert event.confidence == 0.92
    assert event.bbox == [150.0, 500.0, 280.0, 620.0]
    assert event.severity == SeverityLevel.HIGH
    assert event.source == "yolov8n"

    # Also verify string input for Enum coercion
    event2 = DetectionEvent(
        frame_id=2,
        timestamp=0.3333,
        class_name="fallen_branch",
        confidence=0.85,
        bbox=[200.0, 550.0, 400.0, 700.0],
        severity="CRITICAL",  # String converted to enum
        source="yolov8n",
    )
    assert event2.severity == SeverityLevel.CRITICAL


def test_invalid_confidence():
    """Verify confidence must be strictly within [0.0, 1.0]."""
    # Over 1.0
    with pytest.raises(ValidationError) as excinfo:
        DetectionEvent(
            frame_id=1,
            timestamp=0.0,
            class_name="obstacle",
            confidence=1.05,
            bbox=[10.0, 20.0, 30.0, 40.0],
            severity=SeverityLevel.MEDIUM,
            source="test",
        )
    assert "confidence" in str(excinfo.value)

    # Below 0.0
    with pytest.raises(ValidationError) as excinfo:
        DetectionEvent(
            frame_id=1,
            timestamp=0.0,
            class_name="obstacle",
            confidence=-0.1,
            bbox=[10.0, 20.0, 30.0, 40.0],
            severity=SeverityLevel.MEDIUM,
            source="test",
        )
    assert "confidence" in str(excinfo.value)


def test_invalid_bbox():
    """Verify bbox length must be 4 and x2 >= x1, y2 >= y1."""
    # Too few coordinates
    with pytest.raises(ValidationError) as exc:
        DetectionEvent(
            frame_id=1,
            timestamp=0.0,
            class_name="obstacle",
            confidence=0.8,
            bbox=[10.0, 20.0, 30.0],
            severity=SeverityLevel.LOW,
            source="test",
        )
    assert "bbox must contain exactly 4 coordinates" in str(exc.value)

    # Too many coordinates
    with pytest.raises(ValidationError) as exc:
        DetectionEvent(
            frame_id=1,
            timestamp=0.0,
            class_name="obstacle",
            confidence=0.8,
            bbox=[10.0, 20.0, 30.0, 40.0, 50.0],
            severity=SeverityLevel.LOW,
            source="test",
        )
    assert "bbox must contain exactly 4 coordinates" in str(exc.value)

    # Inverted X coordinates (x2 < x1)
    with pytest.raises(ValidationError) as exc:
        DetectionEvent(
            frame_id=1,
            timestamp=0.0,
            class_name="obstacle",
            confidence=0.8,
            bbox=[300.0, 100.0, 150.0, 200.0],
            severity=SeverityLevel.LOW,
            source="test",
        )
    assert "x2 (150.0) must be >= x1 (300.0)" in str(exc.value)

    # Inverted Y coordinates (y2 < y1)
    with pytest.raises(ValidationError) as exc:
        DetectionEvent(
            frame_id=1,
            timestamp=0.0,
            class_name="obstacle",
            confidence=0.8,
            bbox=[100.0, 400.0, 200.0, 250.0],
            severity=SeverityLevel.LOW,
            source="test",
        )
    assert "y2 (250.0) must be >= y1 (400.0)" in str(exc.value)


def test_valid_vegetation_assessment():
    """Verify valid VegetationAssessment."""
    veg = VegetationAssessment(score=0.42, intrusion_detected=True)
    assert veg.score == 0.42
    assert veg.intrusion_detected is True


def test_invalid_vegetation_score():
    """Verify vegetation score bounds."""
    with pytest.raises(ValidationError):
        VegetationAssessment(score=1.1, intrusion_detected=False)

    with pytest.raises(ValidationError):
        VegetationAssessment(score=-0.05, intrusion_detected=False)


def test_valid_geometry_assessment():
    """Verify valid GeometryAssessment."""
    geom = GeometryAssessment(alignment_score=0.98, status=GeometryStatus.NORMAL)
    assert geom.alignment_score == 0.98
    assert geom.status == GeometryStatus.NORMAL

    geom2 = GeometryAssessment(alignment_score=0.55, status="WARNING")
    assert geom2.status == GeometryStatus.WARNING


def test_invalid_geometry_score():
    """Verify geometry alignment score bounds."""
    with pytest.raises(ValidationError):
        GeometryAssessment(alignment_score=1.5, status=GeometryStatus.NORMAL)

    with pytest.raises(ValidationError):
        GeometryAssessment(alignment_score=-0.2, status=GeometryStatus.NORMAL)


def test_complete_vision_result():
    """Verify composite VisionResult construction with optional fields."""
    detection = DetectionEvent(
        frame_id=10,
        timestamp=1.6667,
        class_name="tree_branch",
        confidence=0.88,
        bbox=[120.0, 480.0, 300.0, 600.0],
        severity=SeverityLevel.HIGH,
        source="yolov8n",
    )
    veg = VegetationAssessment(score=0.65, intrusion_detected=True)
    geom = GeometryAssessment(alignment_score=0.91, status=GeometryStatus.ADVISORY)

    result = VisionResult(
        frame_id=10,
        source_frame=50,
        timestamp=1.6667,
        detections=[detection],
        vegetation=veg,
        geometry=geom,
    )

    assert result.frame_id == 10
    assert result.source_frame == 50
    assert len(result.detections) == 1
    assert result.vegetation.score == 0.65
    assert result.geometry.status == GeometryStatus.ADVISORY

    # Verify default empty fields
    minimal = VisionResult(frame_id=1, source_frame=5, timestamp=0.1667)
    assert minimal.detections == []
    assert minimal.vegetation is None
    assert minimal.geometry is None


def test_serialization_round_trip():
    """Verify JSON and dict serialization/deserialization."""
    original = VisionResult(
        frame_id=42,
        source_frame=210,
        timestamp=7.0,
        detections=[
            DetectionEvent(
                frame_id=42,
                timestamp=7.0,
                class_name="debris",
                confidence=0.79,
                bbox=[200.0, 500.0, 350.0, 650.0],
                severity=SeverityLevel.MEDIUM,
                source="yolov8n",
            )
        ],
        vegetation=VegetationAssessment(score=0.15, intrusion_detected=False),
        geometry=GeometryAssessment(alignment_score=0.99, status=GeometryStatus.NORMAL),
    )

    # 1. JSON Round-trip
    json_str = original.model_dump_json()
    assert isinstance(json_str, str)
    restored_from_json = VisionResult.model_validate_json(json_str)
    assert restored_from_json == original

    # 2. Dict Round-trip
    data_dict = original.model_dump()
    assert isinstance(data_dict, dict)
    restored_from_dict = VisionResult.model_validate(data_dict)
    assert restored_from_dict == original


def test_negative_frame_and_timestamp_validation():
    """Verify negative frame IDs and timestamps are disallowed."""
    # Negative frame_id in DetectionEvent
    with pytest.raises(ValidationError):
        DetectionEvent(
            frame_id=-1,
            timestamp=0.0,
            class_name="test",
            confidence=0.5,
            bbox=[0.0, 0.0, 10.0, 10.0],
            severity=SeverityLevel.LOW,
            source="test",
        )

    # Negative timestamp in DetectionEvent
    with pytest.raises(ValidationError):
        DetectionEvent(
            frame_id=1,
            timestamp=-1.5,
            class_name="test",
            confidence=0.5,
            bbox=[0.0, 0.0, 10.0, 10.0],
            severity=SeverityLevel.LOW,
            source="test",
        )

    # Negative source_frame in VisionResult
    with pytest.raises(ValidationError):
        VisionResult(frame_id=1, source_frame=-10, timestamp=0.0)

    # Negative timestamp in VisionResult
    with pytest.raises(ValidationError):
        VisionResult(frame_id=1, source_frame=5, timestamp=-0.01)
