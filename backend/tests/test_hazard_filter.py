"""Unit tests for ego-track hazard filtering and railway semantic policy."""

from dataclasses import dataclass
from typing import List, Optional
import pytest

from ai_pipeline.detection.hazard_filter import (
    EgoTrackCorridorConfig,
    HazardPolicy,
    TrackHazardFilter,
)
from ai_pipeline.models.vision_events import DetectionEvent, SeverityLevel


@dataclass
class MockDetection:
    bbox: list[float]
    class_name: Optional[str] = "obstacle"


def test_detection_inside_ego_track():
    """A detection near the center of the running track should pass."""
    hazard_filter = TrackHazardFilter()
    bbox = [900.0, 700.0, 1020.0, 900.0]

    assert hazard_filter.is_in_ego_track(
        bbox,
        image_width=1920,
        image_height=1080,
    )


def test_detection_outside_ego_track():
    """A detection far to the side (adjacent track/platform) should be rejected."""
    hazard_filter = TrackHazardFilter()
    bbox = [1500.0, 700.0, 1750.0, 900.0]

    assert not hazard_filter.is_in_ego_track(
        bbox,
        image_width=1920,
        image_height=1080,
    )


def test_detection_above_track_region():
    """A detection above the configured track corridor should be rejected."""
    hazard_filter = TrackHazardFilter()
    bbox = [900.0, 100.0, 1020.0, 300.0]

    assert not hazard_filter.is_in_ego_track(
        bbox,
        image_width=1920,
        image_height=1080,
    )


def test_invalid_bbox_is_rejected():
    """Zero-area bounding boxes should not be treated as hazards."""
    hazard_filter = TrackHazardFilter()
    bbox = [900.0, 700.0, 900.0, 700.0]

    assert not hazard_filter.is_in_ego_track(
        bbox,
        image_width=1920,
        image_height=1080,
    )


def test_adjacent_train_detection_is_not_hazard():
    """Adjacent/parallel train on platform or side track must NOT be classified as an on-track obstacle."""
    hazard_filter = TrackHazardFilter()

    # Right-hand adjacent train (typical bounding box on side platform)
    adj_train_right = DetectionEvent(
        frame_id=1,
        timestamp=0.0,
        class_name="train",
        confidence=0.75,
        bbox=[1080.0, 486.0, 1764.0, 1030.0],
        severity=SeverityLevel.HIGH,
        source="yolo",
    )

    # Left-hand adjacent train
    adj_train_left = DetectionEvent(
        frame_id=1,
        timestamp=0.0,
        class_name="train",
        confidence=0.82,
        bbox=[150.0, 486.0, 520.0, 800.0],
        severity=SeverityLevel.HIGH,
        source="yolo",
    )

    assert not hazard_filter.is_hazard(adj_train_right, image_width=1920, image_height=1080)
    assert not hazard_filter.is_hazard(adj_train_left, image_width=1920, image_height=1080)

    filtered = hazard_filter.filter_detections(
        [adj_train_right, adj_train_left],
        image_width=1920,
        image_height=1080,
    )
    assert len(filtered) == 0


def test_train_inside_ego_track_policy_toggle():
    """Verify policy toggle for treat_train_as_hazard when a train is positioned in ego track."""
    center_train = DetectionEvent(
        frame_id=1,
        timestamp=0.0,
        class_name="train",
        confidence=0.80,
        bbox=[850.0, 600.0, 1070.0, 850.0],
        severity=SeverityLevel.CRITICAL,
        source="yolo",
    )

    # 1. Default policy: treat_train_as_hazard=False -> NOT a hazard
    default_filter = TrackHazardFilter(policy=HazardPolicy(treat_train_as_hazard=False))
    assert not default_filter.is_hazard(center_train, image_width=1920, image_height=1080)

    # 2. Custom policy: treat_train_as_hazard=True -> flagged as hazard because it's in ego corridor
    strict_filter = TrackHazardFilter(policy=HazardPolicy(treat_train_as_hazard=True))
    assert strict_filter.is_hazard(center_train, image_width=1920, image_height=1080)


def test_person_debris_vehicle_inside_ego_corridor_is_hazard():
    """Genuinely hazardous classes (person, vehicle, debris, animal, boulder) inside ego corridor are accepted."""
    hazard_filter = TrackHazardFilter()

    # Person crossing active track
    person = DetectionEvent(
        frame_id=1,
        timestamp=0.0,
        class_name="person",
        confidence=0.85,
        bbox=[920.0, 650.0, 1000.0, 820.0],
        severity=SeverityLevel.CRITICAL,
        source="yolo",
    )

    # Rock boulder on running rail
    rock = DetectionEvent(
        frame_id=1,
        timestamp=0.0,
        class_name="boulder",
        confidence=0.78,
        bbox=[880.0, 750.0, 960.0, 840.0],
        severity=SeverityLevel.HIGH,
        source="yolo",
    )

    # Fallen tree branch
    tree = DetectionEvent(
        frame_id=1,
        timestamp=0.0,
        class_name="fallen_tree",
        confidence=0.88,
        bbox=[800.0, 700.0, 1120.0, 850.0],
        severity=SeverityLevel.CRITICAL,
        source="yolo",
    )

    assert hazard_filter.is_hazard(person, image_width=1920, image_height=1080)
    assert hazard_filter.is_hazard(rock, image_width=1920, image_height=1080)
    assert hazard_filter.is_hazard(tree, image_width=1920, image_height=1080)

    results = hazard_filter.filter_detections([person, rock, tree], image_width=1920, image_height=1080)
    assert len(results) == 3


def test_object_on_adjacent_track_is_filtered():
    """Hazardous objects (person, cow) located on adjacent tracks/platforms outside ego corridor are filtered out."""
    hazard_filter = TrackHazardFilter()

    # Person standing on adjacent passenger platform (far right)
    person_on_platform = DetectionEvent(
        frame_id=1,
        timestamp=0.0,
        class_name="person",
        confidence=0.90,
        bbox=[1600.0, 500.0, 1680.0, 700.0],
        severity=SeverityLevel.MEDIUM,
        source="yolo",
    )

    # Animal grazing on far left embankment
    cow_on_embankment = DetectionEvent(
        frame_id=1,
        timestamp=0.0,
        class_name="cow",
        confidence=0.85,
        bbox=[100.0, 600.0, 240.0, 750.0],
        severity=SeverityLevel.MEDIUM,
        source="yolo",
    )

    assert not hazard_filter.is_hazard(person_on_platform, image_width=1920, image_height=1080)
    assert not hazard_filter.is_hazard(cow_on_embankment, image_width=1920, image_height=1080)

    filtered = hazard_filter.filter_detections([person_on_platform, cow_on_embankment], image_width=1920, image_height=1080)
    assert len(filtered) == 0


def test_benign_infrastructure_filtering():
    """Benign railway infrastructure objects (benches, stop signs) are filtered out from hazard list."""
    hazard_filter = TrackHazardFilter()

    bench = DetectionEvent(
        frame_id=1,
        timestamp=0.0,
        class_name="bench",
        confidence=0.80,
        bbox=[900.0, 700.0, 1020.0, 800.0],
        severity=SeverityLevel.LOW,
        source="yolo",
    )

    assert not hazard_filter.is_hazard(bench, image_width=1920, image_height=1080)


def test_filter_detections():
    """Only true hazards inside the ego-track corridor should remain."""
    hazard_filter = TrackHazardFilter()

    detections = [
        MockDetection([900.0, 700.0, 1020.0, 900.0], class_name="rock_debris"),
        MockDetection([1500.0, 700.0, 1750.0, 900.0], class_name="rock_debris"),
    ]

    filtered = hazard_filter.filter_detections(
        detections,
        image_width=1920,
        image_height=1080,
    )

    assert len(filtered) == 1
    assert filtered[0] is detections[0]


def test_empty_detection_list():
    """An empty detection list should remain empty."""
    hazard_filter = TrackHazardFilter()

    assert hazard_filter.filter_detections(
        [],
        image_width=1920,
        image_height=1080,
    ) == []


def test_invalid_image_dimensions():
    """Invalid image dimensions should raise ValueError."""
    hazard_filter = TrackHazardFilter()

    with pytest.raises(ValueError):
        hazard_filter.is_in_ego_track(
            [900.0, 700.0, 1020.0, 900.0],
            image_width=0,
            image_height=1080,
        )