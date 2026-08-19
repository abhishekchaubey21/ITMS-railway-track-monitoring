"""Unit tests for ego-track hazard filtering."""

from dataclasses import dataclass

import pytest

from ai_pipeline.detection.hazard_filter import (
    EgoTrackCorridorConfig,
    TrackHazardFilter,
)


@dataclass
class MockDetection:
    bbox: list[float]


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
    """A detection far to the side should be rejected."""

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


def test_filter_detections():
    """Only detections inside the ego-track corridor should remain."""

    hazard_filter = TrackHazardFilter()

    detections = [
        MockDetection([900.0, 700.0, 1020.0, 900.0]),
        MockDetection([1500.0, 700.0, 1750.0, 900.0]),
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