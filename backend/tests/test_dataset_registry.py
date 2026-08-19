"""Unit tests for dataset registry model and metadata service."""

from pathlib import Path
import pytest

from ai_pipeline.models.dataset_registry import (
    DatasetRegistry,
    CandidateDataset,
    PrimaryVideoSource,
)


def test_dataset_registry_from_file():
    """Verify loading and parsing of data/dataset_registry.json."""
    registry = DatasetRegistry.from_file("data/dataset_registry.json")

    assert registry.schema_version == "1.0.0"
    assert "ITMS" in registry.project
    assert registry.primary_video.name == "Indian Railways Front-Facing Train POV"
    assert registry.primary_video.duration_seconds > 180.0
    assert len(registry.candidate_datasets) >= 4


def test_dataset_registry_get_by_id():
    """Verify querying dataset by ID."""
    registry = DatasetRegistry.from_file("data/dataset_registry.json")

    railsem = registry.get_dataset_by_id("railsem19")
    assert railsem is not None
    assert railsem.name == "RailSem19"
    assert railsem.academic_hackathon_usable is True
    assert railsem.relevance_score >= 0.90

    obstacle_ds = registry.get_dataset_by_id("rail_obstacle_roboflow")
    assert obstacle_ds is not None
    assert "rock_debris" in obstacle_ds.defect_or_object_classes

    non_existent = registry.get_dataset_by_id("non_existent_dataset_id")
    assert non_existent is None


def test_dataset_registry_query_by_capability():
    """Verify querying datasets mapped to specific capabilities."""
    registry = DatasetRegistry.from_file("data/dataset_registry.json")

    obstacle_matches = registry.get_datasets_for_capability("obstruction")
    assert len(obstacle_matches) >= 1
    assert obstacle_matches[0].id == "rail_obstacle_roboflow"

    sensor_matches = registry.get_datasets_for_capability("imu")
    assert len(sensor_matches) >= 1
    assert sensor_matches[0].id == "osdar23"


def test_dataset_registry_list_usable():
    """Verify filtering datasets approved for hackathon/academic use."""
    registry = DatasetRegistry.from_file("data/dataset_registry.json")
    usable = registry.list_usable_datasets()
    assert len(usable) == len(registry.candidate_datasets)
    assert all(d.academic_hackathon_usable for d in usable)


def test_dataset_registry_to_dict():
    """Verify dictionary serialization."""
    registry = DatasetRegistry.from_file("data/dataset_registry.json")
    data = registry.to_dict()

    assert data["schema_version"] == "1.0.0"
    assert "primary_video_source" in data
    assert "candidate_datasets" in data


def test_dataset_registry_file_not_found(tmp_path: Path):
    """Verify exception when registry file is missing."""
    with pytest.raises(FileNotFoundError):
        DatasetRegistry.from_file(tmp_path / "missing_registry.json")
