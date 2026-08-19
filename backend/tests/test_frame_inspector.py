"""Unit tests for frame quality inspector module."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from ai_pipeline.detection.frame_inspector import FrameInspector


@pytest.fixture
def synthetic_frame_dataset(tmp_path: Path) -> Path:
    """Creates a temporary folder with 6 synthetic frame images and manifest.json."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames_list = []

    for i in range(6):
        filename = f"frame_{i+1:06d}_src{i*5:06d}.jpg"
        img_path = frames_dir / filename
        # Generate image with varying patterns
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        img[:, :] = (i * 30 + 50, 100, 150)
        # Add high-contrast edges to simulate sharpness
        cv2.rectangle(img, (20, 20), (140, 100), (255, 255, 255), 2)
        cv2.imwrite(str(img_path), img)

        frames_list.append(
            {
                "extracted_index": i + 1,
                "source_frame_index": i * 5,
                "timestamp_seconds": round(i * 5 / 30.0, 4),
                "filename": filename,
                "width": 160,
                "height": 120,
            }
        )

    manifest = {
        "source_video": "mock.mp4",
        "total_source_frames": 30,
        "extracted_count": 6,
        "sample_interval": 5,
        "frames": frames_list,
    }

    manifest_path = frames_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return manifest_path


def test_frame_inspector_synthetic_dataset(synthetic_frame_dataset: Path, tmp_path: Path):
    """Verify inspection statistics, representative sampling, and contact sheet creation."""
    out_dir = tmp_path / "inspection"
    inspector = FrameInspector(blur_threshold=50.0)

    report = inspector.inspect_dataset(
        manifest_path=synthetic_frame_dataset,
        output_dir=out_dir,
        num_representative=4,
    )

    assert report.total_manifest_frames == 6
    assert report.valid_frames == 6
    assert report.corrupt_frames == 0
    assert report.missing_frames == 0
    assert report.resolution == (160, 120)
    assert report.avg_brightness > 0
    assert report.is_suitable_for_cv is True

    # Contact sheet verification
    contact_sheet_path = Path(report.contact_sheet_path)
    assert contact_sheet_path.exists()
    cs_img = cv2.imread(str(contact_sheet_path))
    assert cs_img is not None
    assert cs_img.shape[0] > 0 and cs_img.shape[1] > 0

    # JSON report verification
    json_path = out_dir / "inspection_report.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
    assert saved_report["valid_frames"] == 6


def test_frame_inspector_corrupt_and_missing(tmp_path: Path):
    """Verify detection of missing and corrupted image files."""
    frames_dir = tmp_path / "corrupt_test"
    frames_dir.mkdir()

    # Frame 1: Valid
    valid_file = frames_dir / "frame_000001_src000000.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(valid_file), img)

    # Frame 2: Corrupted non-image content
    corrupt_file = frames_dir / "frame_000002_src000005.jpg"
    with open(corrupt_file, "w") as f:
        f.write("corrupted non image binary data")

    # Frame 3: Missing (not created on disk)

    manifest = {
        "frames": [
            {"extracted_index": 1, "source_frame_index": 0, "timestamp_seconds": 0.0, "filename": valid_file.name},
            {"extracted_index": 2, "source_frame_index": 5, "timestamp_seconds": 0.16, "filename": corrupt_file.name},
            {"extracted_index": 3, "source_frame_index": 10, "timestamp_seconds": 0.33, "filename": "missing.jpg"},
        ]
    }

    manifest_path = frames_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    inspector = FrameInspector()
    report = inspector.inspect_dataset(manifest_path, output_dir=tmp_path / "out", num_representative=3)

    assert report.total_manifest_frames == 3
    assert report.valid_frames == 1
    assert report.corrupt_frames == 1
    assert report.missing_frames == 1


def test_representative_indices_distribution():
    """Verify even distribution of sampled representative frame indices."""
    inspector = FrameInspector()

    # 100 frames sampled into 5 representative points
    indices = inspector.select_representative_indices(total_frames=100, num_samples=5)
    assert indices == [0, 25, 50, 74, 99] or indices == [0, 25, 50, 75, 99]
    assert indices[0] == 0
    assert indices[-1] == 99

    # Edge cases
    assert inspector.select_representative_indices(0, 5) == []
    assert inspector.select_representative_indices(3, 5) == [0, 1, 2]
