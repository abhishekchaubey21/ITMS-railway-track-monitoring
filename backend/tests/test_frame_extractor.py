"""Unit tests for frame extraction pipeline module."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from ai_pipeline.detection.frame_extractor import FrameExtractor


@pytest.fixture
def synthetic_video_path(tmp_path: Path) -> Path:
    """Creates a small temporary 20-frame video fixture for unit tests."""
    video_file = tmp_path / "synthetic_track.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    width, height = 320, 240
    writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

    for i in range(20):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (i * 10, 80, 120)
        writer.write(frame)

    writer.release()
    return video_file


def test_frame_extractor_synthetic_video(synthetic_video_path: Path, tmp_path: Path):
    """Verify frame extraction, file naming, and manifest generation on synthetic video."""
    out_dir = tmp_path / "extracted_frames"
    extractor = FrameExtractor(output_dir=out_dir)

    summary = extractor.extract(
        video_path=synthetic_video_path,
        sample_interval=5,
        image_format="jpg",
    )

    # 20 source frames with sample_interval=5 -> 4 extracted frames (0, 5, 10, 15)
    assert summary.total_source_frames == 20
    assert summary.extracted_count == 4
    assert summary.sample_interval == 5
    assert summary.effective_fps == 6.0
    assert summary.resolution == (320, 240)

    # Verify extracted image files on disk
    extracted_files = sorted(list(out_dir.glob("*.jpg")))
    assert len(extracted_files) == 4
    assert extracted_files[0].name == "frame_000001_src000000.jpg"
    assert extracted_files[1].name == "frame_000002_src000005.jpg"
    assert extracted_files[2].name == "frame_000003_src000010.jpg"
    assert extracted_files[3].name == "frame_000004_src000015.jpg"

    # Verify image validity
    img = cv2.imread(str(extracted_files[0]))
    assert img is not None
    assert img.shape == (240, 320, 3)

    # Verify manifest.json
    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["extracted_count"] == 4
    assert manifest["sample_interval"] == 5
    assert len(manifest["frames"]) == 4
    assert manifest["frames"][0]["source_frame_index"] == 0
    assert manifest["frames"][1]["source_frame_index"] == 5


def test_frame_extractor_max_frames(synthetic_video_path: Path, tmp_path: Path):
    """Verify max_frames cap works correctly."""
    out_dir = tmp_path / "max_test"
    extractor = FrameExtractor(output_dir=out_dir)

    summary = extractor.extract(
        video_path=synthetic_video_path,
        sample_interval=2,
        max_frames=3,
    )

    assert summary.extracted_count == 3
    extracted_files = list(out_dir.glob("*.jpg"))
    assert len(extracted_files) == 3


def test_frame_extractor_invalid_inputs(tmp_path: Path):
    """Verify exceptions for missing files or invalid intervals."""
    extractor = FrameExtractor(output_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        extractor.extract("non_existent_video.mp4")

    dummy_file = tmp_path / "dummy.mp4"
    dummy_file.touch()

    with pytest.raises(ValueError):
        extractor.extract(dummy_file, sample_interval=0)
