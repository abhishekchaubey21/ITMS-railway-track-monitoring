"""Test video reader and frame sampling module."""

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from ai_pipeline.detection.video_reader import VideoReader


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """Creates a temporary synthetic video for testing."""
    video_file = tmp_path / "test_run.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    width, height = 320, 240
    writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

    # Write 15 dummy frames
    for i in range(15):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add simple pattern
        frame[:, :] = (i * 15, 100, 150)
        writer.write(frame)

    writer.release()
    return video_file


def test_video_reader_metadata(sample_video_path: Path):
    """Verify video metadata extraction."""
    reader = VideoReader(sample_video_path)
    meta = reader.get_metadata()

    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == 10.0
    assert meta.total_frames == 15
    assert meta.duration_seconds > 0.0
    reader.close()


def test_video_reader_sampling(sample_video_path: Path):
    """Verify sequential frame reading and interval sampling."""
    with VideoReader(sample_video_path) as reader:
        all_frames = list(reader.read_frames(sample_interval=1))
        assert len(all_frames) == 15

        sampled_frames = list(reader.read_frames(sample_interval=3))
        # 15 frames sampled with interval 3 gives 5 frames (indices 0, 3, 6, 9, 12)
        assert len(sampled_frames) == 5
        assert [idx for idx, _ in sampled_frames] == [0, 3, 6, 9, 12]


def test_video_reader_file_not_found():
    """Verify appropriate error on nonexistent file."""
    with pytest.raises(FileNotFoundError):
        VideoReader("non_existent_video_path.mp4")
