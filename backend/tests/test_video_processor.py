"""Unit tests for VideoProcessor module."""

import json
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import pytest

from ai_pipeline.detection.video_processor import (
    VideoProcessor,
    VideoProcessingSummary,
    annotate_frame,
)
from ai_pipeline.detection.vision_pipeline import VisionPipeline
from ai_pipeline.models.vision_events import (
    DetectionEvent,
    GeometryAssessment,
    GeometryStatus,
    SeverityLevel,
    VegetationAssessment,
    VisionResult,
)


@pytest.fixture
def synthetic_video_path(tmp_path: Path) -> Path:
    """Creates a small temporary 15-frame video fixture for unit tests."""
    video_file = tmp_path / "synthetic_track.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    width, height = 320, 240
    writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

    for i in range(15):
        frame = np.full((height, width, 3), fill_value=int(i * 12), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_file


class MockVisionPipeline:
    """Mock VisionPipeline producing deterministic results without model execution."""

    def __init__(self, detections: Optional[List[DetectionEvent]] = None, fail_on_frame: Optional[int] = None):
        self.mock_detections = detections
        self.fail_on_frame = fail_on_frame

    def process_frame(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        source_frame: Optional[int] = None,
        timestamp: float = 0.0,
    ) -> VisionResult:
        if self.fail_on_frame is not None and frame_id == self.fail_on_frame:
            raise RuntimeError(f"Simulated pipeline failure on frame {frame_id}")

        src_frame = source_frame if source_frame is not None else frame_id
        detections = self.mock_detections if self.mock_detections is not None else [
            DetectionEvent(
                frame_id=frame_id,
                timestamp=timestamp,
                class_name="track_obstacle",
                confidence=0.91,
                bbox=[40.0, 50.0, 120.0, 160.0],
                severity=SeverityLevel.HIGH,
                source="mock_detector",
            )
        ]

        veg = VegetationAssessment(score=0.25, intrusion_detected=(frame_id % 2 == 1))
        geom = GeometryAssessment(alignment_score=0.96, status=GeometryStatus.NORMAL)

        return VisionResult(
            frame_id=frame_id,
            source_frame=src_frame,
            timestamp=timestamp,
            detections=detections,
            vegetation=veg,
            geometry=geom,
        )


def test_video_processor_initialization():
    """Verify default and custom initialization options."""
    processor = VideoProcessor()
    assert isinstance(processor.pipeline, VisionPipeline)
    assert processor.fourcc == "mp4v"
    assert processor.show_roi is True
    assert processor.show_hud is True

    custom_pipeline = MockVisionPipeline()
    proc_custom = VideoProcessor(pipeline=custom_pipeline, fourcc="avc1", show_roi=False, show_hud=False)
    assert proc_custom.pipeline == custom_pipeline
    assert proc_custom.fourcc == "avc1"
    assert proc_custom.show_roi is False
    assert proc_custom.show_hud is False


def test_video_processor_invalid_input_path(tmp_path: Path):
    """Verify error handling on non-existent input files or invalid sample interval."""
    processor = VideoProcessor(pipeline=MockVisionPipeline())
    missing_video = tmp_path / "non_existent.mp4"
    out_video = tmp_path / "out.mp4"

    with pytest.raises(FileNotFoundError):
        processor.process_video(missing_video, out_video)

    existing_stub = tmp_path / "stub.mp4"
    existing_stub.touch()

    with pytest.raises(ValueError, match="sample_interval must be >= 1"):
        processor.process_video(existing_stub, out_video, sample_interval=0)


def test_video_processor_invalid_empty_video(tmp_path: Path):
    """Verify error handling on zero-byte/empty video files."""
    empty_video = tmp_path / "empty_video.mp4"
    empty_video.touch()

    out_video = tmp_path / "out.mp4"
    processor = VideoProcessor(pipeline=MockVisionPipeline())

    with pytest.raises(ValueError, match="(Failed to open video file|invalid dimensions or zero frames)"):
        processor.process_video(empty_video, out_video)


def test_video_processor_mock_pipeline_execution(synthetic_video_path: Path, tmp_path: Path):
    """Verify full end-to-end processing with mock pipeline, output video, and JSON generation."""
    out_video = tmp_path / "annotated_output.mp4"
    out_json = tmp_path / "vision_results.json"

    mock_pipe = MockVisionPipeline()
    processor = VideoProcessor(pipeline=mock_pipe)

    # Process every 3rd frame of 15 source frames -> 5 processed frames
    summary: VideoProcessingSummary = processor.process_video(
        input_path=synthetic_video_path,
        output_video_path=out_video,
        output_json_path=out_json,
        sample_interval=3,
    )

    assert summary.total_source_frames == 15
    assert summary.processed_frames == 5
    assert summary.fps == 10.0
    assert summary.resolution == (320, 240)
    assert summary.total_detections == 5
    assert summary.critical_detections == 5
    assert summary.vegetation_intrusions == 3  # frames 1, 3, 5

    # Check output video exists and has content
    assert out_video.exists()
    assert out_video.stat().st_size > 0

    # Check output JSON exists and is valid
    assert out_json.exists()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_source_frames"] == 15
    assert data["processed_frames"] == 5
    assert len(data["results"]) == 5
    assert data["results"][0]["frame_id"] == 1
    assert data["results"][0]["source_frame"] == 0
    assert data["results"][1]["source_frame"] == 3
    assert data["results"][4]["source_frame"] == 12


def test_video_processor_output_resolution_and_fps(synthetic_video_path: Path, tmp_path: Path):
    """Verify that generated output MP4 maintains valid resolution and FPS."""
    out_video = tmp_path / "test_out.mp4"
    processor = VideoProcessor(pipeline=MockVisionPipeline())

    processor.process_video(
        input_path=synthetic_video_path,
        output_video_path=out_video,
        sample_interval=1,
    )

    cap = cv2.VideoCapture(str(out_video))
    try:
        assert cap.isOpened()
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert out_w == 320
        assert out_h == 240
    finally:
        cap.release()


def test_video_processor_max_frames_constraint(synthetic_video_path: Path, tmp_path: Path):
    """Verify max_frames parameter limits processing length."""
    out_video = tmp_path / "limited_out.mp4"
    processor = VideoProcessor(pipeline=MockVisionPipeline())

    summary = processor.process_video(
        input_path=synthetic_video_path,
        output_video_path=out_video,
        sample_interval=1,
        max_frames=4,
    )

    assert summary.processed_frames == 4
    assert len(summary.results) == 4


def test_annotate_frame_rendering():
    """Verify annotate_frame visual rendering across multiple detections and telemetry states."""
    img = np.full((720, 1280, 3), fill_value=90, dtype=np.uint8)

    det_crit = DetectionEvent(
        frame_id=1,
        timestamp=0.5,
        class_name="person",
        confidence=0.88,
        bbox=[100.0, 150.0, 250.0, 400.0],
        severity=SeverityLevel.CRITICAL,
        source="test",
    )
    det_low = DetectionEvent(
        frame_id=1,
        timestamp=0.5,
        class_name="debris",
        confidence=0.35,
        bbox=[400.0, 450.0, 480.0, 520.0],
        severity=SeverityLevel.LOW,
        source="test",
    )

    res = VisionResult(
        frame_id=1,
        source_frame=15,
        timestamp=0.5,
        detections=[det_crit, det_low],
        vegetation=VegetationAssessment(score=0.45, intrusion_detected=True),
        geometry=GeometryAssessment(alignment_score=0.92, status=GeometryStatus.ADVISORY),
    )

    annotated = annotate_frame(
        image=img,
        result=res,
        roi_bounds=(300, 700, 100, 1180),
        show_roi=True,
        show_hud=True,
    )

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img.shape
    # Frame content has been altered by drawings
    assert not np.array_equal(annotated, img)


def test_video_processor_resource_cleanup_on_error(synthetic_video_path: Path, tmp_path: Path):
    """Verify that video reader and writer resources are cleanly released when pipeline fails."""
    out_video = tmp_path / "error_out.mp4"
    failing_pipe = MockVisionPipeline(fail_on_frame=2)
    processor = VideoProcessor(pipeline=failing_pipe)

    with pytest.raises(RuntimeError, match="Simulated pipeline failure on frame 2"):
        processor.process_video(
            input_path=synthetic_video_path,
            output_video_path=out_video,
            sample_interval=1,
        )

    # Verify that file lock on out_video has been released and file can be manipulated
    assert out_video.exists()
    out_video.unlink()
    assert not out_video.exists()
