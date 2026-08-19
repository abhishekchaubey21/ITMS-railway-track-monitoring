"""Frame extraction utility for ITMS video preprocessing pipeline.

Extracts sampled frames from video inputs with sequential index tracking,
preserving source frame timestamps and indices in filenames and manifest metadata.

NOTE: This module only performs frame extraction. It does NOT perform defect detection.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, List, Optional
import cv2

from ai_pipeline.detection.video_reader import VideoReader, VideoMetadata


@dataclass
class FrameInfo:
    """Metadata for an individual extracted frame."""
    extracted_index: int
    source_frame_index: int
    timestamp_seconds: float
    filename: str
    width: int
    height: int


@dataclass
class ExtractionSummary:
    """Summary of a completed frame extraction run."""
    source_video: str
    total_source_frames: int
    extracted_count: int
    sample_interval: int
    source_fps: float
    effective_fps: float
    resolution: tuple[int, int]
    output_dir: str
    manifest_path: str


class FrameExtractor:
    """Extracts and saves sampled video frames with metadata tracking."""

    def __init__(self, output_dir: str | Path = "data/processed/frames") -> None:
        self.output_dir = Path(output_dir)

    def extract(
        self,
        video_path: str | Path,
        sample_interval: int = 5,
        image_format: str = "jpg",
        jpeg_quality: int = 95,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ExtractionSummary:
        """Extracts frames from a video file at the specified interval.

        Args:
            video_path: Path to the input video.
            sample_interval: Extract 1 frame every N source frames (default: 5).
            image_format: Image format extension ('jpg' or 'png').
            jpeg_quality: JPEG compression quality (1-100).
            max_frames: Maximum number of frames to extract (optional).
            progress_callback: Optional callable receiving (current_extracted_count, total_estimated).

        Returns:
            ExtractionSummary containing run statistics and manifest location.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Source video not found: {video_path}")

        if sample_interval < 1:
            raise ValueError(f"sample_interval must be >= 1, got {sample_interval}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        frames_meta: List[FrameInfo] = []
        extracted_seq = 1

        with VideoReader(video_path) as reader:
            metadata: VideoMetadata = reader.get_metadata()
            source_fps = metadata.fps if metadata.fps > 0 else 30.0
            effective_fps = round(source_fps / sample_interval, 2)
            total_source_frames = metadata.total_frames
            est_extracts = (
                total_source_frames // sample_interval
                if total_source_frames > 0
                else 0
            )

            encode_params = []
            if image_format.lower() in ("jpg", "jpeg"):
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            elif image_format.lower() == "png":
                encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]

            for src_idx, frame in reader.read_frames(sample_interval=sample_interval):
                if max_frames and extracted_seq > max_frames:
                    break

                timestamp = round(src_idx / source_fps, 4)
                frame_filename = f"frame_{extracted_seq:06d}_src{src_idx:06d}.{image_format}"
                frame_path = self.output_dir / frame_filename

                cv2.imwrite(str(frame_path), frame, encode_params)

                h, w = frame.shape[:2]
                frames_meta.append(
                    FrameInfo(
                        extracted_index=extracted_seq,
                        source_frame_index=src_idx,
                        timestamp_seconds=timestamp,
                        filename=frame_filename,
                        width=w,
                        height=h,
                    )
                )

                if progress_callback:
                    progress_callback(extracted_seq, est_extracts)

                extracted_seq += 1

        manifest_path = self.output_dir / "manifest.json"
        summary_dict = {
            "source_video": str(video_path.resolve()),
            "source_filename": video_path.name,
            "total_source_frames": metadata.total_frames,
            "extracted_count": len(frames_meta),
            "sample_interval": sample_interval,
            "source_fps": source_fps,
            "effective_fps": effective_fps,
            "resolution": [metadata.width, metadata.height],
            "frames": [asdict(f) for f in frames_meta],
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        return ExtractionSummary(
            source_video=str(video_path.resolve()),
            total_source_frames=metadata.total_frames,
            extracted_count=len(frames_meta),
            sample_interval=sample_interval,
            source_fps=source_fps,
            effective_fps=effective_fps,
            resolution=(metadata.width, metadata.height),
            output_dir=str(self.output_dir.resolve()),
            manifest_path=str(manifest_path.resolve()),
        )
