"""Video reading and frame extraction module for ITMS video pipeline.

NOTE: This foundation module performs frame extraction and video metadata inspection.
It does NOT perform defect detection yet.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Metadata extracted from a video source."""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str


class VideoReader:
    """Sequentially reads frames and extracts metadata from local video files."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        self._cap: Optional[cv2.VideoCapture] = None

    def _open(self) -> cv2.VideoCapture:
        """Opens or retrieves the OpenCV video capture handle."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.video_path))
            if not self._cap.isOpened():
                raise ValueError(f"Failed to open video file: {self.video_path}")
        return self._cap

    def get_metadata(self) -> VideoMetadata:
        """Reads video properties including FPS, resolution, total frame count, and duration."""
        cap = self._open()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)])

        duration = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0

        return VideoMetadata(
            path=str(self.video_path.resolve()),
            width=width,
            height=height,
            fps=round(fps, 2),
            total_frames=total_frames,
            duration_seconds=round(duration, 2),
            codec=codec.strip(),
        )

    def read_frames(
        self, sample_interval: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yields (frame_index, frame_array) sequentially.
        
        Args:
            sample_interval: Step size for sampling frames (default: 1 = every frame).
        
        Yields:
            Tuple of (frame_index, numpy BGR frame).
        """
        if sample_interval < 1:
            raise ValueError(f"sample_interval must be >= 1, got {sample_interval}")

        cap = self._open()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % sample_interval == 0:
                yield frame_idx, frame

            frame_idx += 1

    def close(self) -> None:
        """Releases the underlying video capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
