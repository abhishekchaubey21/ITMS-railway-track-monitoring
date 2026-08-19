"""Video-level inference and annotated video generation pipeline for ITMS.

Processes railway input video frame-by-frame through VisionPipeline,
rendering rich visual overlays (obstacle bounding boxes, severities, vegetation density,
HUD telemetry) and writing an annotated MP4 video along with machine-readable JSON results.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from ai_pipeline.detection.track_roi import TrackROIExtractor
from ai_pipeline.detection.video_reader import VideoReader, VideoMetadata
from ai_pipeline.detection.vision_pipeline import VisionPipeline
from ai_pipeline.models.vision_events import (
    DetectionEvent,
    GeometryAssessment,
    SeverityLevel,
    VegetationAssessment,
    VisionResult,
)


@dataclass
class VideoProcessingSummary:
    """Summary of a completed video inference and annotation run."""

    input_video: str
    output_video: str
    output_json: Optional[str]
    total_source_frames: int
    processed_frames: int
    resolution: Tuple[int, int]
    fps: float
    total_detections: int
    critical_detections: int
    vegetation_intrusions: int
    results: List[VisionResult]


def annotate_frame(
    image: np.ndarray,
    result: VisionResult,
    roi_bounds: Optional[Tuple[int, int, int, int]] = None,
    show_roi: bool = True,
    show_hud: bool = True,
) -> np.ndarray:
    """Draws detection bounding boxes, vegetation metrics, and HUD banner on video frame."""
    annotated = image.copy()
    h, w = image.shape[:2]

    # 1. Subtle Track ROI corridor outline
    if show_roi and roi_bounds is not None:
        ymin, ymax, xmin, xmax = roi_bounds
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (255, 180, 0), 1)
        cv2.putText(
            annotated,
            "Track Bed ROI",
            (xmin + 6, ymin + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )

    # 2. Draw Obstacle/Hazard Bounding Boxes
    severity_colors = {
        "LOW": (0, 255, 255),      # Yellow
        "MEDIUM": (0, 165, 255),   # Orange
        "HIGH": (0, 80, 255),      # Dark Orange/Red
        "CRITICAL": (0, 0, 255),   # Red
    }

    for det in result.detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        # Ensure within image boundaries
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        sev_key = det.severity.value if hasattr(det.severity, "value") else str(det.severity)
        color = severity_colors.get(sev_key, (0, 255, 0))

        # Box border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label tag
        label = f"{det.class_name} {det.confidence*100:.1f}% [{sev_key}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        tag_y1 = max(0, y1 - th - 8)
        tag_y2 = y1
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 8, tag_y2), (20, 20, 20), -1)
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 8, tag_y2), color, 1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, tag_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. HUD Telemetry Overlay
    if show_hud:
        hud_w, hud_h = 460, 80
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (18, 18, 18), -1)
        annotated = cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0)
        cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h), (255, 180, 0), 1)

        # Header
        cv2.putText(
            annotated,
            f"ITMS Track Inspection | Frame #{result.frame_id:04d} (Src #{result.source_frame:05d})",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Timestamp & Obstacles
        obs_text = f"Obstacles: {len(result.detections)}"
        obs_color = (0, 255, 120) if len(result.detections) == 0 else (0, 120, 255)
        cv2.putText(
            annotated,
            f"Time: {result.timestamp:.2f}s | {obs_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            obs_color,
            1,
            cv2.LINE_AA,
        )

        # Vegetation & Geometry Status
        veg_str = "Vegetation: N/A"
        veg_color = (180, 180, 180)
        if result.vegetation is not None:
            v_score = result.vegetation.score * 100.0
            if result.vegetation.intrusion_detected:
                veg_str = f"Vegetation: {v_score:.1f}% [INTRUSION]"
                veg_color = (0, 0, 255)
            else:
                veg_str = f"Vegetation: {v_score:.1f}% [CLEAR]"
                veg_color = (0, 230, 120)

        geom_str = ""
        if result.geometry is not None:
            geom_str = f" | Track: {result.geometry.status.value}"

        cv2.putText(
            annotated,
            f"{veg_str}{geom_str}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            veg_color,
            1,
            cv2.LINE_AA,
        )

    return annotated


class VideoProcessor:
    """Orchestrates end-to-end video inference and produces annotated video and JSON results."""

    def __init__(
        self,
        pipeline: Optional[VisionPipeline] = None,
        fourcc: str = "mp4v",
        show_roi: bool = True,
        show_hud: bool = True,
    ) -> None:
        """Initializes VideoProcessor with VisionPipeline and video encoding preferences.

        Args:
            pipeline: Injected or default VisionPipeline instance.
            fourcc: FourCC codec string for output video (default 'mp4v').
            show_roi: Whether to render the track ROI box.
            show_hud: Whether to render the telemetry HUD banner.
        """
        self.pipeline = pipeline or VisionPipeline()
        self.fourcc = fourcc
        self.show_roi = show_roi
        self.show_hud = show_hud

    def _create_video_writer(
        self,
        output_path: Path,
        fps: float,
        width: int,
        height: int,
    ) -> cv2.VideoWriter:
        """Initializes OpenCV VideoWriter with fallback codecs if primary fails."""
        codecs_to_try = [self.fourcc, "mp4v", "avc1", "H264", "MJPG", "XVID"]
        # Deduplicate while preserving order
        unique_codecs = list(dict.fromkeys(codecs_to_try))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        for codec_str in unique_codecs:
            try:
                fourcc_code = cv2.VideoWriter_fourcc(*codec_str)
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc_code,
                    fps,
                    (width, height),
                )
                if writer.isOpened():
                    return writer
                writer.release()
            except Exception:
                continue

        raise RuntimeError(
            f"Failed to initialize VideoWriter for '{output_path}' with codecs: {unique_codecs}. "
            "Please check that OpenCV is installed with ffmpeg/video writer support."
        )

    def process_video(
        self,
        input_path: str | Path,
        output_video_path: str | Path,
        output_json_path: Optional[str | Path] = None,
        sample_interval: int = 1,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> VideoProcessingSummary:
        """Processes an input video, runs VisionPipeline frame-by-frame, and writes annotated video.

        Args:
            input_path: Path to input railway video file.
            output_video_path: Destination path for annotated MP4 video.
            output_json_path: Destination path for frame-level JSON results (optional).
            sample_interval: Process 1 frame every N source frames (default 1).
            max_frames: Maximum number of frames to process (optional).
            progress_callback: Optional callable receiving (current_processed_count, total_estimated).

        Returns:
            VideoProcessingSummary containing execution metrics and all frame VisionResults.
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input video file not found: {input_file}")

        if sample_interval < 1:
            raise ValueError(f"sample_interval must be >= 1, got {sample_interval}")

        out_vid_file = Path(output_video_path)
        out_vid_file.parent.mkdir(parents=True, exist_ok=True)

        writer: Optional[cv2.VideoWriter] = None
        results: List[VisionResult] = []
        critical_count = 0
        veg_intrusion_count = 0
        total_detections_count = 0

        # Read metadata and stream frames
        with VideoReader(input_file) as reader:
            metadata: VideoMetadata = reader.get_metadata()
            w, h = metadata.width, metadata.height
            total_source_frames = metadata.total_frames
            source_fps = metadata.fps if metadata.fps > 0 else 30.0
            effective_fps = round(source_fps / sample_interval, 2)

            if w <= 0 or h <= 0 or total_source_frames <= 0:
                raise ValueError(
                    f"Video file has invalid dimensions or zero frames: {w}x{h}, {total_source_frames} frames."
                )

            # Pre-calculate ROI bounds if available
            roi_bounds = None
            if hasattr(self.pipeline, "roi_extractor") and self.pipeline.roi_extractor:
                roi_bounds = self.pipeline.roi_extractor.get_pixel_bounds(h, w)

            # Initialize output VideoWriter
            writer = self._create_video_writer(
                output_path=out_vid_file,
                fps=effective_fps,
                width=w,
                height=h,
            )

            est_total = (
                min(total_source_frames // sample_interval, max_frames)
                if max_frames
                else total_source_frames // sample_interval
            )

            processed_seq = 1

            try:
                for src_idx, frame in reader.read_frames(sample_interval=sample_interval):
                    if max_frames is not None and (processed_seq - 1) >= max_frames:
                        break

                    timestamp = round(src_idx / source_fps, 4)

                    # 1. Process frame with VisionPipeline
                    result: VisionResult = self.pipeline.process_frame(
                        image=frame,
                        frame_id=processed_seq,
                        source_frame=src_idx,
                        timestamp=timestamp,
                    )
                    results.append(result)

                    # Tally metrics
                    total_detections_count += len(result.detections)
                    for det in result.detections:
                        sev_val = det.severity.value if hasattr(det.severity, "value") else str(det.severity)
                        if sev_val in ("CRITICAL", "HIGH"):
                            critical_count += 1

                    if result.vegetation and result.vegetation.intrusion_detected:
                        veg_intrusion_count += 1

                    # 2. Render visual annotations
                    annotated = annotate_frame(
                        image=frame,
                        result=result,
                        roi_bounds=roi_bounds,
                        show_roi=self.show_roi,
                        show_hud=self.show_hud,
                    )

                    # 3. Write annotated frame to output video
                    writer.write(annotated)

                    if progress_callback:
                        progress_callback(processed_seq, max(1, est_total))

                    processed_seq += 1

            finally:
                if writer is not None:
                    writer.release()

        # 4. Save JSON results if requested
        out_json_file = Path(output_json_path) if output_json_path else None
        if out_json_file:
            out_json_file.parent.mkdir(parents=True, exist_ok=True)
            summary_dict = {
                "input_video": str(input_file.resolve()),
                "output_video": str(out_vid_file.resolve()),
                "total_source_frames": metadata.total_frames,
                "processed_frames": len(results),
                "sample_interval": sample_interval,
                "source_fps": source_fps,
                "effective_fps": effective_fps,
                "resolution": [w, h],
                "total_detections": total_detections_count,
                "critical_detections": critical_count,
                "vegetation_intrusions": veg_intrusion_count,
                "results": [r.model_dump() for r in results],
            }
            with open(out_json_file, "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, indent=2)

        return VideoProcessingSummary(
            input_video=str(input_file.resolve()),
            output_video=str(out_vid_file.resolve()),
            output_json=str(out_json_file.resolve()) if out_json_file else None,
            total_source_frames=metadata.total_frames,
            processed_frames=len(results),
            resolution=(w, h),
            fps=effective_fps,
            total_detections=total_detections_count,
            critical_detections=critical_count,
            vegetation_intrusions=veg_intrusion_count,
            results=results,
        )
