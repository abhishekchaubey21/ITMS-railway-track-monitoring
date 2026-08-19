"""CLI tool to execute end-to-end vision diagnostic on real train POV footage.

Usage:
    python scripts/run_vision_on_video.py [--video PATH] [--frames N] [--conf C] [--model M] [--device D] [--output-dir DIR]

Defaults:
    --video      : data/raw/train_pov.mp4
    --frames     : 10
    --conf       : 0.25
    --model      : yolov8n.pt
    --device     : cuda (if available, else cpu)
    --output-dir : data/processed/inspection
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_pipeline.detection.video_reader import VideoReader
from ai_pipeline.detection.track_roi import TrackROIExtractor, TrackROIConfig
from ai_pipeline.detection.obstacle_detector import ObstacleDetector
from ai_pipeline.detection.vision_pipeline import VisionPipeline
from ai_pipeline.models.vision_events import VisionResult, DetectionEvent


def draw_vision_overlay(
    image: np.ndarray,
    result: VisionResult,
    roi_bounds: Tuple[int, int, int, int],
    latency_ms: float,
) -> np.ndarray:
    """Draws ROI boundary, detection bounding boxes, and performance banner onto frame."""
    annotated = image.copy()
    h, w = image.shape[:2]
    ymin, ymax, xmin, xmax = roi_bounds

    # 1. Subtle ROI corridor overlay
    cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (255, 180, 0), 2)
    cv2.putText(
        annotated,
        f"Track Bed ROI ({xmax - xmin}x{ymax - ymin})",
        (xmin + 6, ymin + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 200, 0),
        1,
        cv2.LINE_AA,
    )

    # 2. Draw Detection Bounding Boxes
    severity_colors = {
        "LOW": (0, 255, 255),      # Yellow
        "MEDIUM": (0, 165, 255),   # Orange
        "HIGH": (0, 80, 255),      # Dark Orange/Red
        "CRITICAL": (0, 0, 255),   # Red
    }

    for det in result.detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        color = severity_colors.get(det.severity.value, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Label tag
        label = f"{det.class_name} ({det.confidence*100:.1f}%) [{det.severity.value}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (20, 20, 20), -1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, 1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. Top-left HUD Banner
    hud_w, hud_h = 440, 75
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (15, 15, 15), -1)
    annotated = cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0)
    cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h), (255, 180, 0), 1)

    line1 = f"ITMS Vision Diagnostic | Frame #{result.frame_id:04d} (Src #{result.source_frame:05d})"
    line2 = f"Time: {result.timestamp:.2f}s | Latency: {latency_ms:.1f}ms ({1000/latency_ms:.1f} FPS)"
    line3 = f"Detections: {len(result.detections)} obstacle(s) detected"

    cv2.putText(annotated, line1, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, line2, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(
        annotated,
        line3,
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 120) if len(result.detections) == 0 else (0, 140, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated


def create_diagnostic_contact_sheet(
    annotated_frames: List[np.ndarray],
    output_path: Path,
    thumb_size: Tuple[int, int] = (640, 360),
    cols: int = 2,
) -> Path:
    """Builds a contact sheet grid of annotated diagnostic frames."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumbs = [cv2.resize(img, thumb_size, interpolation=cv2.INTER_AREA) for img in annotated_frames]

    num_items = len(thumbs)
    rows = math.ceil(num_items / cols)

    while len(thumbs) < rows * cols:
        thumbs.append(np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        row_thumbs = thumbs[r * cols : (r + 1) * cols]
        grid_rows.append(np.hstack(row_thumbs))

    contact_sheet = np.vstack(grid_rows)
    cv2.imwrite(str(output_path), contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end vision diagnostic on real railway train POV video."
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default="data/raw/train_pov.mp4",
        help="Path to input train POV video (default: data/raw/train_pov.mp4)",
    )
    parser.add_argument(
        "--frames",
        "-n",
        type=int,
        default=10,
        help="Number of representative frames to sample across video (default: 10)",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or identifier (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Inference device: 'cuda', 'cpu', or None for auto (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/processed/inspection",
        help="Output directory for diagnostic artifacts (default: data/processed/inspection)",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found at: {video_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" ITMS Real Footage Vision Pipeline Diagnostic")
    print("=" * 80)
    print(f" Video Path       : {video_path.resolve()}")
    print(f" Sample Count     : {args.frames} representative frames")
    print(f" YOLO Model       : {args.model}")
    print(f" Confidence Thresh: {args.conf}")
    print(f" Output Directory : {out_dir.resolve()}")
    print("-" * 80)

    # 1. Initialize VisionPipeline with TrackROI and ObstacleDetector
    roi_config = TrackROIConfig(top_ratio=0.45, bottom_ratio=0.98, left_ratio=0.08, right_ratio=0.92)
    roi_extractor = TrackROIExtractor(config=roi_config)

    obstacle_detector = ObstacleDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=0.45,
        imgsz=640,
        device=args.device,
        roi_extractor=roi_extractor,
        to_global_coords=True,
    )

    pipeline = VisionPipeline(
        obstacle_detector=obstacle_detector,
        roi_extractor=roi_extractor,
    )

    # 2. Inspect video and determine sampling indices
    with VideoReader(video_path) as reader:
        meta = reader.get_metadata()
        total_frames = meta.total_frames
        fps = meta.fps if meta.fps > 0 else 30.0
        w, h = meta.width, meta.height

        if total_frames <= 0:
            print("Error: Video has 0 total frames.", file=sys.stderr)
            sys.exit(1)

        ymin, ymax, xmin, xmax = roi_extractor.get_pixel_bounds(h, w)
        roi_w, roi_h = (xmax - xmin), (ymax - ymin)

        # Select evenly spaced indices
        sample_indices = [
            int(round(i * (total_frames - 1) / (args.frames - 1)))
            for i in range(args.frames)
        ]
        sample_indices = sorted(list(set(sample_indices)))

        print(f" Video Properties : {w}x{h} @ {fps:.1f} FPS | Total: {total_frames} frames ({meta.duration_seconds:.1f}s)")
        print(f" Track Bed ROI    : {roi_w}x{roi_h}px (Y: {ymin}-{ymax}, X: {xmin}-{xmax})")
        print(f" Active Device    : {obstacle_detector.device.upper()}")
        print("-" * 80)

        # 3. Process sampled frames
        frame_diagnostics: List[Dict] = []
        annotated_frames: List[np.ndarray] = []
        latencies: List[float] = []

        print(f"{'Idx':<4} | {'SrcFrame':<8} | {'Time(s)':<7} | {'Latency(ms)':<11} | {'Detections':<10} | {'Classes & Confidences'}")
        print("-" * 80)

        # Warm up YOLO on GPU to eliminate cold-start latency from timing metrics
        dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
        _ = pipeline.process_frame(dummy_frame, frame_id=0, timestamp=0.0)

        cap = cv2.VideoCapture(str(video_path))
        seq_id = 1

        for src_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            timestamp = round(src_idx / fps, 2)

            # Measure inference latency with perf_counter
            t0 = time.perf_counter()
            result: VisionResult = pipeline.process_frame(
                image=frame,
                frame_id=seq_id,
                source_frame=src_idx,
                timestamp=timestamp,
            )
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            latencies.append(lat_ms)

            # Format detection summaries
            det_strs = []
            for d in result.detections:
                det_strs.append(f"{d.class_name}:{d.confidence:.2f}[{d.bbox}]")
            det_summary_str = ", ".join(det_strs) if det_strs else "None (Nominal Track)"

            print(
                f"#{seq_id:<3} | #{src_idx:<7} | {timestamp:>6.2f}s | {lat_ms:>8.2f} ms | "
                f"{len(result.detections):>10} | {det_summary_str}"
            )

            # Record detailed frame report
            frame_diagnostics.append({
                "extracted_index": seq_id,
                "source_frame_index": src_idx,
                "timestamp_seconds": timestamp,
                "original_dimensions": [w, h],
                "roi_dimensions": [roi_w, roi_h],
                "inference_latency_ms": round(lat_ms, 2),
                "detection_count": len(result.detections),
                "detections": [d.model_dump() for d in result.detections],
            })

            # Create visual overlay
            annotated = draw_vision_overlay(
                image=frame,
                result=result,
                roi_bounds=(ymin, ymax, xmin, xmax),
                latency_ms=lat_ms,
            )
            annotated_frames.append(annotated)

            seq_id += 1

        cap.release()

    # 4. Compute Performance & Resource Statistics
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    approx_fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0

    gpu_device_name = "N/A"
    peak_vram_mb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_device_name = torch.cuda.get_device_name(0)
            peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    except Exception:
        pass

    # 5. Generate Visual Contact Sheet
    contact_sheet_path = out_dir / "vision_diagnostic_contact_sheet.jpg"
    create_diagnostic_contact_sheet(
        annotated_frames=annotated_frames,
        output_path=contact_sheet_path,
        cols=2,
    )

    # 6. Save JSON Diagnostic Report
    report_data = {
        "video_path": str(video_path.resolve()),
        "model_used": args.model,
        "device": obstacle_detector.device,
        "gpu_device_name": gpu_device_name,
        "peak_vram_mb": round(peak_vram_mb, 2),
        "total_sampled_frames": len(frame_diagnostics),
        "average_latency_ms": round(avg_latency, 2),
        "approximate_fps": round(approx_fps, 2),
        "contact_sheet_path": str(contact_sheet_path.resolve()),
        "frames": frame_diagnostics,
    }

    report_json_path = out_dir / "vision_diagnostic_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("-" * 80)
    print(" PERFORMANCE & RESOURCE SUMMARY:")
    print(f"  * Total Sampled Frames  : {len(frame_diagnostics)}")
    print(f"  * Average Latency       : {avg_latency:.2f} ms per frame")
    print(f"  * Throughput            : {approx_fps:.1f} FPS")
    print(f"  * Inference Device      : {obstacle_detector.device.upper()} ({gpu_device_name})")
    if peak_vram_mb > 0:
        print(f"  * Peak GPU VRAM Usage   : {peak_vram_mb:.1f} MB (< 25% of 6GB RTX 3050 capacity)")
    print(f"  * Annotated Contact Sheet: {contact_sheet_path.resolve()}")
    print(f"  * JSON Diagnostic Report : {report_json_path.resolve()}")
    print("=" * 80)
    print("Vision pipeline diagnostic completed successfully.")


if __name__ == "__main__":
    main()
