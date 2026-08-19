"""CLI tool to process railway video and generate annotated MP4 video with JSON results.

Usage:
    python scripts/annotate_video.py --input data/raw/train_pov.mp4 --output data/processed/annotated_train_pov.mp4 [--json results.json] [--sample-interval N] [--max-frames M] [--conf C]

Example:
    python scripts/annotate_video.py -i data/raw/train_pov.mp4 -o data/processed/annotated_pov.mp4 --sample-interval 5 --conf 0.25
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_pipeline.detection.hazard_filter import TrackHazardFilter
from ai_pipeline.detection.obstacle_detector import ObstacleDetector
from ai_pipeline.detection.track_roi import TrackROIExtractor, TrackROIConfig
from ai_pipeline.detection.vegetation_detector import VegetationDetector
from ai_pipeline.detection.vision_pipeline import VisionPipeline
from ai_pipeline.detection.video_processor import VideoProcessor, VideoProcessingSummary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process railway input video and produce an annotated MP4 video with ITMS vision detections."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/raw/train_pov.mp4",
        help="Path to input railway video (default: data/raw/train_pov.mp4)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/processed/annotated_train_pov.mp4",
        help="Path for output annotated MP4 video (default: data/processed/annotated_train_pov.mp4)",
    )
    parser.add_argument(
        "--json",
        "-j",
        type=str,
        default="data/processed/annotated_train_pov.json",
        help="Path for output JSON detection manifest (default: data/processed/annotated_train_pov.json)",
    )
    parser.add_argument(
        "--sample-interval",
        "-s",
        type=int,
        default=1,
        help="Process 1 frame every N source frames (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=None,
        help="Maximum number of frames to process (optional)",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.25,
        help="Obstacle detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or identifier (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Inference device: 'cuda', 'cpu', or auto-detect (default: auto)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input video not found at: {input_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print(" ITMS Video Annotation Pipeline")
    print("=" * 80)
    print(f" Input Video     : {input_path.resolve()}")
    print(f" Output Video    : {Path(args.output).resolve()}")
    print(f" Output JSON     : {Path(args.json).resolve()}")
    print(f" Sample Interval : 1 frame every {args.sample_interval} frame(s)")
    if args.max_frames:
        print(f" Max Frames      : {args.max_frames}")
    print(f" YOLO Model      : {args.model} (conf >= {args.conf})")
    print("-" * 80)

    # 1. Setup Vision Pipeline components
    roi_config = TrackROIConfig(top_ratio=0.45, bottom_ratio=0.98, left_ratio=0.08, right_ratio=0.92)
    roi_extractor = TrackROIExtractor(config=roi_config)
    hazard_filter = TrackHazardFilter()

    obstacle_detector = ObstacleDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=0.45,
        imgsz=640,
        device=args.device,
        roi_extractor=roi_extractor,
        hazard_filter=hazard_filter,
        to_global_coords=True,
    )

    vegetation_detector = VegetationDetector(roi_extractor=roi_extractor, hazard_filter=hazard_filter)

    pipeline = VisionPipeline(
        obstacle_detector=obstacle_detector,
        roi_extractor=roi_extractor,
        vegetation_assessor=vegetation_detector,
    )

    processor = VideoProcessor(
        pipeline=pipeline,
        fourcc="mp4v",
        show_roi=True,
        show_hud=True,
    )

    # 2. Progress reporting callback
    def on_progress(current: int, total: int) -> None:
        pct = (current / total * 100.0) if total > 0 else 0.0
        sys.stdout.write(f"\r Processing: Frame {current}/{total} ({pct:.1f}%)")
        sys.stdout.flush()

    # 3. Process video
    t0 = time.perf_counter()
    summary: VideoProcessingSummary = processor.process_video(
        input_path=input_path,
        output_video_path=args.output,
        output_json_path=args.json,
        sample_interval=args.sample_interval,
        max_frames=args.max_frames,
        progress_callback=on_progress,
    )
    t1 = time.perf_counter()
    sys.stdout.write("\n")

    elapsed = t1 - t0
    fps_throughput = (summary.processed_frames / elapsed) if elapsed > 0 else 0.0

    print("-" * 80)
    print(" VIDEO PROCESSING COMPLETE:")
    print(f"  * Total Source Frames   : {summary.total_source_frames}")
    print(f"  * Processed Frames      : {summary.processed_frames}")
    print(f"  * Elapsed Time          : {elapsed:.2f} s ({fps_throughput:.1f} FPS processing speed)")
    print(f"  * Resolution / Output FPS: {summary.resolution[0]}x{summary.resolution[1]} @ {summary.fps:.2f} FPS")
    print(f"  * Total Obstacles Found : {summary.total_detections} (Critical/High: {summary.critical_detections})")
    print(f"  * Vegetation Intrusions : {summary.vegetation_intrusions} frame(s)")
    print(f"  * Annotated Video Saved : {summary.output_video}")
    print(f"  * JSON Manifest Saved   : {summary.output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
