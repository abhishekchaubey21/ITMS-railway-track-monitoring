"""CLI utility for extracting sampled frames from railway inspection video.

Usage:
    python scripts/extract_frames.py [video_path] [--output-dir DIR] [--sample-interval N] [--format {jpg,png}] [--quality Q]

Default:
    video_path      : data/raw/train_pov.mp4
    output-dir      : data/processed/frames
    sample-interval : 5 (extracts ~6 FPS from 30 FPS video)
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_pipeline.detection.frame_extractor import FrameExtractor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and sample video frames for ITMS preprocessing."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default="data/raw/train_pov.mp4",
        help="Path to input video (default: data/raw/train_pov.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/processed/frames",
        help="Output directory for extracted frames (default: data/processed/frames)",
    )
    parser.add_argument(
        "--sample-interval",
        "-s",
        type=int,
        default=5,
        help="Frame sampling interval / stride (default: 5 -> ~6 FPS from 30 FPS)",
    )
    parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format (default: jpg)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (1-100, default: 95)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional max number of frames to extract (for testing)",
    )

    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 65)
    print(" ITMS Video Frame Extraction Stage")
    print("=" * 65)
    print(f" Input Video     : {video_path.resolve()}")
    print(f" Output Directory: {Path(args.output_dir).resolve()}")
    print(f" Sample Interval : Every {args.sample_interval} frames")
    print(f" Output Format   : {args.format.upper()} (Quality: {args.quality})")
    if args.max_frames:
        print(f" Max Frame Limit : {args.max_frames}")
    print("-" * 65)

    extractor = FrameExtractor(output_dir=args.output_dir)

    start_time = time.time()
    last_print = 0

    def on_progress(current: int, total_est: int) -> None:
        nonlocal last_print
        now = time.time()
        if now - last_print > 1.0 or current == total_est:
            pct = (current / total_est * 100) if total_est > 0 else 0
            print(f"\r Extracting frames: {current}/{total_est} ({pct:.1f}%)", end="", flush=True)
            last_print = now

    try:
        summary = extractor.extract(
            video_path=video_path,
            sample_interval=args.sample_interval,
            image_format=args.format,
            jpeg_quality=args.quality,
            max_frames=args.max_frames,
            progress_callback=on_progress,
        )
        elapsed = time.time() - start_time
        print(f"\r Extracting frames: {summary.extracted_count} frames extracted.   ")
        print("-" * 65)
        print(f" Source Total Frames : {summary.total_source_frames}")
        print(f" Extracted Frames    : {summary.extracted_count}")
        print(f" Source FPS          : {summary.source_fps}")
        print(f" Effective FPS       : {summary.effective_fps}")
        print(f" Resolution          : {summary.resolution[0]}x{summary.resolution[1]}")
        print(f" Time Elapsed        : {elapsed:.2f} seconds")
        print(f" Processing Speed    : {summary.extracted_count / elapsed:.1f} extracted frames/sec")
        print(f" Manifest Saved To   : {summary.manifest_path}")
        print("=" * 65)
        print("Frame extraction completed successfully.")

    except Exception as e:
        print(f"\nError during frame extraction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
