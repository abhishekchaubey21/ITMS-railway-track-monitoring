"""CLI utility to inspect video files for the ITMS video processing pipeline.

Usage:
    python scripts/inspect_video.py <video_path> [--sample-interval N]
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_pipeline.detection.video_reader import VideoReader


def inspect_video(video_path: str, sample_interval: int = 1) -> None:
    """Inspects a video and prints metadata and frame sampling statistics."""
    path = Path(video_path)
    if not path.exists():
        print(f"Error: File not found at '{video_path}'", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(" ITMS Video Inspection Utility")
    print("=" * 60)
    print(f"File: {path.name}")
    print(f"Full Path: {path.resolve()}")

    try:
        reader = VideoReader(path)
        meta = reader.get_metadata()

        print("-" * 60)
        print(f" Resolution      : {meta.width} x {meta.height}")
        print(f" FPS             : {meta.fps}")
        print(f" Total Frames    : {meta.total_frames}")
        print(f" Duration        : {meta.duration_seconds:.2f} seconds")
        print(f" Codec / FourCC  : {meta.codec if meta.codec else 'N/A'}")
        print("-" * 60)

        if sample_interval > 1:
            sampled_count = 0
            for idx, frame in reader.read_frames(sample_interval=sample_interval):
                sampled_count += 1
            print(f" Sampled Frames  : {sampled_count} (every {sample_interval} frames)")
            print("-" * 60)

        reader.close()
        print("Inspection completed successfully.")
        print("=" * 60)

    except Exception as e:
        print(f"Error inspecting video: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect video properties and metadata for ITMS video pipeline."
    )
    parser.add_argument("video_path", type=str, help="Path to local video file")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=1,
        help="Frame sampling interval / stride (default: 1)",
    )
    args = parser.parse_args()
    inspect_video(args.video_path, args.sample_interval)


if __name__ == "__main__":
    main()
