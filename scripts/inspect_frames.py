"""CLI utility to inspect and validate extracted frames quality for ITMS.

Usage:
    python scripts/inspect_frames.py [manifest_path] [--output-dir DIR] [--samples N] [--blur-thresh T]

Defaults:
    manifest_path : data/processed/frames/manifest.json
    output-dir    : data/processed/inspection
    samples       : 6
    blur-thresh   : 100.0
"""

import argparse
import sys
import time
from pathlib import Path

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

from ai_pipeline.detection.frame_inspector import FrameInspector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and validate dataset quality for extracted ITMS video frames."
    )
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default="data/processed/frames/manifest.json",
        help="Path to frames manifest.json (default: data/processed/frames/manifest.json)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/processed/inspection",
        help="Output directory for inspection artifacts (default: data/processed/inspection)",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=6,
        help="Number of representative frames for contact sheet (default: 6)",
    )
    parser.add_argument(
        "--blur-thresh",
        type=float,
        default=100.0,
        help="Laplacian variance threshold below which frame is flagged as blurry (default: 100.0)",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(" ITMS Frame Dataset Quality & Integrity Inspection")
    print("=" * 70)
    print(f" Manifest Path      : {manifest_path.resolve()}")
    print(f" Output Directory   : {Path(args.output_dir).resolve()}")
    print(f" Representative Grid: {args.samples} frames sampled across video")
    print(f" Blur Threshold     : {args.blur_thresh}")
    print("-" * 70)

    inspector = FrameInspector(blur_threshold=args.blur_thresh)

    start_time = time.time()
    last_print = 0

    def on_progress(current: int, total: int) -> None:
        nonlocal last_print
        now = time.time()
        if now - last_print > 1.0 or current == total:
            pct = (current / total * 100) if total > 0 else 0
            print(f"\r Validating frames: {current}/{total} ({pct:.1f}%)", end="", flush=True)
            last_print = now

    try:
        report = inspector.inspect_dataset(
            manifest_path=manifest_path,
            output_dir=args.output_dir,
            num_representative=args.samples,
            progress_callback=on_progress,
        )
        elapsed = time.time() - start_time
        print(f"\r Validating frames: {report.total_manifest_frames}/{report.total_manifest_frames} complete.   ")
        print("-" * 70)
        print(" DATASET INTEGRITY & STATISTICS:")
        print(f"  * Total Frames Inspected  : {report.total_manifest_frames}")
        print(f"  * Valid Readable Frames   : {report.valid_frames}")
        print(f"  * Corrupt / Missing Frames: {report.corrupt_frames + report.missing_frames} (Corrupt: {report.corrupt_frames}, Missing: {report.missing_frames})")
        print(f"  * Frame Dimensions        : {report.resolution[0]} x {report.resolution[1]}")
        print(f"  * Mean Brightness (0-255) : {report.avg_brightness:.1f} (Min: {report.min_brightness:.1f}, Max: {report.max_brightness:.1f})")
        print(f"  * Mean Contrast (Std Dev) : {report.avg_contrast:.1f}")
        print(f"  * Mean Sharpness (LaplVar): {report.avg_sharpness:.1f} (Min: {report.min_sharpness:.1f}, Max: {report.max_sharpness:.1f})")
        print(f"  * Motion Blurry Frames    : {report.blurry_frames_count} frames (< {report.blur_threshold})")
        print("-" * 70)
        print(" REPRESENTATIVE FRAMES SAMPLED:")
        for r in report.representative_frames:
            print(
                f"  - Extracted #{r['extracted_index']:04d} (Src #{r['source_frame_index']:05d} @ {r['timestamp_seconds']:6.1f}s): "
                f"Brightness={r['mean_brightness']:5.1f}, Sharpness={r['sharpness_laplacian_var']:6.1f} [{r['filename']}]"
            )
        print("-" * 70)
        print(" EVALUATION FOR COMPUTER VISION STAGE:")
        for note in report.summary_notes:
            print(f"  [OK] {note}")
        status_label = "SUITABLE FOR CV PROCESSING" if report.is_suitable_for_cv else "ISSUES DETECTED - REVIEW REQUIRED"
        print(f"\n OVERALL VERDICT: [{status_label}]")
        print(f" Contact Sheet Artifact : {report.contact_sheet_path}")
        print(f" JSON Inspection Report : {Path(args.output_dir) / 'inspection_report.json'}")
        print(f" Inspection Time Taken  : {elapsed:.2f}s ({report.total_manifest_frames / elapsed:.1f} frames/sec)")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during dataset inspection: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
