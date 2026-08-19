"""CLI utility to preview and validate railway track ROI cropping for ITMS.

Usage:
    python scripts/preview_roi.py [manifest_path] [--output-dir DIR] [--samples N] [--top T] [--bottom B] [--left L] [--right R]

Defaults:
    manifest_path : data/processed/frames/manifest.json
    output-dir    : data/processed/inspection
    samples       : 4
    top           : 0.45
    bottom        : 0.98
    left          : 0.08
    right         : 0.92
"""

import argparse
import json
import sys
import time
from pathlib import Path
import cv2

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

from ai_pipeline.detection.track_roi import TrackROIExtractor, TrackROIConfig
from ai_pipeline.detection.frame_inspector import FrameInspector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview and generate railway track ROI visualizations for ITMS."
    )
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default="data/processed/frames/manifest.json",
        help="Path to manifest.json (default: data/processed/frames/manifest.json)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/processed/inspection",
        help="Output directory for ROI preview artifacts (default: data/processed/inspection)",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=4,
        help="Number of representative frames for ROI preview sheet (default: 4)",
    )
    parser.add_argument(
        "--top",
        type=float,
        default=0.45,
        help="Top boundary ratio (default: 0.45)",
    )
    parser.add_argument(
        "--bottom",
        type=float,
        default=0.98,
        help="Bottom boundary ratio (default: 0.98)",
    )
    parser.add_argument(
        "--left",
        type=float,
        default=0.08,
        help="Left boundary ratio (default: 0.08)",
    )
    parser.add_argument(
        "--right",
        type=float,
        default=0.92,
        help="Right boundary ratio (default: 0.92)",
    )
    parser.add_argument(
        "--polygon",
        action="store_true",
        help="Enable perspective polygon masking",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" ITMS Railway Track Region of Interest (ROI) Preview")
    print("=" * 70)
    print(f" Manifest Path      : {manifest_path.resolve()}")
    print(f" Output Directory   : {out_dir.resolve()}")
    print(f" ROI Ratios         : Top={args.top:.2f}, Bottom={args.bottom:.2f}, Left={args.left:.2f}, Right={args.right:.2f}")
    print(f" Polygon Masking    : {'Enabled' if args.polygon else 'Disabled'}")
    print("-" * 70)

    config = TrackROIConfig(
        top_ratio=args.top,
        bottom_ratio=args.bottom,
        left_ratio=args.left,
        right_ratio=args.right,
        enable_polygon_mask=args.polygon,
    )
    roi_extractor = TrackROIExtractor(config=config)

    # Load manifest and select representative frames
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    if not frames:
        print("Error: No frames found in manifest.", file=sys.stderr)
        sys.exit(1)

    inspector = FrameInspector()
    rep_indices = inspector.select_representative_indices(len(frames), args.samples)
    rep_items = [frames[i] for i in rep_indices]

    frames_dir = manifest_path.parent
    sample_frame = frames[0]
    meta = roi_extractor.get_metadata(sample_frame.get("height", 1080), sample_frame.get("width", 1920))

    print(f" ROI GEOMETRY (Based on {meta.original_width}x{meta.original_height} input):")
    print(f"  * Y-Range (Vertical)    : {meta.ymin} to {meta.ymax} (Height: {meta.crop_height}px, {meta.crop_height/meta.original_height*100:.1f}% of frame)")
    print(f"  * X-Range (Horizontal)  : {meta.xmin} to {meta.xmax} (Width:  {meta.crop_width}px, {meta.crop_width/meta.original_width*100:.1f}% of frame)")
    print(f"  * Area Reduction Factor : {(1 - (meta.crop_width * meta.crop_height)/(meta.original_width * meta.original_height))*100:.1f}% background discarded")
    print("-" * 70)

    # Generate visual artifacts
    start_time = time.time()

    # 1. Contact Sheet
    contact_sheet_path = out_dir / "roi_preview_contact_sheet.jpg"
    roi_extractor.create_roi_preview_contact_sheet(
        frames_dir=frames_dir,
        representative_items=rep_items,
        output_path=contact_sheet_path,
    )
    print(f" [OK] Generated ROI Comparison Contact Sheet : {contact_sheet_path.resolve()}")

    # 2. Individual Annotated & Cropped Samples for Frame #1
    sample_img_path = frames_dir / sample_frame["filename"]
    if sample_img_path.exists():
        raw_img = cv2.imread(str(sample_img_path))
        if raw_img is not None:
            annotated = roi_extractor.annotate(raw_img, shade_outside=True)
            crop = roi_extractor.crop(raw_img)

            ann_out = out_dir / "roi_annotated_sample.jpg"
            crop_out = out_dir / "roi_cropped_sample.jpg"

            cv2.imwrite(str(ann_out), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(str(crop_out), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            print(f" [OK] Saved Annotated Full Sample           : {ann_out.resolve()}")
            print(f" [OK] Saved High-Resolution Cropped Sample  : {crop_out.resolve()}")

    # 3. Save ROI Config JSON
    roi_config_path = out_dir / "roi_config.json"
    config_dict = {
        "top_ratio": args.top,
        "bottom_ratio": args.bottom,
        "left_ratio": args.left,
        "right_ratio": args.right,
        "enable_polygon_mask": args.polygon,
        "pixel_bounds": {
            "ymin": meta.ymin,
            "ymax": meta.ymax,
            "xmin": meta.xmin,
            "xmax": meta.xmax,
            "crop_width": meta.crop_width,
            "crop_height": meta.crop_height,
        },
        "original_resolution": [meta.original_width, meta.original_height],
        "crop_resolution": [meta.crop_width, meta.crop_height],
    }
    with open(roi_config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f" [OK] Saved ROI Config Specification       : {roi_config_path.resolve()}")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Track ROI preview completed successfully in {elapsed:.2f}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
