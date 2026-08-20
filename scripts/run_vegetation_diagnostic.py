"""Vegetation intrusion diagnostic runner on representative real video frames.

Runs the updated VegetationDetector on representative frames sampled across the
entire 5530-frame train_pov.mp4 video (including all previous false-positive segment
keyframes and genuine intrusion frames) to evaluate true vs false positive performance,
and generates a comprehensive comparison contact sheet.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_pipeline.detection.vegetation_detector import (
    VegetationDetector,
    VegetationDetectorConfig,
    VegetationAnalysisDetails,
    EgoRunningGaugeConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diagnostic evaluation of updated VegetationDetector on representative video frames."
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default="data/raw/train_pov.mp4",
        help="Path to input video (default: data/raw/train_pov.mp4)",
    )
    parser.add_argument(
        "--output-contact-sheet",
        "-c",
        type=str,
        default="data/processed/inspection/vegetation_intrusion_contact_sheet.jpg",
        help="Path for generated contact sheet",
    )
    parser.add_argument(
        "--output-frames-dir",
        "-f",
        type=str,
        default="data/processed/inspection/vegetation_intrusions",
        help="Directory to save individual diagnostic frame captures",
    )

    args = parser.parse_args()
    video_path = Path(args.video)
    contact_sheet_path = Path(args.output_contact_sheet)
    frames_dir = Path(args.output_frames_dir)
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print(" ITMS Vegetation Intrusion Geometry Diagnostic Evaluation")
    print("=" * 80)
    print(f" Video Path         : {video_path.resolve()}")
    print(f" Output Sheet       : {contact_sheet_path.resolve()}")
    print(f" Frames Directory   : {frames_dir.resolve()}")
    print("-" * 80)

    # Keyframe indices representing:
    # 1. Previous False Positive segments (Seg 1-7: side fence, right embankment, left shoulder)
    # 2. Previous True Positive segment (Seg 8: genuine weeds inside running gauge)
    # 3. Clean baseline track frames sampled across the video
    prev_segment_rep_frames = [
        (602, "Prev Seg 01 Start (Right Embankment)"),
        (628, "Prev Seg 01 Mid (Dense Right Fence)"),
        (653, "Prev Seg 01 End (Right Fence)"),
        (782, "Prev Seg 02 Start (Right Trees)"),
        (824, "Prev Seg 02 Mid (Side Embankment)"),
        (865, "Prev Seg 02 End (Right Brush)"),
        (869, "Prev Seg 03 Start (Trackside)"),
        (875, "Prev Seg 03 End (Trackside)"),
        (878, "Prev Seg 04 (Trackside)"),
        (885, "Prev Seg 05 (Trackside)"),
        (920, "Prev Seg 06 Start (Side Platform)"),
        (973, "Prev Seg 06 Mid (Platform Edge)"),
        (1025, "Prev Seg 06 End (Side Brush)"),
        (4756, "Prev Seg 07 (Right Shoulder)"),
        (5420, "Seg 08 In-Gauge Weeds Start"),
        (5424, "Seg 08 In-Gauge Weeds Mid"),
        (5428, "Seg 08 In-Gauge Weeds Peak"),
        (5430, "Seg 08 In-Gauge Weeds High"),
        (5434, "Seg 08 In-Gauge Weeds End"),
    ]

    baseline_samples = [
        (0, "Baseline (Clean Track Start)"),
        (500, "Baseline (Clean Straight Track)"),
        (1500, "Baseline (Mid Video Track)"),
        (2500, "Baseline (Open Corridor)"),
        (3500, "Baseline (Platform Corridor)"),
    ]

    all_test_frames = sorted(prev_segment_rep_frames + baseline_samples, key=lambda x: x[0])

    detector = VegetationDetector()
    print(f" Initialized VegetationDetector with Inner Gauge Safety Inset: {detector.gauge_config.safety_margin_ratio * 100:.0f}%")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video {video_path}", file=sys.stderr)
        sys.exit(1)

    results = []
    print(f"\n Evaluating {len(all_test_frames)} diagnostic representative frames...")
    print("-" * 95)
    print(f"{'Frame':<8} | {'Description':<38} | {'Score':<8} | {'InGaugeVeg':<11} | {'GaugeDens':<10} | {'Intrusion'}")
    print("-" * 95)

    intrusion_count = 0

    for frame_idx, desc in all_test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        ts = frame_idx / 30.0
        details: VegetationAnalysisDetails = detector.analyze(frame, frame_id=frame_idx, timestamp=ts)

        if details.intrusion_detected:
            intrusion_count += 1

        results.append((frame_idx, desc, ts, frame, details))

        status_str = "YES [!] INTRUSION" if details.intrusion_detected else "NO (Trackside Only)" if details.score > 0 else "NO (Clean Track)"
        print(
            f"#{frame_idx:<7d} | {desc:<38} | {details.score:>6.4f} | "
            f"{details.center_vegetation_pixel_count:>9d} px | {details.center_density:>8.4f} | {status_str}"
        )

    cap.release()
    print("-" * 95)
    print(f" Diagnostic complete: {intrusion_count} intrusion frames detected out of {len(results)} evaluated representative frames.\n")

    # Annotate and save individual frames
    h, w = 1080, 1920
    thumb_w, thumb_h = 480, 270
    annotated_thumbs = []

    # Get gauge polygon in image coordinates
    ymin, ymax, xmin, xmax = detector.roi_extractor.get_pixel_bounds(h, w)
    gauge_top_y = max(ymin, min(int(round(detector.gauge_config.top_y_ratio * h)), ymax))
    gauge_bot_y = max(ymin, min(int(round(detector.gauge_config.bottom_y_ratio * h)), ymax))
    top_left_x, top_right_x = detector.gauge_config.get_inner_gauge_bounds_at_y(gauge_top_y, w, h)
    bot_left_x, bot_right_x = detector.gauge_config.get_inner_gauge_bounds_at_y(gauge_bot_y, w, h)

    gauge_pts = np.array([
        [int(top_left_x), int(gauge_top_y)],
        [int(top_right_x), int(gauge_top_y)],
        [int(bot_right_x), int(gauge_bot_y)],
        [int(bot_left_x), int(gauge_bot_y)],
    ], np.int32)

    for frame_idx, desc, ts, frame, details in results:
        annotated = frame.copy()

        # Draw subtle gauge corridor boundary
        cv2.polylines(annotated, [gauge_pts], True, (0, 220, 255), 2, cv2.LINE_AA)

        # Highlight detected vegetation
        clean_mask = details.binary_mask
        full_clean = np.zeros((h, w), dtype=np.uint8)
        full_clean[ymin:ymax, xmin:xmax] = clean_mask
        annotated[full_clean > 0] = [0, 230, 80]  # Green tint for detected vegetation

        # Highlight in-gauge intrusion pixels in red
        gauge_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(gauge_mask, [gauge_pts], 255)
        in_gauge = cv2.bitwise_and(full_clean, gauge_mask)
        annotated[in_gauge > 0] = [0, 0, 240]

        # Standardized HUD overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (16, 16), (820, 115), (15, 18, 24), -1)
        border_color = (0, 0, 240) if details.intrusion_detected else (0, 200, 100) if details.score > 0 else (120, 120, 120)
        cv2.rectangle(overlay, (16, 16), (820, 115), border_color, 2)
        annotated = cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0)

        # Status badge
        badge_text = "! IN-GAUGE INTRUSION" if details.intrusion_detected else "TRACKSIDE VEGETATION" if details.score > 0 else "CLEAN TRACK"
        badge_bg = (0, 0, 220) if details.intrusion_detected else (0, 140, 60) if details.score > 0 else (60, 60, 60)
        cv2.rectangle(annotated, (28, 28), (320, 60), badge_bg, -1)
        cv2.putText(annotated, badge_text, (38, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(
            annotated,
            f"Frame #{frame_idx} ({ts:.2f}s)",
            (335, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            f"Veg Score: {details.score:.4f}  |  In-Gauge Dens: {details.center_density:.4f}  |  In-Gauge Veg: {details.center_vegetation_pixel_count}px",
            (28, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 230, 240),
            1,
            cv2.LINE_AA,
        )

        # Save individual full-res frame
        filename = f"diag_frame_{frame_idx:04d}_{'intrusion' if details.intrusion_detected else 'trackside' if details.score > 0 else 'clean'}.jpg"
        cv2.imwrite(str(frames_dir / filename), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Resize thumbnail for contact sheet
        thumb = cv2.resize(annotated, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        # Thumbnail footer label
        foot_ov = thumb.copy()
        cv2.rectangle(foot_ov, (0, thumb_h - 26), (thumb_w, thumb_h), (10, 10, 10), -1)
        thumb = cv2.addWeighted(foot_ov, 0.8, thumb, 0.2, 0)
        cv2.putText(
            thumb,
            f"F#{frame_idx} | {desc[:28]} | Score: {details.score:.3f}",
            (8, thumb_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (220, 230, 240),
            1,
            cv2.LINE_AA,
        )
        annotated_thumbs.append(thumb)

    # Build Contact Sheet (4 columns x ceil(N/4) rows)
    num_cols = 4
    num_rows = int(np.ceil(len(annotated_thumbs) / num_cols))
    header_h = 80
    sheet_w = num_cols * thumb_w
    sheet_h = header_h + num_rows * thumb_h

    sheet = np.full((sheet_h, sheet_w, 3), 20, dtype=np.uint8)

    # Top Header
    cv2.rectangle(sheet, (0, 0), (sheet_w, header_h), (25, 30, 40), -1)
    cv2.line(sheet, (0, header_h), (sheet_w, header_h), (0, 180, 240), 2)
    cv2.putText(
        sheet,
        "ITMS — VEGETATION INTRUSION GEOMETRY DIAGNOSTIC CONTACT SHEET",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        f"Evaluated {len(results)} Representative Frames | Inner Gauge Margin: {detector.gauge_config.safety_margin_ratio*100:.0f}% | In-Gauge Intrusions: {intrusion_count} | Trackside Filtered: {len(results)-intrusion_count}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    # Place thumbnails
    for idx, thumb in enumerate(annotated_thumbs):
        r = idx // num_cols
        c = idx % num_cols
        y_pos = header_h + r * thumb_h
        x_pos = c * thumb_w
        sheet[y_pos : y_pos + thumb_h, x_pos : x_pos + thumb_w] = thumb

    cv2.imwrite(str(contact_sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f" Contact sheet successfully created at {contact_sheet_path.resolve()} ({contact_sheet_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
