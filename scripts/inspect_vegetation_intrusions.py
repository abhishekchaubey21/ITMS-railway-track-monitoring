"""Diagnostic script to inspect and visualize detected vegetation intrusions for ITMS.

Reads results_final.json, identifies all frames with vegetation intrusion flags,
groups contiguous frame sequences into distinct segments, prints segment statistics,
extracts representative frames (first, middle, last) from train_pov.mp4, and compiles
a comprehensive annotated contact sheet and individual frame captures.

Usage:
    python scripts/inspect_vegetation_intrusions.py [options]

Options:
    --results, -r       Path to results_final.json (default: data/processed/results_final.json)
    --video, -v         Path to raw video (default: data/raw/train_pov.mp4)
    --output-dir, -o    Base directory for inspection outputs (default: data/processed/inspection)
    --contact-sheet, -c Path for contact sheet image (default: data/processed/inspection/vegetation_intrusion_contact_sheet.jpg)
    --frames-dir, -f    Directory for individual frame crops (default: data/processed/inspection/vegetation_intrusions)
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
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


@dataclass
class RepresentativeFrame:
    """Metadata for a representative frame extracted from a segment."""
    role: str  # "start", "middle", "end"
    source_frame: int
    timestamp: float
    score: float
    raw_image: Optional[np.ndarray] = None
    annotated_image: Optional[np.ndarray] = None
    saved_path: Optional[Path] = None


@dataclass
class IntrusionSegment:
    """Grouped contiguous segment of vegetation intrusion frames."""
    segment_id: int
    frames: List[Dict[str, Any]]
    first_source_frame: int
    last_source_frame: int
    start_time: float
    end_time: float
    frame_count: int
    min_score: float
    max_score: float
    avg_score: float
    representative_frames: List[RepresentativeFrame]


def load_intrusion_frames(results_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Loads results JSON and extracts all frames with vegetation intrusion detected."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    intrusions = [
        item for item in results
        if item.get("vegetation", {}).get("intrusion_detected") is True
    ]
    return data, intrusions


def group_contiguous_segments(intrusion_frames: List[Dict[str, Any]]) -> List[IntrusionSegment]:
    """Groups intrusion frame records into contiguous segments based on consecutive source frames."""
    if not intrusion_frames:
        return []

    # Sort by source_frame to ensure correct chronological sequence
    sorted_frames = sorted(intrusion_frames, key=lambda x: x.get("source_frame", 0))

    raw_segments: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = []

    for frame in sorted_frames:
        if not current_group:
            current_group.append(frame)
        else:
            prev_src = current_group[-1].get("source_frame", -999)
            curr_src = frame.get("source_frame", -999)
            if curr_src == prev_src + 1:
                current_group.append(frame)
            else:
                raw_segments.append(current_group)
                current_group = [frame]

    if current_group:
        raw_segments.append(current_group)

    segments: List[IntrusionSegment] = []
    for idx, group in enumerate(raw_segments, start=1):
        first_frame = group[0]
        last_frame = group[-1]
        mid_idx = len(group) // 2
        mid_frame = group[mid_idx]

        scores = [item.get("vegetation", {}).get("score", 0.0) for item in group]
        min_s = float(min(scores))
        max_s = float(max(scores))
        avg_s = float(sum(scores) / len(scores))

        rep_list = [
            RepresentativeFrame(
                role="start",
                source_frame=int(first_frame.get("source_frame", 0)),
                timestamp=float(first_frame.get("timestamp", 0.0)),
                score=float(first_frame.get("vegetation", {}).get("score", 0.0)),
            ),
            RepresentativeFrame(
                role="middle",
                source_frame=int(mid_frame.get("source_frame", 0)),
                timestamp=float(mid_frame.get("timestamp", 0.0)),
                score=float(mid_frame.get("vegetation", {}).get("score", 0.0)),
            ),
            RepresentativeFrame(
                role="end",
                source_frame=int(last_frame.get("source_frame", 0)),
                timestamp=float(last_frame.get("timestamp", 0.0)),
                score=float(last_frame.get("vegetation", {}).get("score", 0.0)),
            ),
        ]

        segment = IntrusionSegment(
            segment_id=idx,
            frames=group,
            first_source_frame=int(first_frame.get("source_frame", 0)),
            last_source_frame=int(last_frame.get("source_frame", 0)),
            start_time=float(first_frame.get("timestamp", 0.0)),
            end_time=float(last_frame.get("timestamp", 0.0)),
            frame_count=len(group),
            min_score=min_s,
            max_score=max_s,
            avg_score=avg_s,
            representative_frames=rep_list,
        )
        segments.append(segment)

    return segments


def extract_frames_from_video(
    video_path: Path,
    segments: List[IntrusionSegment],
) -> None:
    """Extracts raw representative frame images from video file for each segment."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        for seg in segments:
            for rep in seg.representative_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, rep.source_frame)
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"Warning: Failed to read frame {rep.source_frame} from {video_path}", file=sys.stderr)
                    # Create blank fallback frame
                    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                rep.raw_image = frame
    finally:
        cap.release()


def annotate_frame_image(
    image: np.ndarray,
    segment_id: int,
    total_segments: int,
    role: str,
    source_frame: int,
    timestamp: float,
    score: float,
) -> np.ndarray:
    """Draws standardized inspection HUD overlay on representative intrusion frame image.
    
    Draws:
      - 'INTRUSION' badge / indicator
      - Source frame number
      - Timestamp in seconds
      - Vegetation intrusion score
      - Segment role (Start / Middle / End)
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Create top-left info panel
    panel_w = int(w * 0.46)
    panel_h = 105
    overlay = annotated.copy()
    cv2.rectangle(overlay, (16, 16), (16 + panel_w, 16 + panel_h), (15, 18, 24), -1)
    cv2.rectangle(overlay, (16, 16), (16 + panel_w, 16 + panel_h), (0, 70, 255), 2)  # Orange/Red border
    annotated = cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0)

    # Intrusion Badge (Red box with bold white text)
    badge_x, badge_y = 28, 28
    badge_w, badge_h = 160, 32
    cv2.rectangle(annotated, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (0, 0, 220), -1)
    cv2.putText(
        annotated,
        "! INTRUSION",
        (badge_x + 12, badge_y + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Segment and Role Label
    role_str = role.upper()
    seg_label = f"Segment {segment_id:02d}/{total_segments:02d} [{role_str}]"
    cv2.putText(
        annotated,
        seg_label,
        (badge_x + badge_w + 14, badge_y + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )

    # Metrics Line: Frame, Time, Score
    line2 = f"Frame: #{source_frame:d}  |  Time: {timestamp:.2f}s  |  Veg Score: {score:.4f}"
    cv2.putText(
        annotated,
        line2,
        (badge_x + 4, badge_y + badge_h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    return annotated


def save_individual_frames(
    segments: List[IntrusionSegment],
    output_dir: Path,
) -> List[Path]:
    """Saves all annotated representative frames to the specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []
    total_segs = len(segments)

    for seg in segments:
        for rep in seg.representative_frames:
            if rep.raw_image is None:
                continue

            # Annotate frame
            annotated = annotate_frame_image(
                image=rep.raw_image,
                segment_id=seg.segment_id,
                total_segments=total_segs,
                role=rep.role,
                source_frame=rep.source_frame,
                timestamp=rep.timestamp,
                score=rep.score,
            )
            rep.annotated_image = annotated

            filename = f"seg{seg.segment_id:02d}_frame{rep.source_frame:04d}_{rep.role}.jpg"
            file_path = output_dir / filename
            cv2.imwrite(str(file_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            rep.saved_path = file_path
            saved_files.append(file_path)

    return saved_files


def create_contact_sheet(
    segments: List[IntrusionSegment],
    output_path: Path,
    thumb_width: int = 640,
    thumb_height: int = 360,
) -> Path:
    """Builds a structured visual contact sheet grid containing 3 representative frames per segment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_height = 90
    col_header_height = 40
    num_rows = len(segments)
    num_cols = 3  # Start, Middle, End

    grid_width = num_cols * thumb_width
    grid_height = header_height + col_header_height + (num_rows * thumb_height)

    # Master canvas with dark slate theme
    canvas = np.full((grid_height, grid_width, 3), 20, dtype=np.uint8)

    # Top Title Banner
    cv2.rectangle(canvas, (0, 0), (grid_width, header_height), (28, 33, 44), -1)
    cv2.line(canvas, (0, header_height), (grid_width, header_height), (0, 180, 240), 2)

    cv2.putText(
        canvas,
        "ITMS — RAILWAY VEGETATION INTRUSION INSPECTION SHEET",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    total_frames = sum(s.frame_count for s in segments)
    sub_text = f"Dataset: results_final.json  |  Total Segments: {len(segments)}  |  Total Intrusion Frames: {total_frames}  |  Source: train_pov.mp4"
    cv2.putText(
        canvas,
        sub_text,
        (24, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 200, 215),
        1,
        cv2.LINE_AA,
    )

    # Column Sub-headers
    cv2.rectangle(
        canvas,
        (0, header_height),
        (grid_width, header_height + col_header_height),
        (35, 42, 54),
        -1,
    )
    col_titles = [
        "1. SEGMENT START FRAME",
        "2. SEGMENT MIDDLE FRAME (PEAK / MID)",
        "3. SEGMENT END FRAME",
    ]
    for c_idx, title in enumerate(col_titles):
        col_x = c_idx * thumb_width
        cv2.putText(
            canvas,
            title,
            (col_x + 18, header_height + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        if c_idx > 0:
            cv2.line(
                canvas,
                (col_x, header_height),
                (col_x, grid_height),
                (50, 58, 70),
                1,
            )

    # Render each segment row
    for r_idx, seg in enumerate(segments):
        row_y = header_height + col_header_height + (r_idx * thumb_height)

        for c_idx, rep in enumerate(seg.representative_frames):
            col_x = c_idx * thumb_width

            if rep.annotated_image is not None:
                thumb = cv2.resize(
                    rep.annotated_image,
                    (thumb_width, thumb_height),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                thumb = np.zeros((thumb_height, thumb_width, 3), dtype=np.uint8)

            # Segment row banner on bottom of thumbnail
            overlay = thumb.copy()
            cv2.rectangle(overlay, (0, thumb_height - 30), (thumb_width, thumb_height), (10, 10, 10), -1)
            thumb = cv2.addWeighted(overlay, 0.75, thumb, 0.25, 0)

            footer_txt = f"Seg #{seg.segment_id:02d} ({seg.frame_count:d}f, {seg.start_time:.1f}s-{seg.end_time:.1f}s) | Min:{seg.min_score:.3f} Max:{seg.max_score:.3f} Avg:{seg.avg_score:.3f}"
            cv2.putText(
                thumb,
                footer_txt,
                (10, thumb_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (200, 220, 240),
                1,
                cv2.LINE_AA,
            )

            canvas[row_y : row_y + thumb_height, col_x : col_x + thumb_width] = thumb

        # Draw horizontal row separator
        cv2.line(
            canvas,
            (0, row_y + thumb_height),
            (grid_width, row_y + thumb_height),
            (60, 70, 85),
            1,
        )

    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return output_path


def print_segments_summary(segments: List[IntrusionSegment]) -> None:
    """Prints formatted summary table of contiguous intrusion segments."""
    print("\n" + "=" * 100)
    print(" ITMS CONTIGUOUS VEGETATION INTRUSION SEGMENT ANALYSIS")
    print("=" * 100)
    header = (
        f"{'Seg':<4} | {'Start Frame':<11} | {'End Frame':<9} | "
        f"{'Start Time':<10} | {'End Time':<8} | {'Frames':<6} | "
        f"{'Min Score':<9} | {'Max Score':<9} | {'Avg Score':<9}"
    )
    print(header)
    print("-" * 100)

    total_frames = 0
    all_scores: List[float] = []

    for seg in segments:
        total_frames += seg.frame_count
        seg_scores = [f.get("vegetation", {}).get("score", 0.0) for f in seg.frames]
        all_scores.extend(seg_scores)

        row = (
            f"#{seg.segment_id:<3d} | "
            f"Src #{seg.first_source_frame:<7d} | "
            f"Src #{seg.last_source_frame:<5d} | "
            f"{seg.start_time:>7.2f}s   | "
            f"{seg.end_time:>6.2f}s | "
            f"{seg.frame_count:>6d} | "
            f"{seg.min_score:>9.4f} | "
            f"{seg.max_score:>9.4f} | "
            f"{seg.avg_score:>9.4f}"
        )
        print(row)

    print("-" * 100)
    overall_min = min(all_scores) if all_scores else 0.0
    overall_max = max(all_scores) if all_scores else 0.0
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

    summary_row = (
        f"{'TOTAL':<4} | {'8 Segments':<11} | {'':<9} | "
        f"{segments[0].start_time:>7.2f}s   | "
        f"{segments[-1].end_time:>6.2f}s | "
        f"{total_frames:>6d} | "
        f"{overall_min:>9.4f} | "
        f"{overall_max:>9.4f} | "
        f"{overall_avg:>9.4f}"
    )
    print(summary_row)
    print("=" * 100 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and visualize vegetation intrusions detected in ITMS video run."
    )
    parser.add_argument(
        "--results",
        "-r",
        type=str,
        default="data/processed/results_final.json",
        help="Path to results_final.json (default: data/processed/results_final.json)",
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default="data/raw/train_pov.mp4",
        help="Path to raw video (default: data/raw/train_pov.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/processed/inspection",
        help="Base output inspection directory (default: data/processed/inspection)",
    )
    parser.add_argument(
        "--contact-sheet",
        "-c",
        type=str,
        default="data/processed/inspection/vegetation_intrusion_contact_sheet.jpg",
        help="Path to output contact sheet image",
    )
    parser.add_argument(
        "--frames-dir",
        "-f",
        type=str,
        default="data/processed/inspection/vegetation_intrusions",
        help="Directory to save individual representative frame captures",
    )

    args = parser.parse_args()

    results_path = Path(args.results)
    video_path = Path(args.video)
    contact_sheet_path = Path(args.contact_sheet)
    frames_dir = Path(args.frames_dir)

    print("=" * 80)
    print(" ITMS Vegetation Intrusion Diagnostic Inspector")
    print("=" * 80)
    print(f" Results File       : {results_path.resolve()}")
    print(f" Input Video        : {video_path.resolve()}")
    print(f" Frames Output Dir  : {frames_dir.resolve()}")
    print(f" Contact Sheet Path : {contact_sheet_path.resolve()}")
    print("-" * 80)

    # 1 & 2. Load results and find intrusion frames
    data, intrusions = load_intrusion_frames(results_path)
    print(f" Found {len(intrusions)} intrusion frames out of {len(data.get('results', []))} processed frames.")

    if not intrusions:
        print("No vegetation intrusions found in results.")
        return

    # 3. Group into contiguous segments
    segments = group_contiguous_segments(intrusions)
    print(f" Grouped into {len(segments)} contiguous intrusion segments.")

    # 4. Print segment summary
    print_segments_summary(segments)

    # 5. Extract representative frames from raw video
    print(f" Extracting representative frames (first, middle, last) from {video_path.name}...")
    extract_frames_from_video(video_path, segments)

    # 7 & 8. Annotate and save individual frame captures
    print(f" Annotating and saving individual representative frames to {frames_dir}...")
    saved_frames = save_individual_frames(segments, frames_dir)
    print(f" Saved {len(saved_frames)} individual frame images.")

    # 6. Generate contact sheet
    print(f" Generating comprehensive contact sheet at {contact_sheet_path}...")
    create_contact_sheet(segments, contact_sheet_path)
    print(f" Contact sheet successfully created ({contact_sheet_path.stat().st_size / 1024:.1f} KB).")

    print("\n" + "=" * 80)
    print(" Inspection Complete!")
    print(f" Contact Sheet : {contact_sheet_path}")
    print(f" Frame Crops   : {frames_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
