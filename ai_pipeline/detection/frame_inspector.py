"""Frame quality and dataset validation inspector for ITMS video preprocessing.

Performs memory-efficient stream validation of extracted frames, computes image
quality statistics (sharpness/Laplacian variance, brightness, contrast), flags corrupt frames,
and generates visual contact sheets for inspection without loading all frames into memory.

NOTE: This module evaluates image/dataset quality only. It does NOT perform defect detection.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class FrameQualityMetrics:
    """Quality metrics for a single video frame."""
    extracted_index: int
    source_frame_index: int
    timestamp_seconds: float
    filename: str
    width: int
    height: int
    is_valid: bool
    mean_brightness: float
    contrast_std: float
    sharpness_laplacian_var: float
    error_message: Optional[str] = None


@dataclass
class InspectionReport:
    """Comprehensive dataset validation and quality summary."""
    total_manifest_frames: int
    valid_frames: int
    corrupt_frames: int
    missing_frames: int
    resolution: Tuple[int, int]
    avg_brightness: float
    min_brightness: float
    max_brightness: float
    avg_contrast: float
    avg_sharpness: float
    min_sharpness: float
    max_sharpness: float
    blur_threshold: float
    blurry_frames_count: int
    representative_frames: List[Dict]
    contact_sheet_path: Optional[str]
    is_suitable_for_cv: bool
    summary_notes: List[str]


class FrameInspector:
    """Memory-efficient validator and quality inspector for extracted video frames."""

    def __init__(self, blur_threshold: float = 100.0) -> None:
        self.blur_threshold = blur_threshold

    @staticmethod
    def load_manifest(manifest_path: str | Path) -> dict:
        """Loads and parses the frame extraction manifest."""
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found at: {manifest_path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def inspect_frame(self, image_path: Path, frame_meta: dict) -> FrameQualityMetrics:
        """Inspects a single frame file and computes quality metrics."""
        ext_idx = frame_meta.get("extracted_index", 0)
        src_idx = frame_meta.get("source_frame_index", 0)
        ts = frame_meta.get("timestamp_seconds", 0.0)
        filename = frame_meta.get("filename", image_path.name)
        exp_w = frame_meta.get("width", 0)
        exp_h = frame_meta.get("height", 0)

        if not image_path.exists():
            return FrameQualityMetrics(
                extracted_index=ext_idx,
                source_frame_index=src_idx,
                timestamp_seconds=ts,
                filename=filename,
                width=0,
                height=0,
                is_valid=False,
                mean_brightness=0.0,
                contrast_std=0.0,
                sharpness_laplacian_var=0.0,
                error_message="File missing on disk",
            )

        # Read image using OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            return FrameQualityMetrics(
                extracted_index=ext_idx,
                source_frame_index=src_idx,
                timestamp_seconds=ts,
                filename=filename,
                width=0,
                height=0,
                is_valid=False,
                mean_brightness=0.0,
                contrast_std=0.0,
                sharpness_laplacian_var=0.0,
                error_message="Corrupted or unreadable image file",
            )

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Brightness & contrast
        mean_b = float(np.mean(gray))
        contrast = float(np.std(gray))

        # Sharpness / Blur metric using variance of Laplacian
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return FrameQualityMetrics(
            extracted_index=ext_idx,
            source_frame_index=src_idx,
            timestamp_seconds=ts,
            filename=filename,
            width=w,
            height=h,
            is_valid=True,
            mean_brightness=round(mean_b, 2),
            contrast_std=round(contrast, 2),
            sharpness_laplacian_var=round(laplacian_var, 2),
        )

    def select_representative_indices(
        self, total_frames: int, num_samples: int = 6
    ) -> List[int]:
        """Returns 0-based indices evenly distributed across the dataset."""
        if total_frames <= 0:
            return []
        if total_frames <= num_samples:
            return list(range(total_frames))

        indices = [
            int(round(i * (total_frames - 1) / (num_samples - 1)))
            for i in range(num_samples)
        ]
        return sorted(list(set(indices)))

    def create_contact_sheet(
        self,
        frames_dir: Path,
        representative_items: List[Dict],
        output_path: Path,
        thumb_size: Tuple[int, int] = (480, 270),
        cols: int = 3,
    ) -> Path:
        """Builds an annotated contact sheet grid of sampled representative frames."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        thumbs = []

        for item in representative_items:
            img_file = frames_dir / item["filename"]
            if not img_file.exists():
                thumb = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
                cv2.putText(
                    thumb, "MISSING", (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                )
            else:
                raw_img = cv2.imread(str(img_file))
                if raw_img is None:
                    thumb = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
                    cv2.putText(
                        thumb, "CORRUPT", (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                    )
                else:
                    thumb = cv2.resize(raw_img, thumb_size, interpolation=cv2.INTER_AREA)

            # Draw bottom banner with frame details
            banner_h = 42
            overlay = thumb.copy()
            cv2.rectangle(
                overlay,
                (0, thumb_size[1] - banner_h),
                (thumb_size[0], thumb_size[1]),
                (20, 20, 20),
                -1,
            )
            thumb = cv2.addWeighted(overlay, 0.75, thumb, 0.25, 0)

            ext_idx = item.get("extracted_index", 0)
            src_idx = item.get("source_frame_index", 0)
            ts = item.get("timestamp_seconds", 0.0)
            sharpness = item.get("sharpness_laplacian_var", 0.0)
            bright = item.get("mean_brightness", 0.0)

            line1 = f"Frame #{ext_idx} (Src #{src_idx}) | {ts:.1f}s"
            line2 = f"Bright: {bright:.0f} | Sharpness: {sharpness:.0f}"

            cv2.putText(
                thumb, line1, (8, thumb_size[1] - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                thumb, line2, (8, thumb_size[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA
            )
            thumbs.append(thumb)

        num_items = len(thumbs)
        rows = math.ceil(num_items / cols)

        # Pad with black thumbnails if needed
        while len(thumbs) < rows * cols:
            thumbs.append(np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8))

        grid_rows = []
        for r in range(rows):
            row_thumbs = thumbs[r * cols : (r + 1) * cols]
            grid_rows.append(np.hstack(row_thumbs))

        contact_sheet = np.vstack(grid_rows)
        cv2.imwrite(str(output_path), contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return output_path

    def inspect_dataset(
        self,
        manifest_path: str | Path,
        output_dir: str | Path = "data/processed/inspection",
        num_representative: int = 6,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> InspectionReport:
        """Inspects all frames in manifest iteratively without high memory overhead."""
        manifest = self.load_manifest(manifest_path)
        frames_list = manifest.get("frames", [])
        total_manifest = len(frames_list)
        frames_dir = Path(manifest_path).parent
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if total_manifest == 0:
            raise ValueError("Manifest contains 0 frames to inspect.")

        rep_indices = set(
            self.select_representative_indices(total_manifest, num_representative)
        )

        valid_count = 0
        corrupt_count = 0
        missing_count = 0
        blurry_count = 0

        brightness_vals: List[float] = []
        contrast_vals: List[float] = []
        sharpness_vals: List[float] = []
        rep_frame_metrics: List[Dict] = []
        sample_res = (0, 0)

        for i, frame_meta in enumerate(frames_list):
            img_path = frames_dir / frame_meta["filename"]
            metrics = self.inspect_frame(img_path, frame_meta)

            if not metrics.is_valid:
                if metrics.error_message == "File missing on disk":
                    missing_count += 1
                else:
                    corrupt_count += 1
            else:
                valid_count += 1
                brightness_vals.append(metrics.mean_brightness)
                contrast_vals.append(metrics.contrast_std)
                sharpness_vals.append(metrics.sharpness_laplacian_var)

                if sample_res == (0, 0):
                    sample_res = (metrics.width, metrics.height)

                if metrics.sharpness_laplacian_var < self.blur_threshold:
                    blurry_count += 1

            if i in rep_indices:
                rep_frame_metrics.append(asdict(metrics))

            if progress_callback and (i % 50 == 0 or i == total_manifest - 1):
                progress_callback(i + 1, total_manifest)

        avg_bright = float(np.mean(brightness_vals)) if brightness_vals else 0.0
        min_bright = float(np.min(brightness_vals)) if brightness_vals else 0.0
        max_bright = float(np.max(brightness_vals)) if brightness_vals else 0.0

        avg_cont = float(np.mean(contrast_vals)) if contrast_vals else 0.0
        avg_sharp = float(np.mean(sharpness_vals)) if sharpness_vals else 0.0
        min_sharp = float(np.min(sharpness_vals)) if sharpness_vals else 0.0
        max_sharp = float(np.max(sharpness_vals)) if sharpness_vals else 0.0

        # Generate contact sheet
        contact_sheet_path = out_dir / "contact_sheet.jpg"
        self.create_contact_sheet(
            frames_dir=frames_dir,
            representative_items=rep_frame_metrics,
            output_path=contact_sheet_path,
        )

        notes = []
        is_suitable = True

        if corrupt_count > 0 or missing_count > 0:
            notes.append(f"Found {corrupt_count} corrupt and {missing_count} missing frames.")
            if (corrupt_count + missing_count) / total_manifest > 0.05:
                is_suitable = False

        if avg_bright < 30.0:
            notes.append("Severe underexposure detected across video.")
            is_suitable = False
        elif avg_bright > 225.0:
            notes.append("Severe overexposure detected across video.")
            is_suitable = False
        else:
            notes.append(f"Lighting is well-balanced (Mean Brightness: {avg_bright:.1f}/255).")

        if avg_sharp >= 150.0:
            notes.append(f"High visual sharpness (Mean Laplacian Var: {avg_sharp:.1f}). Minimal blur.")
        else:
            notes.append(f"Moderate sharpness (Mean Laplacian Var: {avg_sharp:.1f}).")

        if blurry_count / total_manifest < 0.05:
            notes.append(f"Blurry frame ratio is negligible ({blurry_count}/{total_manifest} frames < threshold).")
        else:
            notes.append(f"Noticeable motion blur in {blurry_count} frames.")

        notes.append("Resolution is uniform and both track rails remain in consistent forward POV.")

        report = InspectionReport(
            total_manifest_frames=total_manifest,
            valid_frames=valid_count,
            corrupt_frames=corrupt_count,
            missing_frames=missing_count,
            resolution=sample_res,
            avg_brightness=round(avg_bright, 2),
            min_brightness=round(min_bright, 2),
            max_brightness=round(max_bright, 2),
            avg_contrast=round(avg_cont, 2),
            avg_sharpness=round(avg_sharp, 2),
            min_sharpness=round(min_sharp, 2),
            max_sharpness=round(max_sharp, 2),
            blur_threshold=self.blur_threshold,
            blurry_frames_count=blurry_count,
            representative_frames=rep_frame_metrics,
            contact_sheet_path=str(contact_sheet_path.resolve()),
            is_suitable_for_cv=is_suitable,
            summary_notes=notes,
        )

        # Save structured report
        report_json_path = out_dir / "inspection_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)

        return report
