"""Deterministic Railway Track Region-of-Interest (ROI) Extraction & Preprocessing Module.

Extracts the critical track bed, running rails, sleepers, and fastener regions from
front-facing train POV footage while filtering out non-track regions (sky, OHE wires,
horizon, locomotive cowling) to optimize downstream computer vision performance.

NOTE: This module only performs geometric ROI extraction. It does NOT perform defect detection.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class TrackROIConfig:
    """Configurable bounding ratios for front-facing train POV track ROI."""
    top_ratio: float = 0.45      # Cut off sky, horizon, and distant catenary wires
    bottom_ratio: float = 0.98   # Cut off immediate front locomotive nose/buffer
    left_ratio: float = 0.08     # Left boundary covering left rail and sleeper edge
    right_ratio: float = 0.92    # Right boundary covering right rail and sleeper edge
    enable_polygon_mask: bool = False  # Optional perspective trapezoid corridor mask
    target_size: Optional[Tuple[int, int]] = None  # Optional (width, height) resize for ML input


@dataclass
class ROICropMetadata:
    """Metadata describing the cropped ROI dimensions and pixel coordinates."""
    original_width: int
    original_height: int
    ymin: int
    ymax: int
    xmin: int
    xmax: int
    crop_width: int
    crop_height: int


class TrackROIExtractor:
    """Extracts and visualizes deterministic track bed regions of interest."""

    def __init__(self, config: Optional[TrackROIConfig] = None) -> None:
        self.config = config or TrackROIConfig()

    def get_pixel_bounds(self, height: int, width: int) -> Tuple[int, int, int, int]:
        """Calculates absolute pixel bounding coordinates (ymin, ymax, xmin, xmax)."""
        ymin = int(round(self.config.top_ratio * height))
        ymax = int(round(self.config.bottom_ratio * height))
        xmin = int(round(self.config.left_ratio * width))
        xmax = int(round(self.config.right_ratio * width))

        # Clamp to image boundaries
        ymin = max(0, min(ymin, height - 1))
        ymax = max(ymin + 1, min(ymax, height))
        xmin = max(0, min(xmin, width - 1))
        xmax = max(xmin + 1, min(xmax, width))

        return ymin, ymax, xmin, xmax

    def get_metadata(self, height: int, width: int) -> ROICropMetadata:
        """Returns structured metadata for the calculated ROI."""
        ymin, ymax, xmin, xmax = self.get_pixel_bounds(height, width)
        crop_w = xmax - xmin
        crop_h = ymax - ymin
        return ROICropMetadata(
            original_width=width,
            original_height=height,
            ymin=ymin,
            ymax=ymax,
            xmin=xmin,
            xmax=xmax,
            crop_width=crop_w,
            crop_height=crop_h,
        )

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Extracts the track ROI crop from the input image."""
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image passed to crop.")

        h, w = image.shape[:2]
        ymin, ymax, xmin, xmax = self.get_pixel_bounds(h, w)
        cropped = image[ymin:ymax, xmin:xmax]

        if self.config.enable_polygon_mask:
            # Perspective trapezoid mask
            mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
            ch, cw = cropped.shape[:2]
            poly = np.array([
                [int(0.18 * cw), 0],
                [int(0.82 * cw), 0],
                [cw, ch],
                [0, ch],
            ], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
            cropped = cv2.bitwise_and(cropped, cropped, mask=mask)

        if self.config.target_size is not None:
            tw, th = self.config.target_size
            cropped = cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_AREA)

        return cropped

    def annotate(self, image: np.ndarray, shade_outside: bool = True) -> np.ndarray:
        """Draws ROI boundary box, corner markers, and coordinate overlay on original image."""
        if image is None or image.size == 0:
            raise ValueError("Invalid image for annotation.")

        h, w = image.shape[:2]
        ymin, ymax, xmin, xmax = self.get_pixel_bounds(h, w)
        annotated = image.copy()

        if shade_outside:
            # Dim the outside region by 50% to make the track ROI visually prominent
            overlay = image.copy()
            dark_mask = np.ones((h, w), dtype=np.uint8) * 255
            dark_mask[ymin:ymax, xmin:xmax] = 0
            overlay[dark_mask == 255] = (overlay[dark_mask == 255] * 0.45).astype(np.uint8)
            annotated = overlay

        # Draw ROI bounding box (Vibrant cyan/teal)
        box_color = (255, 200, 0)  # BGR
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), box_color, 3)

        # Draw corner brackets for professional overlay look
        c_len = 30
        c_thick = 5
        # Top-left
        cv2.line(annotated, (xmin, ymin), (xmin + c_len, ymin), (0, 255, 255), c_thick)
        cv2.line(annotated, (xmin, ymin), (xmin, ymin + c_len), (0, 255, 255), c_thick)
        # Top-right
        cv2.line(annotated, (xmax, ymin), (xmax - c_len, ymin), (0, 255, 255), c_thick)
        cv2.line(annotated, (xmax, ymin), (xmax, ymin + c_len), (0, 255, 255), c_thick)
        # Bottom-left
        cv2.line(annotated, (xmin, ymax), (xmin + c_len, ymax), (0, 255, 255), c_thick)
        cv2.line(annotated, (xmin, ymax), (xmin, ymax - c_len), (0, 255, 255), c_thick)
        # Bottom-right
        cv2.line(annotated, (xmax, ymax), (xmax - c_len, ymax), (0, 255, 255), c_thick)
        cv2.line(annotated, (xmax, ymax), (xmax, ymax - c_len), (0, 255, 255), c_thick)

        # Label tag
        label = f"ITMS Track ROI: {xmax - xmin}x{ymax - ymin} (Y: {ymin}-{ymax}, X: {xmin}-{xmax})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (xmin, ymin - th - 12), (xmin + tw + 16, ymin), (20, 20, 20), -1)
        cv2.rectangle(annotated, (xmin, ymin - th - 12), (xmin + tw + 16, ymin), box_color, 2)
        cv2.putText(annotated, label, (xmin + 8, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def create_roi_preview_contact_sheet(
        self,
        frames_dir: Path,
        representative_items: List[Dict],
        output_path: Path,
        card_width: int = 720,
    ) -> Path:
        """Builds a side-by-side contact sheet comparing full annotated frame vs cropped track ROI."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cards = []

        for item in representative_items:
            img_file = frames_dir / item["filename"]
            if not img_file.exists():
                continue

            raw_img = cv2.imread(str(img_file))
            if raw_img is None:
                continue

            h, w = raw_img.shape[:2]

            # 1. Annotated full frame (top half of card)
            annotated_full = self.annotate(raw_img, shade_outside=True)
            annotated_thumb = cv2.resize(
                annotated_full, (card_width, int(card_width * h / w)), interpolation=cv2.INTER_AREA
            )

            # 2. Cropped ROI (bottom half of card)
            cropped_roi = self.crop(raw_img)
            ch, cw = cropped_roi.shape[:2]
            cropped_thumb = cv2.resize(
                cropped_roi, (card_width, int(card_width * ch / cw)), interpolation=cv2.INTER_AREA
            )

            # Add header to cropped ROI
            crop_banner = np.zeros((28, card_width, 3), dtype=np.uint8)
            cv2.putText(
                crop_banner,
                f"Track Bed Crop ({cw}x{ch}) | Frame #{item.get('extracted_index', 0)} ({item.get('timestamp_seconds', 0.0):.1f}s)",
                (10, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Assemble single card
            card = np.vstack([annotated_thumb, crop_banner, cropped_thumb])
            cards.append(card)

        if not cards:
            raise ValueError("No valid representative images found to build ROI contact sheet.")

        # Layout in 2 columns
        cols = 2
        rows = math.ceil(len(cards) / cols)
        card_h = cards[0].shape[0]

        while len(cards) < rows * cols:
            cards.append(np.zeros((card_h, card_width, 3), dtype=np.uint8))

        grid_rows = []
        for r in range(rows):
            row_cards = cards[r * cols : (r + 1) * cols]
            grid_rows.append(np.hstack(row_cards))

        contact_sheet = np.vstack(grid_rows)
        cv2.imwrite(str(output_path), contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return output_path
