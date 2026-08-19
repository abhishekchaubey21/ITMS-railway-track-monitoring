"""Vegetation encroachment detection module for ITMS railway track monitoring.

Implements explainable, deterministic color-space and morphological analysis
(Excess Green index ExG + HSV segmentation + morphological filtering) bounded within
the track bed ROI to detect weeds, shrubs, and vegetation encroaching into the railway corridor.

NOTE: This module implements an optical heuristic assessment for maintenance logging
and prototype demonstration. It does NOT claim certified railway safety clearance.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from ai_pipeline.detection.hazard_filter import TrackHazardFilter, EgoTrackCorridorConfig
from ai_pipeline.detection.track_roi import TrackROIExtractor
from ai_pipeline.models.vision_events import VegetationAssessment


@dataclass
class VegetationDetectorConfig:
    """Configuration options for vegetation detection."""

    exg_threshold: float = 15.0
    exg_norm_threshold: float = 0.06
    hue_range: Tuple[int, int] = (25, 88)
    min_saturation: int = 30
    min_value: int = 30
    morph_kernel_size: int = 3
    min_contour_area: int = 30
    score_multiplier: float = 5.0
    intrusion_density_threshold: float = 0.02
    center_corridor_ratio: Tuple[float, float] = (0.32, 0.68)


@dataclass
class VegetationAnalysisDetails:
    """Diagnostic details from vegetation detection."""

    score: float
    intrusion_detected: bool
    vegetation_pixel_count: int
    roi_pixel_count: int
    raw_density: float
    center_vegetation_pixel_count: int
    center_roi_pixel_count: int
    center_density: float
    binary_mask: np.ndarray


class VegetationDetector:
    """Deterministic vegetation encroachment detector using Excess Green (ExG) and HSV morphology."""

    def __init__(
        self,
        config: Optional[VegetationDetectorConfig] = None,
        roi_extractor: Optional[TrackROIExtractor] = None,
        hazard_filter: Optional[TrackHazardFilter] = None,
    ) -> None:
        """Initialize the VegetationDetector.

        Args:
            config: Detection thresholds and morphological parameters.
            roi_extractor: Pre-configured TrackROIExtractor instance.
            hazard_filter: Optional TrackHazardFilter to establish ego-track perspective corridor.
        """
        self.config = config or VegetationDetectorConfig()
        self.roi_extractor = roi_extractor or TrackROIExtractor()
        self.hazard_filter = hazard_filter or TrackHazardFilter()

    def _compute_exg_mask(self, bgr_image: np.ndarray) -> np.ndarray:
        """Compute binary mask where Excess Green index indicates vegetation."""
        b = bgr_image[:, :, 0].astype(np.float32)
        g = bgr_image[:, :, 1].astype(np.float32)
        r = bgr_image[:, :, 2].astype(np.float32)

        # Unnormalized ExG: 2*G - R - B
        exg = 2.0 * g - r - b

        # Normalized ExG: 2*g_norm - r_norm - b_norm
        total = r + g + b + 1e-6
        r_norm = r / total
        g_norm = g / total
        b_norm = b / total
        exg_norm = 2.0 * g_norm - r_norm - b_norm

        exg_mask = (exg > self.config.exg_threshold) & (exg_norm > self.config.exg_norm_threshold)
        return exg_mask.astype(np.uint8) * 255

    def _compute_hsv_mask(self, bgr_image: np.ndarray) -> np.ndarray:
        """Compute binary mask where HSV color properties indicate green vegetation."""
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h_min, h_max = self.config.hue_range
        s_min = self.config.min_saturation
        v_min = self.config.min_value

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def _apply_morphology(self, binary_mask: np.ndarray) -> np.ndarray:
        """Apply morphological opening and closing to suppress noise and bridge foliage."""
        ksize = max(1, self.config.morph_kernel_size)
        if ksize % 2 == 0:
            ksize += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        # Open to remove isolated noise speckles
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Close to bridge adjacent leaves/patches
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # Filter out small contours below minimum area
        if self.config.min_contour_area > 0:
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_mask = np.zeros_like(closed)
            for cnt in contours:
                if cv2.contourArea(cnt) >= self.config.min_contour_area:
                    cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            return filtered_mask

        return closed

    def _build_ego_corridor_mask(
        self,
        roi_shape: Tuple[int, int],
        image_shape: Tuple[int, int],
        roi_bounds: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Constructs a binary mask representing the ego-track corridor within the ROI crop."""
        rh, rw = roi_shape
        h, w = image_shape
        ymin, ymax, xmin, xmax = roi_bounds

        corridor_mask = np.zeros((rh, rw), dtype=np.uint8)

        if self.hazard_filter is not None:
            # Generate trapezoidal perspective polygon in ROI coordinate space
            top_left_x, top_right_x = self.hazard_filter.get_corridor_bounds_at_y(ymin, w, h)
            bot_left_x, bot_right_x = self.hazard_filter.get_corridor_bounds_at_y(ymax, w, h)

            pts = np.array([
                [int(round(top_left_x - xmin)), 0],
                [int(round(top_right_x - xmin)), 0],
                [int(round(bot_right_x - xmin)), rh],
                [int(round(bot_left_x - xmin)), rh],
            ], dtype=np.int32)

            cv2.fillPoly(corridor_mask, [pts], 255)
        else:
            # Fallback to configured center ratio
            cx_min = int(rw * self.config.center_corridor_ratio[0])
            cx_max = int(rw * self.config.center_corridor_ratio[1])
            corridor_mask[:, cx_min:cx_max] = 255

        return corridor_mask

    def analyze(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> VegetationAnalysisDetails:
        """Perform comprehensive vegetation analysis on input image.

        Args:
            image: Full-frame BGR image.
            frame_id: Frame index.
            timestamp: Video timestamp in seconds.

        Returns:
            VegetationAnalysisDetails containing score, mask, and diagnostic metrics.
        """
        if image is None:
            raise ValueError("Input image cannot be None")
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Input image must be a numpy ndarray, got {type(image)}")
        if image.size == 0 or len(image.shape) < 2 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Input image array is empty or has invalid shape")
        if frame_id < 0:
            raise ValueError(f"frame_id must be non-negative, got {frame_id}")
        if timestamp < 0.0:
            raise ValueError(f"timestamp must be non-negative, got {timestamp}")

        h, w = image.shape[:2]

        # 1. Extract Track Bed ROI
        if self.roi_extractor:
            roi_crop = self.roi_extractor.crop(image)
            ymin, ymax, xmin, xmax = self.roi_extractor.get_pixel_bounds(h, w)
        else:
            roi_crop = image
            ymin, ymax, xmin, xmax = 0, h, 0, w

        rh, rw = roi_crop.shape[:2]
        roi_pixels = rh * rw

        if roi_pixels == 0:
            empty_mask = np.zeros((0, 0), dtype=np.uint8)
            return VegetationAnalysisDetails(
                score=0.0,
                intrusion_detected=False,
                vegetation_pixel_count=0,
                roi_pixel_count=0,
                raw_density=0.0,
                center_vegetation_pixel_count=0,
                center_roi_pixel_count=0,
                center_density=0.0,
                binary_mask=empty_mask,
            )

        # 2. Compute ExG & HSV masks
        exg_mask = self._compute_exg_mask(roi_crop)
        hsv_mask = self._compute_hsv_mask(roi_crop)
        combined_mask = cv2.bitwise_and(exg_mask, hsv_mask)

        # 3. Morphological filtering
        clean_mask = self._apply_morphology(combined_mask)

        # 4. Compute overall vegetation density & score across track bed
        veg_pixels = int(np.count_nonzero(clean_mask))
        raw_density = veg_pixels / float(roi_pixels)
        scaled_score = min(1.0, max(0.0, raw_density * self.config.score_multiplier))

        # 5. Compute ego-track corridor encroachment
        ego_corridor_mask = self._build_ego_corridor_mask(
            roi_shape=(rh, rw),
            image_shape=(h, w),
            roi_bounds=(ymin, ymax, xmin, xmax),
        )

        ego_veg_mask = cv2.bitwise_and(clean_mask, ego_corridor_mask)
        center_veg_pixels = int(np.count_nonzero(ego_veg_mask))
        center_roi_pixels = int(np.count_nonzero(ego_corridor_mask))

        if center_roi_pixels > 0:
            center_density = center_veg_pixels / float(center_roi_pixels)
        else:
            center_density = 0.0

        # Intrusion is triggered strictly if vegetation encroaches into the active ego-track corridor
        intrusion_detected = center_density >= self.config.intrusion_density_threshold

        return VegetationAnalysisDetails(
            score=round(scaled_score, 4),
            intrusion_detected=intrusion_detected,
            vegetation_pixel_count=veg_pixels,
            roi_pixel_count=roi_pixels,
            raw_density=raw_density,
            center_vegetation_pixel_count=center_veg_pixels,
            center_roi_pixel_count=center_roi_pixels,
            center_density=center_density,
            binary_mask=clean_mask,
        )

    def assess(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> VegetationAssessment:
        """Perform vegetation assessment and return validated VegetationAssessment schema.

        Args:
            image: Full-frame BGR image.
            frame_id: Frame index.
            timestamp: Video timestamp in seconds.

        Returns:
            VegetationAssessment object.
        """
        details = self.analyze(image, frame_id=frame_id, timestamp=timestamp)
        return VegetationAssessment(
            score=details.score,
            intrusion_detected=details.intrusion_detected,
        )

    def detect(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> VegetationAssessment:
        """Alias for assess() to maintain polymorphic detector interface."""
        return self.assess(image, frame_id=frame_id, timestamp=timestamp)

    def __call__(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> VegetationAssessment:
        """Callable interface for VisionPipeline integration."""
        return self.assess(image, frame_id=frame_id, timestamp=timestamp)
