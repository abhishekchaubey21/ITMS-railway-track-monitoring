"""Unit tests for VegetationDetector module."""

import cv2
import numpy as np
import pytest

from ai_pipeline.detection.track_roi import TrackROIExtractor, TrackROIConfig
from ai_pipeline.detection.vegetation_detector import (
    VegetationDetector,
    VegetationDetectorConfig,
    VegetationAnalysisDetails,
    EgoRunningGaugeConfig,
)
from ai_pipeline.detection.vision_pipeline import VisionPipeline
from ai_pipeline.models.vision_events import VegetationAssessment, VisionResult


def test_vegetation_detector_initialization():
    """Verify default initialization and configuration."""
    detector = VegetationDetector()
    assert isinstance(detector.config, VegetationDetectorConfig)
    assert isinstance(detector.roi_extractor, TrackROIExtractor)
    assert isinstance(detector.gauge_config, EgoRunningGaugeConfig)
    assert detector.config.exg_threshold == 15.0
    assert detector.config.min_contour_area == 30
    assert detector.gauge_config.safety_margin_ratio == 0.10


def test_vegetation_detector_invalid_inputs():
    """Verify error handling on invalid, empty, or wrong-type images."""
    detector = VegetationDetector()

    # None input
    with pytest.raises(ValueError, match="Input image cannot be None"):
        detector.assess(None)

    # Wrong type
    with pytest.raises(TypeError, match="Input image must be a numpy ndarray"):
        detector.assess("not_an_image")

    # Empty array
    with pytest.raises(ValueError, match="empty or has invalid shape"):
        detector.assess(np.array([]))

    # 0-dimension shape
    with pytest.raises(ValueError, match="empty or has invalid shape"):
        detector.assess(np.zeros((0, 0, 3), dtype=np.uint8))

    # Negative indices
    valid_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="frame_id must be non-negative"):
        detector.assess(valid_img, frame_id=-1)

    with pytest.raises(ValueError, match="timestamp must be non-negative"):
        detector.assess(valid_img, frame_id=0, timestamp=-0.5)


def test_no_vegetation_synthetic_ballast():
    """Verify that pure grey/ballast and concrete surfaces yield 0.0 vegetation score."""
    detector = VegetationDetector()

    # Synthetic grey ballast image (1920x1080)
    grey_img = np.full((1080, 1920, 3), fill_value=128, dtype=np.uint8)
    assessment = detector.assess(grey_img)

    assert isinstance(assessment, VegetationAssessment)
    assert assessment.score == 0.0
    assert assessment.intrusion_detected is False

    # Synthetic brown/rust surface (R > G > B)
    rust_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    rust_img[:, :, 2] = 160  # Red
    rust_img[:, :, 1] = 90   # Green
    rust_img[:, :, 0] = 50   # Blue

    rust_assessment = detector.assess(rust_img)
    assert rust_assessment.score == 0.0
    assert rust_assessment.intrusion_detected is False


def test_obvious_vegetation_detection():
    """Verify detection and scoring of prominent green vegetation in track ROI."""
    detector = VegetationDetector()

    # Create base dark grey image (ballast)
    img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)

    # Place a prominent green vegetation patch inside track ROI & running gauge (Y: 700-850, X: 900-1020)
    # Bright green BGR: B=30, G=180, R=40
    img[700:850, 900:1020] = [30, 180, 40]

    details: VegetationAnalysisDetails = detector.analyze(img, frame_id=1, timestamp=0.5)

    assert details.vegetation_pixel_count > 0
    assert details.score > 0.0
    assert details.raw_density > 0.0
    assert details.intrusion_detected is True

    assessment = detector.assess(img)
    assert isinstance(assessment, VegetationAssessment)
    assert assessment.score > 0.0
    assert 0.0 <= assessment.score <= 1.0
    assert assessment.intrusion_detected is True


def test_threshold_behavior_and_sensitivity():
    """Verify that adjusting exg_threshold or min_saturation changes detector sensitivity."""
    # Subtle pale green patch
    img = np.full((1080, 1920, 3), fill_value=100, dtype=np.uint8)
    img[600:750, 850:1050] = [70, 120, 75]  # Mild excess green

    # Standard detector
    det_standard = VegetationDetector(config=VegetationDetectorConfig(exg_threshold=15.0))
    res_std = det_standard.assess(img)

    # Strict high-threshold detector
    det_strict = VegetationDetector(config=VegetationDetectorConfig(exg_threshold=150.0))
    res_strict = det_strict.assess(img)

    assert res_std.score > res_strict.score
    assert res_strict.score == 0.0


def test_morphology_noise_suppression():
    """Verify that isolated 1-pixel noise speckles are filtered out by morphology."""
    config = VegetationDetectorConfig(min_contour_area=30, morph_kernel_size=3)
    detector = VegetationDetector(config=config)

    img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)

    # Scatter a few isolated 1-pixel bright green noise dots in ROI
    for (y, x) in [(600, 700), (620, 850), (710, 920), (780, 1050)]:
        img[y, x] = [30, 200, 30]

    details = detector.analyze(img)
    # Morphological opening and minimum contour area should completely eliminate isolated 1-pixel noise
    assert details.vegetation_pixel_count == 0
    assert details.score == 0.0
    assert details.intrusion_detected is False


def test_roi_restriction_behavior():
    """Verify that vegetation outside track ROI (e.g. sky or extreme periphery) is ignored."""
    roi_config = TrackROIConfig(top_ratio=0.45, bottom_ratio=0.98, left_ratio=0.08, right_ratio=0.92)
    roi_extractor = TrackROIExtractor(config=roi_config)
    detector = VegetationDetector(roi_extractor=roi_extractor)

    # Create image with green patch located exclusively in top sky region (Y: 50-200, X: 500-1000)
    # ROI starts at Y = 0.45 * 1080 = 486
    sky_green_img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)
    sky_green_img[50:200, 500:1000] = [20, 200, 20]

    assessment_sky = detector.assess(sky_green_img)
    assert assessment_sky.score == 0.0
    assert assessment_sky.intrusion_detected is False

    # Now add green patch inside track ROI (Y: 600-750)
    track_green_img = sky_green_img.copy()
    track_green_img[600:750, 850:1050] = [20, 200, 20]

    assessment_track = detector.assess(track_green_img)
    assert assessment_track.score > 0.0


def test_vegetation_beside_right_ego_rail_is_not_intrusion():
    """Vegetation on the right-side fence/embankment/shoulder must yield score > 0 but intrusion = False."""
    detector = VegetationDetector()
    img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)

    # Place vegetation patch on right side beside right rail (X: 1220-1450, Y: 700-850)
    img[700:850, 1220:1450] = [30, 180, 40]

    details = detector.analyze(img)
    assert details.score > 0.0  # Captured in general vegetation score
    assert details.intrusion_detected is False  # Outside the ego running gauge


def test_vegetation_beside_left_ego_rail_is_not_intrusion():
    """Vegetation on the left-side cess/embankment must yield score > 0 but intrusion = False."""
    detector = VegetationDetector()
    img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)

    # Place vegetation patch on left side beside left rail (X: 520-720, Y: 700-850)
    img[700:850, 520:720] = [30, 180, 40]

    details = detector.analyze(img)
    assert details.score > 0.0  # Captured in general vegetation score
    assert details.intrusion_detected is False  # Outside the ego running gauge


def test_vegetation_on_adjacent_track_is_not_intrusion():
    """Vegetation on far-left/right adjacent parallel tracks must yield score > 0 but intrusion = False."""
    detector = VegetationDetector()
    img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)

    # Place vegetation patch on far-left adjacent track (X: 180-350, Y: 600-800)
    img[600:800, 180:350] = [30, 180, 40]

    details = detector.analyze(img)
    assert details.score > 0.0
    assert details.intrusion_detected is False


def test_vegetation_between_two_ego_rails_is_intrusion():
    """Vegetation growing directly between the two ego running rails triggers intrusion = True."""
    detector = VegetationDetector()
    img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)

    # Place vegetation patch strictly inside running gauge (X: 900-1020, Y: 700-850)
    img[700:850, 900:1020] = [30, 180, 40]

    details = detector.analyze(img)
    assert details.score > 0.0
    assert details.intrusion_detected is True
    assert details.center_vegetation_pixel_count > 0


def test_vegetation_crossing_into_gauge_is_intrusion():
    """Vegetation originating outside but overhanging/crossing into the running gauge triggers intrusion = True."""
    detector = VegetationDetector()
    img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)

    # Place vegetation patch spanning across right rail into the center gauge (X: 880-1350, Y: 700-850)
    img[700:850, 880:1350] = [30, 180, 40]

    details = detector.analyze(img)
    assert details.score > 0.0
    assert details.intrusion_detected is True
    assert details.center_vegetation_pixel_count > 0


def test_general_vegetation_score_increases_for_vegetation_outside_gauge():
    """Vegetation density/score increases when trackside vegetation is present without triggering intrusion."""
    detector = VegetationDetector()

    # Base image: no vegetation
    base_img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)
    det_base = detector.analyze(base_img)
    assert det_base.score == 0.0
    assert det_base.intrusion_detected is False

    # Image 1: Right-side trackside vegetation
    img_right = base_img.copy()
    img_right[700:850, 1250:1450] = [30, 180, 40]
    det_right = detector.analyze(img_right)
    assert det_right.score > 0.0
    assert det_right.intrusion_detected is False

    # Image 2: Add left-side embankment vegetation as well
    img_both = img_right.copy()
    img_both[600:750, 200:400] = [30, 180, 40]
    det_both = detector.analyze(img_both)
    assert det_both.score > det_right.score  # General score increases with more trackside vegetation
    assert det_both.intrusion_detected is False  # Still no ego-track gauge intrusion


def test_configurable_safety_margin_behavior():
    """Verify that increasing safety_margin_ratio shrinks the inner gauge corridor."""
    gauge_std = EgoRunningGaugeConfig(safety_margin_ratio=0.10)
    gauge_strict = EgoRunningGaugeConfig(safety_margin_ratio=0.25)

    l_std, r_std = gauge_std.get_inner_gauge_bounds_at_y(800, 1920, 1080)
    l_str, r_str = gauge_strict.get_inner_gauge_bounds_at_y(800, 1920, 1080)

    width_std = r_std - l_std
    width_strict = r_str - l_str
    assert width_strict < width_std


def test_detector_interfaces_and_polymorphism():
    """Verify analyze(), assess(), detect(), and __call__() all return consistent outputs."""
    detector = VegetationDetector()
    img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)
    img[700:850, 900:1020] = [30, 190, 30]

    res_assess = detector.assess(img, frame_id=5, timestamp=1.5)
    res_detect = detector.detect(img, frame_id=5, timestamp=1.5)
    res_call = detector(img, frame_id=5, timestamp=1.5)

    assert res_assess == res_detect
    assert res_assess == res_call
    assert isinstance(res_assess, VegetationAssessment)


def test_integration_with_vision_pipeline():
    """Verify VegetationDetector integrates cleanly into VisionPipeline via vegetation_assessor hook."""
    veg_detector = VegetationDetector()

    class MockDetector:
        def detect(self, image, frame_id=0, timestamp=0.0):
            return []

    pipeline = VisionPipeline(
        obstacle_detector=MockDetector(),
        vegetation_assessor=veg_detector,
    )

    # Frame with green vegetation in track corridor
    img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)
    img[700:850, 900:1020] = [30, 180, 40]

    result: VisionResult = pipeline.process_frame(img, frame_id=12, timestamp=2.0)

    assert isinstance(result, VisionResult)
    assert result.vegetation is not None
    assert isinstance(result.vegetation, VegetationAssessment)
    assert result.vegetation.score > 0.0
    assert result.vegetation.intrusion_detected is True
