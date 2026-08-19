"""Unit tests for VegetationDetector module."""

import cv2
import numpy as np
import pytest

from ai_pipeline.detection.track_roi import TrackROIExtractor, TrackROIConfig
from ai_pipeline.detection.vegetation_detector import (
    VegetationDetector,
    VegetationDetectorConfig,
    VegetationAnalysisDetails,
)
from ai_pipeline.detection.vision_pipeline import VisionPipeline
from ai_pipeline.models.vision_events import VegetationAssessment, VisionResult


def test_vegetation_detector_initialization():
    """Verify default initialization and configuration."""
    detector = VegetationDetector()
    assert isinstance(detector.config, VegetationDetectorConfig)
    assert isinstance(detector.roi_extractor, TrackROIExtractor)
    assert detector.config.exg_threshold == 15.0
    assert detector.config.min_contour_area == 30


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

    # Place a prominent green vegetation patch inside track ROI (Y: 600-800, X: 800-1100)
    # Bright green BGR: B=30, G=180, R=40
    img[600:800, 800:1100] = [30, 180, 40]

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
    img[600:750, 800:1000] = [70, 120, 75]  # Mild excess green (ExG = 240 - 75 - 70 = 95)

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
    track_green_img[600:750, 800:1000] = [20, 200, 20]

    assessment_track = detector.assess(track_green_img)
    assert assessment_track.score > 0.0


def test_detector_interfaces_and_polymorphism():
    """Verify analyze(), assess(), detect(), and __call__() all return consistent outputs."""
    detector = VegetationDetector()
    img = np.full((1080, 1920, 3), fill_value=80, dtype=np.uint8)
    img[600:800, 800:1100] = [30, 190, 30]

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
    img[600:800, 800:1100] = [30, 180, 40]

    result: VisionResult = pipeline.process_frame(img, frame_id=12, timestamp=2.0)

    assert isinstance(result, VisionResult)
    assert result.vegetation is not None
    assert isinstance(result.vegetation, VegetationAssessment)
    assert result.vegetation.score > 0.0
    assert result.vegetation.intrusion_detected is True
