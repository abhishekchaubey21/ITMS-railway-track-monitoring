"""Unit tests for deterministic track ROI extraction module."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from ai_pipeline.detection.track_roi import TrackROIExtractor, TrackROIConfig, ROICropMetadata


def test_track_roi_bounds_calculation():
    """Verify pixel bounds calculation for 1080p frame."""
    config = TrackROIConfig(
        top_ratio=0.45,
        bottom_ratio=0.98,
        left_ratio=0.08,
        right_ratio=0.92,
    )
    extractor = TrackROIExtractor(config=config)
    ymin, ymax, xmin, xmax = extractor.get_pixel_bounds(height=1080, width=1920)

    assert ymin == 486
    assert ymax == 1058
    assert xmin == 154
    assert xmax == 1766

    meta: ROICropMetadata = extractor.get_metadata(height=1080, width=1920)
    assert meta.original_width == 1920
    assert meta.original_height == 1080
    assert meta.crop_width == 1612
    assert meta.crop_height == 572


def test_track_roi_crop_synthetic_image():
    """Verify image cropping produces expected sub-array."""
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img[486:1058, 154:1766] = 200  # Fill ROI with white

    extractor = TrackROIExtractor()
    cropped = extractor.crop(img)

    assert cropped.shape == (572, 1612, 3)
    assert np.all(cropped == 200)


def test_track_roi_target_size_resize():
    """Verify optional resizing to fixed model dimensions."""
    config = TrackROIConfig(target_size=(640, 384))
    extractor = TrackROIExtractor(config=config)

    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cropped = extractor.crop(img)

    assert cropped.shape == (384, 640, 3)


def test_track_roi_polygon_mask():
    """Verify trapezoid polygon masking zeros out upper corner regions."""
    config = TrackROIConfig(enable_polygon_mask=True)
    extractor = TrackROIExtractor(config=config)

    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    cropped = extractor.crop(img)

    assert cropped.shape == (572, 1612, 3)
    # Top-left corner of the crop should be masked to 0
    assert np.all(cropped[0, 0] == 0)
    # Bottom center should remain untouched (255)
    assert np.all(cropped[-1, 800] == 255)


def test_track_roi_annotation():
    """Verify annotation preserves frame shape and applies overlays."""
    extractor = TrackROIExtractor()
    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 100
    annotated = extractor.annotate(img, shade_outside=True)

    assert annotated.shape == (1080, 1920, 3)
    # Shaded top region should be darker than 100
    assert annotated[50, 50, 0] < 100
    # Inside ROI should retain brightness
    assert annotated[600, 600, 0] == 100


def test_track_roi_preview_contact_sheet_synthetic(tmp_path: Path):
    """Verify generation of side-by-side ROI comparison contact sheet."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    items = []
    for i in range(2):
        fn = f"frame_{i+1:06d}.jpg"
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[100:220, 30:290] = 180
        cv2.imwrite(str(frames_dir / fn), img)
        items.append({"filename": fn, "extracted_index": i + 1, "timestamp_seconds": float(i)})

    out_file = tmp_path / "roi_sheet.jpg"
    extractor = TrackROIExtractor()
    extractor.create_roi_preview_contact_sheet(
        frames_dir=frames_dir,
        representative_items=items,
        output_path=out_file,
        card_width=320,
    )

    assert out_file.exists()
    sheet = cv2.imread(str(out_file))
    assert sheet is not None
    assert sheet.shape[0] > 0 and sheet.shape[1] > 0


def test_track_roi_invalid_image():
    """Verify error on empty or None image input."""
    extractor = TrackROIExtractor()
    with pytest.raises(ValueError):
        extractor.crop(None)

    with pytest.raises(ValueError):
        extractor.crop(np.array([]))
