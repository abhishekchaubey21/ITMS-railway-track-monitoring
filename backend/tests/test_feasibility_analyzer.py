"""Unit tests for feasibility analysis module."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from ai_pipeline.detection.feasibility_analyzer import (
    FeasibilityAnalyzer,
    DetectabilityRating,
    VisibilityLevel,
)


def test_feasibility_targets_definition():
    """Verify systematic standard targets configuration and rating integrity."""
    analyzer = FeasibilityAnalyzer()
    targets = analyzer.get_standard_inspection_targets()

    assert len(targets) >= 8

    ratings = {t.detectability_rating for t in targets}
    assert DetectabilityRating.HIGH in ratings
    assert DetectabilityRating.MEDIUM in ratings
    assert DetectabilityRating.LOW in ratings
    assert DetectabilityRating.NOT_FEASIBLE in ratings

    for t in targets:
        assert len(t.target_name) > 0
        assert len(t.reasoning) > 0
        assert len(t.recommended_cv_approach) > 0
        assert len(t.key_limitations) > 0


def test_feasibility_analyzer_build_report():
    """Verify structure and calculations of feasibility study summary."""
    analyzer = FeasibilityAnalyzer()
    manifest_data = {
        "source_filename": "test_video.mp4",
        "resolution": [1920, 1080],
        "frames": [{"filename": "f1.jpg"}, {"filename": "f2.jpg"}],
    }

    summary = analyzer.build_feasibility_report(manifest_data)

    assert summary.dataset_name == "test_video.mp4"
    assert summary.source_resolution == (1920, 1080)
    assert summary.roi_resolution == (1612, 572)
    assert summary.evaluated_frame_count == 2
    assert (
        summary.high_feasibility_count
        + summary.medium_feasibility_count
        + summary.low_feasibility_count
        + summary.not_feasible_count
    ) == len(summary.targets)


def test_feasibility_analyzer_run_analysis_synthetic(tmp_path: Path):
    """Verify end-to-end execution of feasibility analysis on synthetic files."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    out_dir = tmp_path / "inspection"
    docs_dir = tmp_path / "docs"

    frames_list = []
    for i in range(3):
        fn = f"frame_{i+1:06d}.jpg"
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[500:1000, 200:1700] = 120
        cv2.imwrite(str(frames_dir / fn), img)
        frames_list.append(
            {
                "extracted_index": i + 1,
                "source_frame_index": i * 5,
                "timestamp_seconds": float(i),
                "filename": fn,
            }
        )

    manifest_path = frames_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_filename": "synthetic.mp4",
                "resolution": [1920, 1080],
                "frames": frames_list,
            },
            f,
        )

    analyzer = FeasibilityAnalyzer()
    summary = analyzer.run_analysis(
        manifest_path=manifest_path,
        output_dir=out_dir,
        docs_dir=docs_dir,
    )

    # 1. Verify JSON report
    json_path = out_dir / "feasibility_report.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["dataset_name"] == "synthetic.mp4"
    assert len(data["targets"]) == len(summary.targets)

    # 2. Verify contact sheet
    contact_sheet = Path(summary.contact_sheet_path)
    assert contact_sheet.exists()
    sheet_img = cv2.imread(str(contact_sheet))
    assert sheet_img is not None
    assert sheet_img.shape[0] > 0 and sheet_img.shape[1] > 0

    # 3. Verify Markdown doc
    md_path = docs_dir / "feasibility_assessment.md"
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "# ITMS — Railway Track Inspection Feasibility Assessment" in md_text
    assert "Foreign Objects & Track Obstructions" in md_text
    assert "Internal / Subsurface Rail Flaws" in md_text
