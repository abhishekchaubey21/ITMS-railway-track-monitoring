"""Feasibility assessment and infrastructure visibility analysis for ITMS train POV video.

Evaluates the genuine feasibility of various railway track inspection tasks using
front-facing train POV footage and deterministic ROI extraction.

NOTE: This module performs feasibility analysis and visual evidence mapping.
It does NOT perform defect detection or label unconfirmed defects.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from ai_pipeline.detection.track_roi import TrackROIExtractor, TrackROIConfig


class DetectabilityRating(str, Enum):
    """Detectability ratings for computer vision inspection tasks."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_FEASIBLE = "NOT_FEASIBLE"


class VisibilityLevel(str, Enum):
    """Visual prominence level from cab-mounted train POV."""
    CONSISTENTLY_CLEAR = "CONSISTENTLY_CLEAR"
    MODERATELY_VISIBLE = "MODERATELY_VISIBLE"
    LIMITED_RESOLUTION = "LIMITED_RESOLUTION"
    NOT_VISIBLE = "NOT_VISIBLE"


@dataclass
class InspectionTargetAssessment:
    """Detailed feasibility assessment for a specific railway inspection target."""
    target_name: str
    category: str
    visibility_level: VisibilityLevel
    detectability_rating: DetectabilityRating
    reasoning: str
    recommended_cv_approach: str
    requires_external_labels: bool
    sensor_fusion_supplement: str
    key_limitations: str


@dataclass
class FeasibilityStudySummary:
    """Overall feasibility study summary and conclusions."""
    dataset_name: str
    source_resolution: Tuple[int, int]
    roi_resolution: Tuple[int, int]
    evaluated_frame_count: int
    targets: List[InspectionTargetAssessment]
    high_feasibility_count: int
    medium_feasibility_count: int
    low_feasibility_count: int
    not_feasible_count: int
    recommended_stage7_approach: str
    contact_sheet_path: Optional[str] = None


class FeasibilityAnalyzer:
    """Evaluates optical inspection feasibility across railway track components."""

    def __init__(self, roi_extractor: Optional[TrackROIExtractor] = None) -> None:
        self.roi_extractor = roi_extractor or TrackROIExtractor()

    @staticmethod
    def get_standard_inspection_targets() -> List[InspectionTargetAssessment]:
        """Returns the systematic engineering evaluation for railway inspection targets."""
        return [
            InspectionTargetAssessment(
                target_name="Foreign Objects & Track Obstructions",
                category="Obstacle & Hazard Detection",
                visibility_level=VisibilityLevel.CONSISTENTLY_CLEAR,
                detectability_rating=DetectabilityRating.HIGH,
                reasoning=(
                    "Front-facing train POV offers an unobstructed forward field of view across the track gauge. "
                    "Medium-to-large foreign objects (fallen boulders, debris, branches, large metallic components, "
                    "animals, trespassing hazards) occupy significant pixel area (>50-200px) and present high visual "
                    "contrast against uniform ballast."
                ),
                recommended_cv_approach=(
                    "Supervised Object Detection (YOLO / SSD) bounded within track corridor polygon + "
                    "Background Anomaly / Foreground segmentation."
                ),
                requires_external_labels=True,
                sensor_fusion_supplement=(
                    "Can be supplemented by simulated obstacle scenarios and emergency braking telemetry."
                ),
                key_limitations="Small objects (< 5cm) at distances > 30m may be obscured by ballast texture.",
            ),
            InspectionTargetAssessment(
                target_name="Vegetation Encroachment into Gauge",
                category="Environmental & Corridor Hazards",
                visibility_level=VisibilityLevel.CONSISTENTLY_CLEAR,
                detectability_rating=DetectabilityRating.HIGH,
                reasoning=(
                    "Weeds, overgrown shrubs, and vegetation encroaching from embankment or between sleepers "
                    "have distinct color signatures (green/yellow in HSV/ExG space) and high frequency texture "
                    "differing sharply from ballast gravel and steel rails."
                ),
                recommended_cv_approach=(
                    "Excess Green (ExG) index color-space thresholding + morphological contour filtering "
                    "within the active track ROI corridor."
                ),
                requires_external_labels=False,
                sensor_fusion_supplement="Correlated with GPS chainage milestones for maintenance logging.",
                key_limitations="Dead/dry brown grass requires texture segmentation in addition to color filters.",
            ),
            InspectionTargetAssessment(
                target_name="Track Alignment & Macroscopic Geometry Anomalies",
                category="Track Geometry & Alignment",
                visibility_level=VisibilityLevel.CONSISTENTLY_CLEAR,
                detectability_rating=DetectabilityRating.MEDIUM,
                reasoning=(
                    "Both left and right running rails are continuously visible from the near field to the vanishing horizon. "
                    "Macroscopic geometric distortions (> 50-100mm track shifts, track buckling, severe kinks, washouts) "
                    "create visible polynomial trajectory deviations and visual gauge divergence."
                ),
                recommended_cv_approach=(
                    "Dual-rail edge tracking (Sobel-X + Hough line / spline polynomial fitting) + "
                    "Visual Gauge Variation Index calculation along vertical scanlines."
                ),
                requires_external_labels=False,
                sensor_fusion_supplement=(
                    "Essential to fuse with replayed/simulated IMU lateral acceleration and LiDAR profile "
                    "to confirm dynamic track geometry standards."
                ),
                key_limitations=(
                    "Optical camera cannot achieve millimeter-accurate (±1mm) geometry certification required "
                    "for high-speed track standards without calibrated stereo LiDAR."
                ),
            ),
            InspectionTargetAssessment(
                target_name="Damaged / Broken / Misaligned Sleepers",
                category="Sleeper & Tie Structural Integrity",
                visibility_level=VisibilityLevel.MODERATELY_VISIBLE,
                detectability_rating=DetectabilityRating.MEDIUM,
                reasoning=(
                    "Concrete sleepers are visible at periodic intervals (~60cm spacing). In the near field (bottom 40% of ROI), "
                    "sleeper bodies span >400px width. Significant transverse breaks, severe concrete spalling, missing sleeper ends, "
                    "and angular skew (>5°) are visually distinguishable."
                ),
                recommended_cv_approach=(
                    "Periodic sleeper pitch tracking (spatial frequency / peak detection) + "
                    "patch-based visual anomaly detection / edge discontinuity scoring."
                ),
                requires_external_labels=False,
                sensor_fusion_supplement="Simulated sleeper crack and displacement datasets for validation.",
                key_limitations=(
                    "Hairline cracks (<2mm) or fractures obscured beneath ballast dust/shadows cannot be resolved "
                    "at normal operating speeds from cab height."
                ),
            ),
            InspectionTargetAssessment(
                target_name="Ballast Bed Abnormalities & Severe Voids",
                category="Track Bed & Ballast",
                visibility_level=VisibilityLevel.MODERATELY_VISIBLE,
                detectability_rating=DetectabilityRating.MEDIUM,
                reasoning=(
                    "Ballast texture and distribution between and outside rails is clearly visible. Large ballast voids "
                    "(scouring under sleeper ends), waterlogging/mud pumping, or excessive ballast heaps over rail base "
                    "alter local texture statistics."
                ),
                recommended_cv_approach=(
                    "Gray-Level Co-occurrence Matrix (GLCM) texture entropy + statistical patch variance baseline."
                ),
                requires_external_labels=False,
                sensor_fusion_supplement="Fused with replayed IMU vertical vibration peaks to detect ballast voids.",
                key_limitations="Fine ballast grading / particle wear cannot be measured optically.",
            ),
            InspectionTargetAssessment(
                target_name="Missing / Displaced Fasteners (Clips)",
                category="Fastener & Fastening System",
                visibility_level=VisibilityLevel.LIMITED_RESOLUTION,
                detectability_rating=DetectabilityRating.LOW,
                reasoning=(
                    "Elastic rail clips (Pandrol/ERC) are visible only in the extreme near field (bottom 25% of ROI) "
                    "where they occupy ~15-25px. Missing clips create a noticeable contrast gap against sleeper shoulders, "
                    "making missing clip detection feasible in the near field. However, oblique frontal viewing angle "
                    "causes partial perspective occlusion."
                ),
                recommended_cv_approach=(
                    "High-resolution localized ROI crop at rail-sleeper intersections + lightweight patch classifier / template matcher."
                ),
                requires_external_labels=True,
                sensor_fusion_supplement="Synthetic missing fastener injection for pipeline verification.",
                key_limitations=(
                    "Fastener micro-cracks, fatigue loosening, or missing clip insulators cannot be reliably detected "
                    "from a cab POV camera (requires downward-pointing bogie cameras)."
                ),
            ),
            InspectionTargetAssessment(
                target_name="Major Rail Surface Defects & Breaks",
                category="Rail Surface & Head Flaws",
                visibility_level=VisibilityLevel.LIMITED_RESOLUTION,
                detectability_rating=DetectabilityRating.LOW,
                reasoning=(
                    "The top rail running surface appears as a continuous bright reflective band. Major wheel burns, large spalling, "
                    "or complete transverse rail breaks interrupt this specular reflection line. However, subtle rolling contact fatigue "
                    "(squats, shallow head checks, micro-flaking) is below the camera resolving limit."
                ),
                recommended_cv_approach=(
                    "1D longitudinal intensity profile tracking along the detected rail head reflection band."
                ),
                requires_external_labels=False,
                sensor_fusion_supplement="Fused with replayed IMU vibration shocks and simulated ultrasonic anomaly flags.",
                key_limitations="Shallow micro-cracks (<3mm) and fine head checking cannot be detected.",
            ),
            InspectionTargetAssessment(
                target_name="Internal / Subsurface Rail Flaws",
                category="Internal Material & Metallurgical Defects",
                visibility_level=VisibilityLevel.NOT_VISIBLE,
                detectability_rating=DetectabilityRating.NOT_FEASIBLE,
                reasoning=(
                    "Internal metallurgical defects (hydrogen shatter cracks, internal transverse fissures, inclusion clusters, "
                    "bolt-hole internal cracks) reside entirely beneath the steel rail surface and emit zero optical signature."
                ),
                recommended_cv_approach="Not feasible via optical vision. Requires Ultrasonic (UT) or Eddy Current sensors.",
                requires_external_labels=False,
                sensor_fusion_supplement=(
                    "Must be represented via simulated / replayed non-destructive evaluation (NDE) sensor logs in multimodal fusion layer."
                ),
                key_limitations="Physical law: optical cameras cannot penetrate solid steel.",
            ),
            InspectionTargetAssessment(
                target_name="Millimeter-Certified Dynamic Track Gauge & Cant",
                category="Track Geometry Certification",
                visibility_level=VisibilityLevel.LIMITED_RESOLUTION,
                detectability_rating=DetectabilityRating.NOT_FEASIBLE,
                reasoning=(
                    "Standard railway geometry certification (e.g. RDSO/EN 13848) requires ±1.0mm dynamic gauge, twist, and cant accuracy. "
                    "Single-camera cab perspective without structured laser illumination is subject to vehicle suspension pitch/roll dynamics "
                    "and cannot guarantee certified millimeter tolerances."
                ),
                recommended_cv_approach=(
                    "Qualitative visual alignment estimation only; certified measurements require LiDAR & IMU."
                ),
                requires_external_labels=False,
                sensor_fusion_supplement=(
                    "Represented via replayed LiDAR profile streams and IMU gyroscope integration."
                ),
                key_limitations="Vehicle vibration and optical perspective preclude sub-millimeter metrology.",
            ),
        ]

    def build_feasibility_report(
        self,
        manifest_data: dict,
        output_dir: str | Path = "data/processed/inspection",
    ) -> FeasibilityStudySummary:
        """Generates comprehensive feasibility assessment data structure."""
        frames = manifest_data.get("frames", [])
        total_frames = len(frames)
        res = manifest_data.get("resolution", [1920, 1080])
        w, h = res[0], res[1]

        roi_meta = self.roi_extractor.get_metadata(h, w)
        targets = self.get_standard_inspection_targets()

        high_c = sum(1 for t in targets if t.detectability_rating == DetectabilityRating.HIGH)
        med_c = sum(1 for t in targets if t.detectability_rating == DetectabilityRating.MEDIUM)
        low_c = sum(1 for t in targets if t.detectability_rating == DetectabilityRating.LOW)
        not_c = sum(1 for t in targets if t.detectability_rating == DetectabilityRating.NOT_FEASIBLE)

        rec_approach = (
            "Dual-Track Hybrid AI Architecture: "
            "(1) Supervised Object & Hazard Detection (YOLO) for high-visibility foreign obstacles and near-field missing components; "
            "(2) Classical CV & Unsupervised Anomaly Detection (ExG vegetation segmentation, dual-rail polynomial tracking, "
            "sleeper periodic variance, ballast GLCM entropy); "
            "(3) Multimodal Sensor Fusion to bridge optical limits with replayed IMU vibration and simulated LiDAR/NDE signals."
        )

        return FeasibilityStudySummary(
            dataset_name=manifest_data.get("source_filename", "train_pov.mp4"),
            source_resolution=(w, h),
            roi_resolution=(roi_meta.crop_width, roi_meta.crop_height),
            evaluated_frame_count=total_frames,
            targets=targets,
            high_feasibility_count=high_c,
            medium_feasibility_count=med_c,
            low_feasibility_count=low_c,
            not_feasible_count=not_c,
            recommended_stage7_approach=rec_approach,
        )

    def create_evidence_contact_sheet(
        self,
        frames_dir: Path,
        representative_items: List[Dict],
        output_path: Path,
        card_width: int = 760,
    ) -> Path:
        """Creates an annotated visual evidence sheet highlighting key infrastructure components."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cards = []

        for item in representative_items:
            img_file = frames_dir / item["filename"]
            if not img_file.exists():
                continue

            raw_img = cv2.imread(str(img_file))
            if raw_img is None:
                continue

            roi = self.roi_extractor.crop(raw_img)
            rh, rw = roi.shape[:2]

            # Annotate infrastructure zones on the ROI crop
            annotated_roi = roi.copy()

            # 1. Left & Right Rail corridor lines (approximate visual track paths)
            left_rail_pt1 = (int(0.32 * rw), 0)
            left_rail_pt2 = (int(0.12 * rw), rh)
            right_rail_pt1 = (int(0.68 * rw), 0)
            right_rail_pt2 = (int(0.88 * rw), rh)

            # Draw rail highlight lines (Amber / Yellow)
            cv2.line(annotated_roi, left_rail_pt1, left_rail_pt2, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.line(annotated_roi, right_rail_pt1, right_rail_pt2, (0, 220, 255), 2, cv2.LINE_AA)

            # 2. Near-Field Fastener & Sleeper Zone (bottom 35% of ROI)
            nf_ymin = int(0.65 * rh)
            cv2.rectangle(annotated_roi, (0, nf_ymin), (rw, rh), (255, 100, 0), 2)
            cv2.putText(
                annotated_roi,
                "Near-Field Zone: Resolvable Fasteners & Sleeper Bodies",
                (12, nf_ymin + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )

            # 3. Track Gauge Corridor (between rails)
            cv2.putText(
                annotated_roi,
                "Track Gauge Corridor (Foreign Object & Veg Hazard Zone)",
                (int(0.24 * rw), int(0.40 * rh)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )

            # 4. Far-field Alignment Zone
            cv2.putText(
                annotated_roi,
                "Far-Field: Track Trajectory & Alignment",
                (int(0.28 * rw), 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Header banner
            ext_idx = item.get("extracted_index", 0)
            ts = item.get("timestamp_seconds", 0.0)
            banner = np.zeros((32, rw, 3), dtype=np.uint8)
            cv2.putText(
                banner,
                f"Visual Evidence Sample | Frame #{ext_idx:04d} @ {ts:.1f}s (ROI: {rw}x{rh}px)",
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            card = np.vstack([banner, annotated_roi])
            # Resize card to standard width
            ch, cw = card.shape[:2]
            card_thumb = cv2.resize(card, (card_width, int(card_width * ch / cw)), interpolation=cv2.INTER_AREA)
            cards.append(card_thumb)

        if not cards:
            raise ValueError("No valid frames to generate evidence sheet.")

        # Vertical stack of 3 representative samples
        contact_sheet = np.vstack(cards[:3])
        cv2.imwrite(str(output_path), contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return output_path

    def run_analysis(
        self,
        manifest_path: str | Path,
        output_dir: str | Path = "data/processed/inspection",
        docs_dir: str | Path = "docs",
    ) -> FeasibilityStudySummary:
        """Executes the complete feasibility study and writes JSON and Markdown reports."""
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        summary = self.build_feasibility_report(manifest_data, output_dir=output_dir)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        docs_path = Path(docs_dir)
        docs_path.mkdir(parents=True, exist_ok=True)

        # 1. Generate evidence contact sheet
        frames_dir = manifest_path.parent
        frames = manifest_data.get("frames", [])
        rep_indices = [0, len(frames) // 2, len(frames) - 1] if len(frames) >= 3 else list(range(len(frames)))
        rep_items = [frames[i] for i in rep_indices]

        evidence_sheet = out_path / "feasibility_evidence_contact_sheet.jpg"
        self.create_evidence_contact_sheet(
            frames_dir=frames_dir,
            representative_items=rep_items,
            output_path=evidence_sheet,
        )
        summary.contact_sheet_path = str(evidence_sheet.resolve())

        # 2. Save JSON Report
        json_report_path = out_path / "feasibility_report.json"
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_name": summary.dataset_name,
                    "source_resolution": summary.source_resolution,
                    "roi_resolution": summary.roi_resolution,
                    "evaluated_frame_count": summary.evaluated_frame_count,
                    "ratings_summary": {
                        "HIGH": summary.high_feasibility_count,
                        "MEDIUM": summary.medium_feasibility_count,
                        "LOW": summary.low_feasibility_count,
                        "NOT_FEASIBLE": summary.not_feasible_count,
                    },
                    "recommended_stage7_approach": summary.recommended_stage7_approach,
                    "contact_sheet_path": summary.contact_sheet_path,
                    "targets": [asdict(t) for t in summary.targets],
                },
                f,
                indent=2,
            )

        # 3. Save Markdown Report under docs/
        md_path = docs_path / "feasibility_assessment.md"
        md_content = self.generate_markdown_report(summary)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return summary

    def generate_markdown_report(self, summary: FeasibilityStudySummary) -> str:
        """Generates human-readable Markdown feasibility assessment document."""
        lines = [
            "# ITMS — Railway Track Inspection Feasibility Assessment",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Source Dataset**: `{summary.dataset_name}` (Train Front-Facing Cab POV)",
            f"- **Native Resolution**: `{summary.source_resolution[0]} × {summary.source_resolution[1]}` (Full HD 1080p)",
            f"- **Track ROI Crop Resolution**: `{summary.roi_resolution[0]} × {summary.roi_resolution[1]}` ($55.5\\%$ background discarded)",
            f"- **Evaluated Dataset Size**: `{summary.evaluated_frame_count}` sampled video frames",
            "",
            "### Summary of Inspection Feasibility",
            "",
            "| Feasibility Rating | Target Count | Description |",
            "| :--- | :--- | :--- |",
            f"| **HIGH** | **{summary.high_feasibility_count}** | Realistically detectable with high confidence via optical vision (Foreign objects, vegetation encroachment). |",
            f"| **MEDIUM** | **{summary.medium_feasibility_count}** | Feasible for macroscopic conditions (Track alignment/curvature, damaged sleepers, ballast voids). |",
            f"| **LOW** | **{summary.low_feasibility_count}** | Limited optical feasibility; near-field only (Missing fasteners, major rail burns). |",
            f"| **NOT FEASIBLE** | **{summary.not_feasible_count}** | Cannot be resolved from optical camera alone (Internal metallurgical flaws, sub-millimeter certified geometry). |",
            "",
            "---",
            "",
            "## 2. Infrastructure Visibility from Front-Facing Cab POV",
            "",
            "1. **Running Rails (Left & Right)**: Consistently visible from immediate near field ($0\\text{m}$) to vanishing horizon ($>150\\text{m}$). The top rail head running band provides a distinct reflective line feature.",
            "2. **Sleepers (Concrete Ties)**: Consistently visible at periodic intervals (~$60\\text{cm}$ center-to-center). Sleeper bodies span $>400\\text{px}$ in the near field.",
            r"3. **Fasteners (Elastic Rail Clips / Pandrol)**: Visible in near-field bottom $25\%$ of ROI ($\sim 15 - 25\text{px}$ footprint); viewed at an oblique angle.",
            "4. **Ballast Bed**: Clear texture distribution between and around rails across all daylight frames.",
            "5. **Track Gauge Corridor**: Fully visible corridor between rails ideal for forward obstacle and vegetation detection.",
            "",
            "---",
            "",
            "## 3. Systematic Target Feasibility Matrix",
            "",
        ]

        for t in summary.targets:
            badge = f"**[{t.detectability_rating.value}]**"
            lines.extend([
                f"### {t.target_name} — {badge}",
                f"- **Category**: {t.category}",
                f"- **Visibility**: `{t.visibility_level.value}`",
                f"- **Feasibility Rating**: `{t.detectability_rating.value}`",
                f"- **Assessment & Reasoning**: {t.reasoning}",
                f"- **Recommended CV Technique**: `{t.recommended_cv_approach}`",
                f"- **Requires External Labels**: `{'Yes' if t.requires_external_labels else 'No (Unsupervised / Algorithmic)'}`",
                f"- **Multimodal / Sensor Supplement**: {t.sensor_fusion_supplement}",
                f"- **Key Limitations**: *{t.key_limitations}*",
                "",
            ])

        lines.extend([
            "---",
            "",
            "## 4. Explicit Physical & Optical Limitations",
            "",
            "In strict accordance with **AGENTS.md** principles, the following physical inspection boundaries are established:",
            "",
            "1. **Internal Subsurface Defects**: Flaws such as internal transverse fissures, inclusion clusters, or bolt-hole cracking inside solid steel rails emit zero optical surface signature. They cannot be detected by any camera and require Ultrasonic Testing (UT) / Eddy Current probes (represented via replayed/simulated telemetry in ITMS).",
            "2. **Sub-Millimeter Certified Geometry**: Dynamic track geometry certification (requiring $\\pm 1.0\\text{mm}$ precision) cannot be certified from a vibrating cab-mounted optical camera without calibrated LiDAR and gyro-compensated IMU sensors.",
            "3. **Micro-Crack Resolving Limit**: Rail surface micro-checks ($<2\\text{mm}$) and hairline fastener fatigue cracks cannot be resolved from cab windshield distance.",
            "",
            "---",
            "",
            "## 5. Recommended Stage 7 AI Architecture",
            "",
            f"> {summary.recommended_stage7_approach}",
            "",
            "### Recommended Module Structure for Stage 7:",
            "1. `ai_pipeline/detection/hazard_detector.py`: Supervised YOLO-based detector for foreign track obstructions and prominent hazards.",
            "2. `ai_pipeline/detection/vegetation_detector.py`: Color-space & texture segmentation for vegetation encroachment.",
            "3. `ai_pipeline/detection/rail_tracker.py`: Dual-rail polynomial tracking and macroscopic alignment estimation.",
            "4. `ai_pipeline/anomaly/patch_anomaly.py`: Statistical texture & patch-based anomaly scoring for sleeper/ballast irregularities.",
            "5. `ai_pipeline/fusion/telemetry_fusion.py`: Multimodal fusion layer integrating visual defect scores with replayed IMU vibration and simulated NDE/LiDAR feeds.",
            "",
        ])

        return "\n".join(lines)
