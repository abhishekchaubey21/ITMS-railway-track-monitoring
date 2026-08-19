"""Defect detection and video processing foundation."""

from ai_pipeline.detection.video_reader import VideoReader, VideoMetadata
from ai_pipeline.detection.frame_extractor import (
    FrameExtractor,
    ExtractionSummary,
    FrameInfo,
)
from ai_pipeline.detection.frame_inspector import (
    FrameInspector,
    InspectionReport,
    FrameQualityMetrics,
)
from ai_pipeline.detection.track_roi import (
    TrackROIExtractor,
    TrackROIConfig,
    ROICropMetadata,
)
from ai_pipeline.detection.feasibility_analyzer import (
    FeasibilityAnalyzer,
    FeasibilityStudySummary,
    InspectionTargetAssessment,
    DetectabilityRating,
    VisibilityLevel,
)
from ai_pipeline.detection.obstacle_detector import (
    ObstacleDetector,
    ObstacleDetectorConfig,
)
from ai_pipeline.detection.vision_pipeline import VisionPipeline

__all__ = [
    "VideoReader",
    "VideoMetadata",
    "FrameExtractor",
    "ExtractionSummary",
    "FrameInfo",
    "FrameInspector",
    "InspectionReport",
    "FrameQualityMetrics",
    "TrackROIExtractor",
    "TrackROIConfig",
    "ROICropMetadata",
    "FeasibilityAnalyzer",
    "FeasibilityStudySummary",
    "InspectionTargetAssessment",
    "DetectabilityRating",
    "VisibilityLevel",
    "ObstacleDetector",
    "ObstacleDetectorConfig",
    "VisionPipeline",
]
