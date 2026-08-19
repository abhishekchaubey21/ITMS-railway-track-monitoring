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
]
