"""Defect detection and video processing foundation."""

from ai_pipeline.detection.video_reader import VideoReader, VideoMetadata
from ai_pipeline.detection.frame_extractor import (
    FrameExtractor,
    ExtractionSummary,
    FrameInfo,
)

__all__ = [
    "VideoReader",
    "VideoMetadata",
    "FrameExtractor",
    "ExtractionSummary",
    "FrameInfo",
]
