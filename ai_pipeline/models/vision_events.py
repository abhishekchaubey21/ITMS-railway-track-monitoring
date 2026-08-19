"""Vision event and detection schemas for ITMS pipeline.

Provides structured, validated Pydantic data models for defect detections,
vegetation assessments, track geometry indicators, and composite frame vision results.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SeverityLevel(str, Enum):
    """Defect and alert severity classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class GeometryStatus(str, Enum):
    """Track geometry and visual alignment status."""
    NORMAL = "NORMAL"
    ADVISORY = "ADVISORY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class DetectionEvent(BaseModel):
    """Single visual defect or object detection event on a video frame."""
    frame_id: int = Field(..., ge=0, description="Sequential extracted frame index")
    timestamp: float = Field(..., ge=0.0, description="Timestamp in seconds from start of video")
    class_name: str = Field(..., min_length=1, description="Detected object/defect class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score (0.0 - 1.0)")
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    severity: SeverityLevel = Field(..., description="Assessed severity level")
    source: str = Field(..., min_length=1, description="Detection source (e.g. 'yolov8n', 'synthetic', 'manual')")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        """Validates that bbox contains exactly 4 coordinates with valid ordering (x2 >= x1, y2 >= y1)."""
        if len(v) != 4:
            raise ValueError(f"bbox must contain exactly 4 coordinates [x1, y1, x2, y2], got {len(v)}")
        x1, y1, x2, y2 = v
        if x2 < x1:
            raise ValueError(f"Invalid bbox: x2 ({x2}) must be >= x1 ({x1})")
        if y2 < y1:
            raise ValueError(f"Invalid bbox: y2 ({y2}) must be >= y1 ({y1})")
        return v


class VegetationAssessment(BaseModel):
    """Vegetation density and track corridor encroachment assessment."""
    score: float = Field(..., ge=0.0, le=1.0, description="Vegetation encroachment density score (0.0 - 1.0)")
    intrusion_detected: bool = Field(..., description="Whether vegetation crosses into active track gauge")


class GeometryAssessment(BaseModel):
    """Visual track alignment, curvature, and gauge consistency assessment."""
    alignment_score: float = Field(..., ge=0.0, le=1.0, description="Track visual alignment score (0.0 - 1.0)")
    status: GeometryStatus = Field(..., description="Track geometry condition status")


class VisionResult(BaseModel):
    """Composite computer-vision inference result for a single video frame."""
    frame_id: int = Field(..., ge=0, description="Sequential extracted frame index")
    source_frame: int = Field(..., ge=0, description="Original source video frame index")
    timestamp: float = Field(..., ge=0.0, description="Video timestamp in seconds")
    detections: List[DetectionEvent] = Field(default_factory=list, description="List of bounding box detections")
    vegetation: Optional[VegetationAssessment] = Field(default=None, description="Vegetation assessment if computed")
    geometry: Optional[GeometryAssessment] = Field(default=None, description="Track geometry assessment if computed")
