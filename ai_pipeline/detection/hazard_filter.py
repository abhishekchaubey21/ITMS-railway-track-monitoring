"""Ego-track hazard filtering for railway vision detections.

Filters object detections according to whether their bounding-box center
falls inside the estimated perspective corridor of the train's own track.

This is a geometric filter only. It does not perform object detection,
tracking, or railway-certified hazard assessment.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple


BBox = Sequence[float]


@dataclass(frozen=True)
class EgoTrackCorridorConfig:
    """Perspective corridor configuration expressed as image ratios."""

    vanishing_x_ratio: float = 0.50
    top_y_ratio: float = 0.45

    top_half_width_ratio: float = 0.12
    bottom_y_ratio: float = 0.98
    bottom_half_width_ratio: float = 0.42

    min_overlap_ratio: float = 0.0


class TrackHazardFilter:
    """Determines whether detections lie within the ego-track corridor."""

    def __init__(
        self,
        config: EgoTrackCorridorConfig | None = None,
    ) -> None:
        self.config = config or EgoTrackCorridorConfig()

    def _corridor_width_at_y(
        self,
        y: float,
        image_width: int,
        image_height: int,
    ) -> float:
        """Return half-width of the corridor at a given image y-coordinate."""

        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")

        top_y = self.config.top_y_ratio * image_height
        bottom_y = self.config.bottom_y_ratio * image_height

        if bottom_y <= top_y:
            raise ValueError("bottom_y_ratio must be greater than top_y_ratio")

        t = (y - top_y) / (bottom_y - top_y)
        t = max(0.0, min(1.0, t))

        top_half_width = self.config.top_half_width_ratio * image_width
        bottom_half_width = self.config.bottom_half_width_ratio * image_width

        return top_half_width + t * (bottom_half_width - top_half_width)

    def is_point_in_ego_track(
        self,
        x: float,
        y: float,
        image_width: int,
        image_height: int,
    ) -> bool:
        """Return True when a point lies inside the ego-track corridor."""

        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")

        top_y = self.config.top_y_ratio * image_height
        bottom_y = self.config.bottom_y_ratio * image_height

        if y < top_y or y > bottom_y:
            return False

        center_x = self.config.vanishing_x_ratio * image_width
        half_width = self._corridor_width_at_y(
            y,
            image_width,
            image_height,
        )

        return abs(x - center_x) <= half_width

    def is_in_ego_track(
        self,
        bbox: BBox,
        image_width: int,
        image_height: int,
    ) -> bool:
        """Return True when the detection center lies inside the ego track."""

        if len(bbox) != 4:
            raise ValueError("bbox must contain exactly four coordinates")

        x1, y1, x2, y2 = map(float, bbox)

        if x2 <= x1 or y2 <= y1:
            return False

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        return self.is_point_in_ego_track(
            center_x,
            center_y,
            image_width,
            image_height,
        )

    def filter_detections(
        self,
        detections: Sequence,
        image_width: int,
        image_height: int,
    ) -> List:
        """Return only detections whose bounding-box centers are on-track."""

        filtered = []

        for detection in detections:
            bbox = getattr(detection, "bbox", None)

            if bbox is None:
                continue

            if self.is_in_ego_track(
                bbox,
                image_width,
                image_height,
            ):
                filtered.append(detection)

        return filtered