"""Ego-track hazard filtering and railway obstacle semantic policy for ITMS.

Distinguishes between actual track hazards (persons, animals, fallen trees, boulders,
vehicles, debris inside the train's own running corridor) and benign railway infrastructure
or adjacent track traffic (stationary/passing trains on parallel tracks, platform benches, signals).

NOTE: This is a geometric and semantic filter prototype. It does not perform physical track
clearance certification or precision stereo LiDAR survey.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

BBox = Sequence[float]


@dataclass(frozen=True)
class EgoTrackCorridorConfig:
    """Perspective corridor configuration representing the train's own track gauge as image ratios."""

    vanishing_x_ratio: float = 0.50
    top_y_ratio: float = 0.45
    top_half_width_ratio: float = 0.07  # ~134px half-width at y=486 on 1080p
    bottom_y_ratio: float = 0.98
    bottom_half_width_ratio: float = 0.22  # ~422px half-width at y=1058 on 1080p (x: 537 - 1382)
    min_overlap_ratio: float = 0.25


@dataclass(frozen=True)
class HazardPolicy:
    """Configurable semantic classification policy for railway hazards."""

    # Object classes considered track hazards when present inside the ego-track corridor
    hazard_classes: Tuple[str, ...] = (
        "person",
        "human",
        "pedestrian",
        "car",
        "truck",
        "motorcycle",
        "bicycle",
        "bus",
        "vehicle",
        "animal",
        "dog",
        "cat",
        "cow",
        "horse",
        "sheep",
        "elephant",
        "bear",
        "bird",
        "fallen_tree",
        "boulder",
        "rock_debris",
        "rock",
        "debris",
        "obstacle",
        "hazard",
        "suitcase",
        "backpack",
        "handbag",
    )

    # Whether to treat COCO 'train' detections as obstacles
    # False by default: adjacent/passing trains are normal railway traffic, not track obstacles
    treat_train_as_hazard: bool = False

    # Non-hazard railway scene objects to explicitly exclude
    benign_classes: Tuple[str, ...] = (
        "train",
        "traffic light",
        "stop sign",
        "bench",
        "chair",
        "potted plant",
        "clock",
        "building",
    )


class TrackHazardFilter:
    """Filters object detections according to ego-track corridor geometry and hazard semantics."""

    def __init__(
        self,
        config: Optional[EgoTrackCorridorConfig] = None,
        policy: Optional[HazardPolicy] = None,
    ) -> None:
        """Initialize TrackHazardFilter.

        Args:
            config: Geometric perspective corridor configuration.
            policy: Semantic hazard classification policy.
        """
        self.config = config or EgoTrackCorridorConfig()
        self.policy = policy or HazardPolicy()
        self._hazard_set: Set[str] = {c.lower() for c in self.policy.hazard_classes}
        self._benign_set: Set[str] = {c.lower() for c in self.policy.benign_classes}

    def _corridor_width_at_y(
        self,
        y: float,
        image_width: int,
        image_height: int,
    ) -> float:
        """Return half-width of the ego-track corridor at a given image y-coordinate."""
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

    def get_corridor_bounds_at_y(
        self,
        y: float,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float]:
        """Return (left_x, right_x) pixel boundaries of the ego corridor at coordinate y."""
        center_x = self.config.vanishing_x_ratio * image_width
        half_width = self._corridor_width_at_y(y, image_width, image_height)
        return (center_x - half_width, center_x + half_width)

    def is_point_in_ego_track(
        self,
        x: float,
        y: float,
        image_width: int,
        image_height: int,
    ) -> bool:
        """Return True when a point lies inside the ego-track perspective corridor."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")

        top_y = self.config.top_y_ratio * image_height
        bottom_y = self.config.bottom_y_ratio * image_height

        if y < top_y or y > bottom_y:
            return False

        left_x, right_x = self.get_corridor_bounds_at_y(y, image_width, image_height)
        return left_x <= x <= right_x

    def is_in_ego_track(
        self,
        bbox: BBox,
        image_width: int,
        image_height: int,
    ) -> bool:
        """Return True when the bounding box base or center lies inside the ego-track corridor."""
        if len(bbox) != 4:
            raise ValueError("bbox must contain exactly four coordinates")

        x1, y1, x2, y2 = map(float, bbox)

        if x2 <= x1 or y2 <= y1:
            return False

        top_y = self.config.top_y_ratio * image_height
        bottom_y = self.config.bottom_y_ratio * image_height

        # If object is entirely outside the active vertical track range
        if y2 < top_y or y1 > bottom_y:
            return False

        # Evaluate at ground/base contact level (y2 clamped to corridor bottom)
        base_y = max(top_y, min(y2, bottom_y))
        c_left, c_right = self.get_corridor_bounds_at_y(base_y, image_width, image_height)

        # 1. Base center check
        base_center_x = (x1 + x2) / 2.0
        if c_left <= base_center_x <= c_right:
            return True

        # 2. Geometric horizontal overlap ratio at base level
        overlap = max(0.0, min(x2, c_right) - max(x1, c_left))
        box_width = max(1e-3, x2 - x1)
        overlap_ratio = overlap / box_width

        if overlap_ratio >= self.config.min_overlap_ratio:
            return True

        # 3. Fallback: Bounding box geometric center point
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return self.is_point_in_ego_track(center_x, center_y, image_width, image_height)

    def is_hazard(
        self,
        detection: Any,
        image_width: int,
        image_height: int,
    ) -> bool:
        """Determines whether a detection represents a true on-track hazard.

        Args:
            detection: DetectionEvent or object with bbox and optional class_name.
            image_width: Width of the full video frame.
            image_height: Height of the full video frame.

        Returns:
            True if the object is semantically a hazard and located inside the ego-track corridor.
        """
        bbox = getattr(detection, "bbox", None)
        if bbox is None:
            return False

        class_name = getattr(detection, "class_name", None)

        # 1. Check semantic class policy
        if class_name is not None:
            cls_lower = str(class_name).lower().strip()

            # Train policy check: adjacent/parallel trains are regular traffic, not hazards
            if cls_lower == "train":
                if not self.policy.treat_train_as_hazard:
                    return False
                return self.is_in_ego_track(bbox, image_width, image_height)

            # Benign infrastructure check (benches, stop signs, traffic lights on platforms)
            if cls_lower in self._benign_set:
                return False

            # If class is explicitly known, must match hazard set
            if self._hazard_set and not any(h in cls_lower for h in self._hazard_set):
                # Class is unrecognized or non-hazard
                return False

        # 2. Check spatial ego-track corridor containment
        return self.is_in_ego_track(bbox, image_width, image_height)

    def filter_detections(
        self,
        detections: Sequence[Any],
        image_width: int,
        image_height: int,
    ) -> List[Any]:
        """Return only detections that constitute genuine hazards inside the ego-track corridor."""
        filtered = []

        for detection in detections:
            if self.is_hazard(detection, image_width, image_height):
                filtered.append(detection)

        return filtered