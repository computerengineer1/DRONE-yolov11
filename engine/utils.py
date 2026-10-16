"""
Engine utilities — Data structures and helper functions.
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

import config as cfg


@dataclass
class Detection:
    """A single detection from the YOLO model."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    confidence: float
    class_name: str = ""

    def __post_init__(self):
        if not self.class_name:
            self.class_name = cfg.CLASS_MAP.get(self.class_id, "UNKNOWN")

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class TrackedTarget:
    """A tracked object with Kalman-filtered state."""
    track_id: int
    bbox: Tuple[int, int, int, int]         # Raw detection bbox
    smoothed_bbox: Tuple[int, int, int, int] # Kalman-smoothed bbox
    class_name: str
    confidence: float
    frames_tracked: int = 0
    frames_since_seen: int = 0
    locked: bool = False
    distance_m: float = 0.0

    @property
    def status(self) -> str:
        if self.locked:
            return "LOCKED"
        elif self.frames_tracked >= cfg.LOCK_FRAMES_THRESHOLD // 2:
            return "ACQUIRING"
        return "TRACKING"

    @property
    def color(self):
        """BGR color based on lock status."""
        if self.locked:
            return cfg.TACTICAL_RED
        elif self.frames_tracked >= cfg.LOCK_FRAMES_THRESHOLD // 2:
            return cfg.TACTICAL_YELLOW
        return cfg.TACTICAL_GREEN


def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """Compute Intersection over Union between two (x1,y1,x2,y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def estimate_distance(bbox_height: int, class_name: str,
                      frame_height: int = 720) -> float:
    """Rough distance estimation based on bbox-to-frame height ratio."""
    if bbox_height <= 0:
        return 0.0
    ref_h = cfg.REFERENCE_HEIGHTS.get(class_name, 2.0)
    # Scale focal length by actual frame height vs reference 720p
    focal = cfg.VIRTUAL_FOCAL_LENGTH * (frame_height / 720.0)
    distance = (ref_h * focal) / bbox_height
    return round(distance, 1)
