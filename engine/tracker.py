"""
Kalman-Filter Multi-Object Tracker with EMA smoothing.
Uses per-object cv2.KalmanFilter + Hungarian algorithm + exponential moving average.
"""

from typing import Dict, List, Tuple
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

import config as cfg
from engine.utils import Detection, TrackedTarget, compute_iou, estimate_distance


class KalmanBoxTracker:
    """Kalman filter + EMA for a single bounding box."""

    _id_counter = 0

    def __init__(self, detection: Detection):
        KalmanBoxTracker._id_counter += 1
        self.track_id = KalmanBoxTracker._id_counter

        # 8 state dims: [cx, cy, w, h, vx, vy, vw, vh]
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0

        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * cfg.KALMAN_PROCESS_NOISE
        for i in range(4, 8):
            self.kf.processNoiseCov[i, i] *= 10.0
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * cfg.KALMAN_MEASUREMENT_NOISE

        cx, cy, w, h = self._bbox_to_cxywh(detection.bbox)
        self.kf.statePost = np.array(
            [cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32
        ).reshape(8, 1)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.frames_tracked = 1
        self.frames_since_seen = 0
        self.raw_bbox = detection.bbox

        # EMA-smoothed bounding box
        self._ema_bbox: Tuple[int, int, int, int] = detection.bbox

    @staticmethod
    def _bbox_to_cxywh(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, float(x2 - x1), float(y2 - y1)

    @staticmethod
    def _cxywh_to_bbox(cx, cy, w, h):
        w, h = max(w, 1), max(h, 1)
        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

    def predict(self) -> Tuple[int, int, int, int]:
        state = self.kf.predict()
        return self._cxywh_to_bbox(state[0, 0], state[1, 0], abs(state[2, 0]), abs(state[3, 0]))

    def update(self, detection: Detection):
        cx, cy, w, h = self._bbox_to_cxywh(detection.bbox)
        self.kf.correct(np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1))
        self.raw_bbox = detection.bbox
        self.confidence = detection.confidence
        self.class_name = detection.class_name
        self.frames_tracked += 1
        self.frames_since_seen = 0

    def get_smoothed_bbox(self) -> Tuple[int, int, int, int]:
        """Get Kalman state + EMA smoothed bbox (eliminates jitter)."""
        state = self.kf.statePost
        kalman_bbox = self._cxywh_to_bbox(
            state[0, 0], state[1, 0], abs(state[2, 0]), abs(state[3, 0])
        )
        # Apply exponential moving average
        a = cfg.EMA_ALPHA
        self._ema_bbox = tuple(
            int(a * k + (1.0 - a) * e)
            for k, e in zip(kalman_bbox, self._ema_bbox)
        )
        return self._ema_bbox


class MultiObjectTracker:
    """Manages multiple Kalman+EMA trackers with Hungarian assignment."""

    def __init__(self):
        self.trackers: Dict[int, KalmanBoxTracker] = {}

    def update(self, detections: List[Detection],
               frame_height: int = 720) -> List[TrackedTarget]:
        """Full update: match detections → Kalman update → build targets."""
        # 1. Predict all
        predicted: Dict[int, Tuple] = {}
        for tid, trk in self.trackers.items():
            predicted[tid] = trk.predict()

        tracker_ids = list(self.trackers.keys())
        matched_t, matched_d = set(), set()

        # 2. Hungarian matching
        if tracker_ids and detections:
            cost = np.zeros((len(tracker_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(tracker_ids):
                for j, det in enumerate(detections):
                    cost[i, j] = 1.0 - compute_iou(predicted[tid], det.bbox)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if 1.0 - cost[r, c] >= cfg.IOU_MATCH_THRESHOLD:
                    self.trackers[tracker_ids[r]].update(detections[c])
                    matched_t.add(tracker_ids[r])
                    matched_d.add(c)

        # 3. Age unmatched trackers
        for tid in tracker_ids:
            if tid not in matched_t:
                self.trackers[tid].frames_since_seen += 1

        # 4. Remove dead
        for tid in [t for t, trk in self.trackers.items()
                    if trk.frames_since_seen > cfg.MAX_TRACKER_AGE]:
            del self.trackers[tid]

        # 5. Create new
        for j, det in enumerate(detections):
            if j not in matched_d:
                new = KalmanBoxTracker(det)
                self.trackers[new.track_id] = new

        return self._build_targets(frame_height)

    def predict_only(self, frame_height: int = 720) -> List[TrackedTarget]:
        """Predict-only step for skipped frames (no YOLO detections)."""
        for trk in self.trackers.values():
            trk.predict()

        # Remove dead trackers
        for tid in [t for t, trk in self.trackers.items()
                    if trk.frames_since_seen > cfg.MAX_TRACKER_AGE]:
            del self.trackers[tid]

        return self._build_targets(frame_height)

    def _build_targets(self, frame_height: int) -> List[TrackedTarget]:
        """Build TrackedTarget list from current trackers."""
        targets = []
        for trk in self.trackers.values():
            if trk.frames_since_seen > 2:
                continue
            smoothed = trk.get_smoothed_bbox()
            bbox_h = smoothed[3] - smoothed[1]
            targets.append(TrackedTarget(
                track_id=trk.track_id,
                bbox=trk.raw_bbox,
                smoothed_bbox=smoothed,
                class_name=trk.class_name,
                confidence=trk.confidence,
                frames_tracked=trk.frames_tracked,
                frames_since_seen=trk.frames_since_seen,
                locked=trk.frames_tracked >= cfg.LOCK_FRAMES_THRESHOLD,
                distance_m=estimate_distance(bbox_h, trk.class_name, frame_height),
            ))
        return targets

    def reset(self):
        self.trackers.clear()
        KalmanBoxTracker._id_counter = 0
