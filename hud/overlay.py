"""
HUD Overlay — Master compositor with focus-target selection & shadow text.
"""

import time
import math
import numpy as np
import cv2
from typing import List, Optional

import config as cfg
from engine.utils import TrackedTarget
from hud.scanner import ScanningLines
from hud.targeting import TargetingSystem, draw_shadow_text
from hud.compass import Compass
from hud.horizon import HorizonIndicator
from hud.effects import VisualEffects


class HUDOverlay:
    """Composes all HUD elements onto the video frame."""

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.scanner = ScanningLines(width, height)
        self.targeting = TargetingSystem(width, height)
        self.compass = Compass(width, height)
        self.horizon = HorizonIndicator(width, height)
        self.effects = VisualEffects(width, height)
        self._start_time = time.time()
        self._frame_count = 0

    def render(self, frame: np.ndarray,
               targets: List[TrackedTarget],
               inference_ms: float = 0.0,
               fps: float = 0.0,
               n_targets: int = 0) -> np.ndarray:
        h, w = frame.shape[:2]
        self._frame_count += 1

        if w != self.w or h != self.h:
            self.w, self.h = w, h
            self.scanner.resize(w, h)
            self.targeting.resize(w, h)
            self.compass.resize(w, h)
            self.horizon.resize(w, h)
            self.effects.resize(w, h)

        # Select focus target (closest to reticle or highest threat)
        focus_id = self._select_focus_target(targets, w, h)

        # 1. Scanning sweep lines
        frame = self.scanner.draw(frame)

        # 2. Target brackets & panels (with focus priority)
        frame = self.targeting.draw(frame, targets, focus_id=focus_id)

        # 3. Compass
        frame = self.compass.draw(frame)

        # 4. Horizon indicator
        frame = self.horizon.draw(frame)

        # 5. Center reticle
        self._draw_center_reticle(frame)

        # 6. Telemetry panel
        self._draw_telemetry(frame, inference_ms, fps, n_targets)

        # 7. System labels
        self._draw_system_labels(frame)

        # 8. Post-processing effects
        frame = self.effects.apply_all(frame)

        return frame

    def _select_focus_target(self, targets: List[TrackedTarget],
                             w: int, h: int) -> Optional[int]:
        """Pick the highest-priority target closest to reticle center."""
        if not targets:
            return None

        cx, cy = w // 2, h // 2
        best_id = None
        best_score = -1.0

        for t in targets:
            tx = (t.smoothed_bbox[0] + t.smoothed_bbox[2]) // 2
            ty = (t.smoothed_bbox[1] + t.smoothed_bbox[3]) // 2
            dist = math.hypot(tx - cx, ty - cy)
            max_dist = math.hypot(w, h) / 2.0

            # Proximity score (1.0 = center, 0.0 = corner)
            proximity = max(0.0, 1.0 - dist / max_dist)

            # Threat priority score (0-5 normalized to 0-1)
            threat = cfg.THREAT_PRIORITY.get(t.class_name, 0) / 5.0

            # Combined score: 60% proximity + 40% threat
            score = 0.6 * proximity + 0.4 * threat

            if score > best_score:
                best_score = score
                best_id = t.track_id

        return best_id

    def _draw_center_reticle(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        gap, arm = 12, 25
        c = cfg.NEON_CYAN

        cv2.line(frame, (cx - arm - gap, cy), (cx - gap, cy), c, 1)
        cv2.line(frame, (cx + gap, cy), (cx + arm + gap, cy), c, 1)
        cv2.line(frame, (cx, cy - arm - gap), (cx, cy - gap), c, 1)
        cv2.line(frame, (cx, cy + gap), (cx, cy + arm + gap), c, 1)
        cv2.circle(frame, (cx, cy), 2, c, -1)

        cs = 6
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            ox, oy = cx + dx * (gap + 2), cy + dy * (gap + 2)
            cv2.line(frame, (ox, oy), (ox + dx * cs, oy), c, 1)
            cv2.line(frame, (ox, oy), (ox, oy + dy * cs), c, 1)

    def _draw_telemetry(self, frame, inference_ms, fps, n_targets):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, lh = 0.4, 18

        px, py, pw, ph = 10, h - 105, 200, 95
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), cfg.NEON_CYAN, 1)

        tx, ty = px + 8, py + lh
        draw_shadow_text(frame, f"FPS: {fps:.1f}", (tx, ty),
                         font, fs, cfg.NEON_CYAN, 1, 1)
        ty += lh
        draw_shadow_text(frame, f"INF: {inference_ms:.1f}ms", (tx, ty),
                         font, fs, cfg.NEON_CYAN, 1, 1)
        ty += lh
        tc = cfg.TACTICAL_RED if n_targets > 0 else cfg.NEON_CYAN
        draw_shadow_text(frame, f"TARGETS: {n_targets}", (tx, ty),
                         font, fs, tc, 1, 1)
        ty += lh
        draw_shadow_text(frame, f"RES: {w}x{h}", (tx, ty),
                         font, fs, cfg.HUD_TEXT, 1, 1)
        ty += lh
        draw_shadow_text(frame, f"DEV: {cfg.DEVICE.upper()}", (tx, ty),
                         font, fs, cfg.HUD_TEXT, 1, 1)

    def _draw_system_labels(self, frame):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        draw_shadow_text(frame, "DRONE RECON v1.0", (12, 78),
                         font, 0.45, cfg.NEON_CYAN, 1, 2)

        ts = time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime())
        tsz = cv2.getTextSize(ts, font, 0.38, 1)[0]
        draw_shadow_text(frame, ts, (w - tsz[0] - 12, h - 12),
                         font, 0.38, cfg.HUD_TEXT, 1, 1)

        if self._frame_count % 40 < 25:
            cv2.circle(frame, (w - 20, 68), 6, cfg.TACTICAL_RED, -1)
            draw_shadow_text(frame, "REC", (w - 55, 72),
                             font, 0.4, cfg.TACTICAL_RED, 1, 1)
