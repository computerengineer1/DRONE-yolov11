"""
Compass — Horizontal heading strip with cardinal directions.
"""

import math
import time
import numpy as np
import cv2

import config as cfg


class Compass:
    """Digital compass strip drawn at the top of the HUD."""

    CARDINALS = {
        0: "N", 45: "NE", 90: "E", 135: "SE",
        180: "S", 225: "SW", 270: "W", 315: "NW",
    }

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self._start_time = time.time()

    def draw(self, frame: np.ndarray, heading: float = -1.0) -> np.ndarray:
        """Draw compass strip at top of frame.

        Args:
            frame: BGR frame to draw on.
            heading: Heading in degrees (0-360). If -1, simulate.
        """
        h, w = frame.shape[:2]

        if heading < 0:
            # Simulate slow heading drift
            elapsed = time.time() - self._start_time
            heading = (elapsed * 8.0 + 45.0 * math.sin(elapsed * 0.3)) % 360.0

        strip_y = 10
        strip_h = 40
        strip_cx = w // 2
        visible_range = cfg.COMPASS_VISIBLE_RANGE
        pixels_per_deg = (w * 0.5) / visible_range

        # ── Dark background strip ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, strip_y), (w, strip_y + strip_h),
                      (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Top and bottom border lines
        cv2.line(frame, (0, strip_y), (w, strip_y), cfg.NEON_CYAN, 1)
        cv2.line(frame, (0, strip_y + strip_h), (w, strip_y + strip_h),
                 cfg.NEON_CYAN, 1)

        # ── Draw degree ticks and labels ──
        half_range = visible_range // 2
        for deg_offset in range(-half_range, half_range + 1):
            deg = (heading + deg_offset) % 360
            x = int(strip_cx + deg_offset * pixels_per_deg)

            if x < 0 or x >= w:
                continue

            deg_int = int(round(deg)) % 360

            # Major tick every 10°, minor every 5°
            if deg_int % 10 == 0:
                tick_top = strip_y + 4
                tick_bot = strip_y + strip_h - 4
                cv2.line(frame, (x, tick_top), (x, tick_bot),
                         cfg.NEON_CYAN, 1)

                # Degree number
                label = str(deg_int)
                text_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0]
                cv2.putText(frame, label,
                            (x - text_size[0] // 2, strip_y + strip_h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            cfg.HUD_TEXT, 1, cv2.LINE_AA)
            elif deg_int % 5 == 0:
                tick_top = strip_y + 10
                tick_bot = strip_y + strip_h - 10
                cv2.line(frame, (x, tick_top), (x, tick_bot),
                         cfg.GRID_COLOR, 1)

            # Cardinal labels
            if deg_int in self.CARDINALS:
                label = self.CARDINALS[deg_int]
                text_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
                cv2.putText(frame, label,
                            (x - text_size[0] // 2, strip_y + 18),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5,
                            cfg.NEON_CYAN, 1, cv2.LINE_AA)

        # ── Center pointer triangle ──
        tri_top = strip_y + strip_h + 2
        tri_size = 6
        pts = np.array([
            [strip_cx, tri_top + tri_size],
            [strip_cx - tri_size, tri_top],
            [strip_cx + tri_size, tri_top],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], cfg.NEON_CYAN)

        # ── Heading readout ──
        hdg_text = f"HDG {heading:05.1f}"
        text_size = cv2.getTextSize(
            hdg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(frame, hdg_text,
                    (strip_cx - text_size[0] // 2, strip_y + strip_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    cfg.NEON_CYAN, 1, cv2.LINE_AA)

        return frame

    def resize(self, width: int, height: int):
        self.w = width
        self.h = height
