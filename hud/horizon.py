"""
Horizon Indicator — Artificial pitch/roll attitude display.
"""

import math
import time
import numpy as np
import cv2

import config as cfg


class HorizonIndicator:
    """Draws an artificial horizon / attitude indicator on the HUD."""

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self._start_time = time.time()

    def draw(self, frame: np.ndarray,
             pitch: float = -999.0, roll: float = -999.0) -> np.ndarray:
        """Draw attitude indicator on the right side of the frame.

        Args:
            frame: BGR frame.
            pitch: Pitch in degrees (positive=nose up). -999 = simulate.
            roll: Roll in degrees (positive=right wing down). -999 = simulate.
        """
        h, w = frame.shape[:2]

        # Simulate gentle movement if no real IMU data
        if pitch < -900:
            t = time.time() - self._start_time
            pitch = 5.0 * math.sin(t * 0.4) + 2.0 * math.sin(t * 1.1)
            roll = 8.0 * math.sin(t * 0.25) + 3.0 * math.cos(t * 0.7)

        # Indicator position & size
        cx = w - 90
        cy = h // 2
        radius = 70

        # ── Dark background circle ──
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), radius + 5, (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # ── Outer ring ──
        cv2.circle(frame, (cx, cy), radius, cfg.NEON_CYAN, 1)

        # ── Roll arc ticks ──
        for angle_deg in range(-60, 61, 15):
            rad = math.radians(angle_deg - 90)  # -90 so 0° is top
            inner_r = radius - 3
            outer_r = radius + (6 if angle_deg % 30 == 0 else 3)
            x1 = int(cx + inner_r * math.cos(rad))
            y1 = int(cy + inner_r * math.sin(rad))
            x2 = int(cx + outer_r * math.cos(rad))
            y2 = int(cy + outer_r * math.sin(rad))
            cv2.line(frame, (x1, y1), (x2, y2), cfg.NEON_CYAN, 1)

        # ── Roll pointer ──
        roll_rad = math.radians(-roll - 90)
        px = int(cx + (radius + 10) * math.cos(roll_rad))
        py = int(cy + (radius + 10) * math.sin(roll_rad))
        cv2.circle(frame, (px, py), 4, cfg.NEON_CYAN, -1)

        # ── Pitch ladder lines (clipped to circle) ──
        roll_rad_draw = math.radians(-roll)
        cos_r = math.cos(roll_rad_draw)
        sin_r = math.sin(roll_rad_draw)
        pixels_per_deg = 3.0  # Scale pitch to pixels

        for pitch_line in range(-30, 31, 10):
            if pitch_line == 0:
                continue

            # Offset from center based on pitch difference
            dy_offset = (pitch_line - pitch) * pixels_per_deg
            half_len = 25 if pitch_line % 20 == 0 else 15

            # Rotated line endpoints
            x1_raw = -half_len
            x2_raw = half_len
            y_raw = -dy_offset

            x1_rot = int(cx + x1_raw * cos_r - y_raw * sin_r)
            y1_rot = int(cy + x1_raw * sin_r + y_raw * cos_r)
            x2_rot = int(cx + x2_raw * cos_r - y_raw * sin_r)
            y2_rot = int(cy + x2_raw * sin_r + y_raw * cos_r)

            # Check if within the circle
            if (math.hypot(x1_rot - cx, y1_rot - cy) < radius - 5 and
                    math.hypot(x2_rot - cx, y2_rot - cy) < radius - 5):
                line_color = cfg.TACTICAL_GREEN if pitch_line > 0 else cfg.TACTICAL_RED
                cv2.line(frame, (x1_rot, y1_rot), (x2_rot, y2_rot),
                         line_color, 1, cv2.LINE_AA)
                # Pitch degree label
                cv2.putText(frame, str(abs(pitch_line)),
                            (x2_rot + 3, y2_rot + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            line_color, 1, cv2.LINE_AA)

        # ── Center horizon line ──
        dy_offset = -pitch * pixels_per_deg
        hor_half = 35
        hx1 = int(cx + (-hor_half) * cos_r - (-dy_offset) * sin_r)
        hy1 = int(cy + (-hor_half) * sin_r + (-dy_offset) * cos_r)
        hx2 = int(cx + hor_half * cos_r - (-dy_offset) * sin_r)
        hy2 = int(cy + hor_half * sin_r + (-dy_offset) * cos_r)
        cv2.line(frame, (hx1, hy1), (hx2, hy2), cfg.TACTICAL_GREEN, 2,
                 cv2.LINE_AA)

        # ── Aircraft symbol (fixed) ──
        cv2.line(frame, (cx - 20, cy), (cx - 8, cy), cfg.TACTICAL_YELLOW, 2)
        cv2.line(frame, (cx + 8, cy), (cx + 20, cy), cfg.TACTICAL_YELLOW, 2)
        cv2.circle(frame, (cx, cy), 3, cfg.TACTICAL_YELLOW, -1)

        # ── Labels ──
        cv2.putText(frame, f"PIT {pitch:+.1f}",
                    (cx - 35, cy + radius + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, cfg.NEON_CYAN, 1,
                    cv2.LINE_AA)
        cv2.putText(frame, f"ROL {roll:+.1f}",
                    (cx - 35, cy + radius + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, cfg.NEON_CYAN, 1,
                    cv2.LINE_AA)

        # ── "ATT" label at top ──
        cv2.putText(frame, "ATT",
                    (cx - 12, cy - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.NEON_CYAN, 1,
                    cv2.LINE_AA)

        return frame

    def resize(self, width: int, height: int):
        self.w = width
        self.h = height
