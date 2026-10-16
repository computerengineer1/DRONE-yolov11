"""
Targeting System — Panel collision avoidance, focus-target priority, shadow text.
"""

import math
import time
import numpy as np
import cv2
from typing import List, Optional, Tuple

import config as cfg
from engine.utils import TrackedTarget

# ── Shadow text utility ─────────────────────────────────────────

def draw_shadow_text(frame, text, pos, font, scale, color,
                     thickness=1, shadow_offset=2):
    """Draw text with a dark drop-shadow for legibility."""
    x, y = pos
    cv2.putText(frame, text, (x + shadow_offset, y + shadow_offset),
                font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color,
                thickness, cv2.LINE_AA)


class TargetingSystem:
    """Corner-bracket targeting with collision-free panels & focus priority."""

    PANEL_W = 160
    PANEL_H_FULL = 82   # Full telemetry panel
    PANEL_H_MINI = 0    # Minimal — no panel, just brackets
    MARGIN = 6

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height

    # ── Public API ──────────────────────────────────────────────

    def draw(self, frame: np.ndarray, targets: List[TrackedTarget],
             focus_id: Optional[int] = None) -> np.ndarray:
        """Draw all targets. focus_id gets full telemetry, others get minimal."""
        if not targets:
            return frame

        # Compute non-overlapping panel positions for focus targets
        panels = self._resolve_panels(targets, focus_id, frame.shape)

        for target in targets:
            is_focus = (target.track_id == focus_id)
            self._draw_brackets(frame, target)

            if is_focus:
                panel = panels.get(target.track_id)
                if panel:
                    self._draw_full_panel(frame, target, panel)
            else:
                self._draw_mini_label(frame, target)

        return frame

    # ── Bracket rendering ───────────────────────────────────────

    def _draw_brackets(self, frame, target: TrackedTarget):
        x1, y1, x2, y2 = target.smoothed_bbox
        w_box, h_box = x2 - x1, y2 - y1
        if w_box <= 0 or h_box <= 0:
            return

        color = target.color
        thickness = 2

        # Pulsing glow for locked targets
        if target.locked:
            pulse = 0.6 + 0.4 * math.sin(time.time() * 6.0)
            thickness = 2 + int(pulse * 2)
            pad = 5
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1 - pad, y1 - pad),
                          (x2 + pad, y2 + pad), color, 2)
            cv2.addWeighted(overlay, 0.12 * pulse, frame,
                            1.0 - 0.12 * pulse, 0, frame)

        bracket_len = max(12, int(min(w_box, h_box) * 0.22))

        # 4 corner brackets
        for (bx, by), (dx, dy) in [
            ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)), ((x2, y2), (-1, -1)),
        ]:
            cv2.line(frame, (bx, by), (bx + dx * bracket_len, by),
                     color, thickness)
            cv2.line(frame, (bx, by), (bx, by + dy * bracket_len),
                     color, thickness)

        # Locked crosshair
        if target.locked:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(frame, (cx - 6, cy), (cx + 6, cy), cfg.TACTICAL_RED, 1)
            cv2.line(frame, (cx, cy - 6), (cx, cy + 6), cfg.TACTICAL_RED, 1)

    # ── Mini label (non-focus targets) ──────────────────────────

    def _draw_mini_label(self, frame, target: TrackedTarget):
        """Tiny class label above the bracket — no panel."""
        x1, y1, x2, y2 = target.smoothed_bbox
        label = f"{target.class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        draw_shadow_text(frame, label, (x1, y1 - 6), font, 0.33,
                         target.color, 1, 1)

    # ── Full telemetry panel (focus target) ─────────────────────

    def _draw_full_panel(self, frame, target: TrackedTarget,
                         panel: dict):
        """Full info panel with leader line."""
        px, py = panel['x'], panel['y']
        pw, ph = self.PANEL_W, self.PANEL_H_FULL
        x1, y1, x2, y2 = target.smoothed_bbox
        color = target.color
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.38
        lh = 16

        # Leader line from bbox to panel
        src_x = x2 if px > x2 else x1
        src_y = min(y1 + 10, (y1 + y2) // 2)
        dst_x = px if px > src_x else px + pw
        dst_y = py + 10
        cv2.line(frame, (src_x, src_y), (dst_x, dst_y), color, 1)

        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph),
                      (5, 5, 5), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 1)

        # Text content
        tx, ty = px + 5, py + lh
        draw_shadow_text(frame, f"TGT-{target.track_id:03d}  {target.class_name}",
                         (tx, ty), font, fs, color, 1, 1)
        ty += lh
        draw_shadow_text(frame, f"CONF: {target.confidence * 100:.1f}%",
                         (tx, ty), font, fs, cfg.HUD_TEXT, 1, 1)
        ty += lh
        draw_shadow_text(frame, f"DIST: ~{target.distance_m:.0f}m",
                         (tx, ty), font, fs, cfg.HUD_TEXT, 1, 1)
        ty += lh
        st_color = cfg.TACTICAL_RED if target.locked else cfg.TACTICAL_YELLOW
        icon = "[X]" if target.locked else "[ ]"
        draw_shadow_text(frame, f"{icon} {target.status}",
                         (tx, ty), font, fs, st_color, 1, 1)

    # ── Panel collision avoidance ───────────────────────────────

    def _resolve_panels(self, targets: List[TrackedTarget],
                        focus_id: Optional[int],
                        frame_shape) -> dict:
        """Compute non-overlapping panel positions for focus targets."""
        h, w = frame_shape[:2]
        panels = {}

        # Only the focus target gets a panel
        focus_targets = [t for t in targets if t.track_id == focus_id]
        if not focus_targets:
            return panels

        for target in focus_targets:
            x1, y1, x2, y2 = target.smoothed_bbox
            pw, ph = self.PANEL_W, self.PANEL_H_FULL

            # Try right side
            px = x2 + 10
            py = y1

            # If right side overflows, try left
            if px + pw > w - 5:
                px = x1 - pw - 10

            # Clamp
            px = max(5, min(px, w - pw - 5))
            py = max(60, min(py, h - ph - 5))

            panels[target.track_id] = {'x': px, 'y': py}

        # Resolve overlaps between multiple panels (future-proof)
        panel_list = list(panels.values())
        panel_list.sort(key=lambda p: p['y'])
        for i in range(1, len(panel_list)):
            for j in range(i):
                if self._overlaps(panel_list[i], panel_list[j]):
                    panel_list[i]['y'] = (panel_list[j]['y']
                                         + self.PANEL_H_FULL + self.MARGIN)
                    panel_list[i]['y'] = min(panel_list[i]['y'],
                                            h - self.PANEL_H_FULL - 5)

        return panels

    def _overlaps(self, a: dict, b: dict) -> bool:
        pw, ph = self.PANEL_W, self.PANEL_H_FULL
        return not (a['x'] + pw < b['x'] or b['x'] + pw < a['x'] or
                    a['y'] + ph < b['y'] or b['y'] + ph < a['y'])

    def resize(self, width: int, height: int):
        self.w, self.h = width, height
