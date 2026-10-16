"""
Scanning Lines — Animated horizontal & vertical sweep effects.
"""

import numpy as np
import cv2
import config as cfg


class ScanningLines:
    """Draws animated scanning sweep lines across the frame."""

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.frame_counter = 0

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw animated scanning lines on the frame."""
        self.frame_counter += 1
        h, w = frame.shape[:2]

        # ── Primary horizontal scan (fast sweep) ──
        scan_y = int((self.frame_counter * cfg.SCAN_SPEED) % h)
        self._draw_scan_band(frame, scan_y, horizontal=True)

        # ── Secondary vertical scan (slower) ──
        scan_x = int((self.frame_counter * cfg.SCAN_SPEED * 0.4) % w)
        self._draw_vertical_band(frame, scan_x)

        # ── Subtle grid overlay ──
        self._draw_grid(frame)

        return frame

    def _draw_scan_band(self, frame: np.ndarray, scan_y: int,
                        horizontal: bool = True):
        """Draw a gradient scanning band at the given y position."""
        h, w = frame.shape[:2]
        band_h = cfg.SCAN_BAND_HEIGHT

        y_start = max(0, scan_y - band_h)
        y_end = min(h, scan_y + 2)

        if y_end <= y_start:
            return

        n_rows = y_end - y_start
        # Alpha gradient: fades from 0 at top of band to peak at scan_y
        alphas = np.linspace(0.0, 0.25, n_rows, dtype=np.float32)
        alphas = alphas.reshape(-1, 1, 1)

        color = np.array(cfg.NEON_CYAN, dtype=np.float32).reshape(1, 1, 3)
        band = frame[y_start:y_end].astype(np.float32)
        frame[y_start:y_end] = (
            band * (1.0 - alphas) + color * alphas
        ).clip(0, 255).astype(np.uint8)

        # Bright leading edge
        edge_y = min(scan_y, h - 1)
        cv2.line(frame, (0, edge_y), (w, edge_y), cfg.NEON_CYAN, 1)

    def _draw_vertical_band(self, frame: np.ndarray, scan_x: int):
        """Draw a subtle vertical scanning band."""
        h, w = frame.shape[:2]
        band_w = 30
        x_start = max(0, scan_x - band_w)
        x_end = min(w, scan_x + 2)

        if x_end <= x_start:
            return

        n_cols = x_end - x_start
        alphas = np.linspace(0.0, 0.1, n_cols, dtype=np.float32)
        alphas = alphas.reshape(1, -1, 1)

        color = np.array(cfg.NEON_BLUE, dtype=np.float32).reshape(1, 1, 3)
        band = frame[:, x_start:x_end].astype(np.float32)
        frame[:, x_start:x_end] = (
            band * (1.0 - alphas) + color * alphas
        ).clip(0, 255).astype(np.uint8)

    def _draw_grid(self, frame: np.ndarray):
        """Draw subtle tactical grid lines."""
        h, w = frame.shape[:2]
        spacing = 80

        for x in range(spacing, w, spacing):
            cv2.line(frame, (x, 0), (x, h), cfg.GRID_COLOR, 1)
        for y in range(spacing, h, spacing):
            cv2.line(frame, (0, y), (w, y), cfg.GRID_COLOR, 1)

    def resize(self, width: int, height: int):
        self.w = width
        self.h = height
