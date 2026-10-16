"""
Visual Effects — Vignette, Chromatic Aberration, CRT Scanlines.
Pre-computes masks for maximum performance.
"""

import numpy as np
import cv2


class VisualEffects:
    """Post-processing effects for the tactical HUD look."""

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self._build_vignette_mask()
        self._build_scanline_mask()

    def _build_vignette_mask(self):
        """Pre-compute radial vignette gradient mask."""
        cols = np.arange(self.w, dtype=np.float32) - self.w / 2.0
        rows = np.arange(self.h, dtype=np.float32) - self.h / 2.0
        X, Y = np.meshgrid(cols, rows)

        sigma = max(self.w, self.h) * 0.55
        mask = np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))
        # Map from [0, 1] → [0.25, 1.0] so edges are darker
        mask = 0.25 + 0.75 * mask
        self._vignette = np.dstack([mask, mask, mask])

    def _build_scanline_mask(self):
        """Pre-compute CRT-style horizontal scanline mask."""
        mask = np.ones((self.h, self.w, 3), dtype=np.float32)
        mask[::3, :, :] = 0.88  # Slight darkening every 3rd row
        self._scanlines = mask

    def apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Apply dark-edge vignette effect."""
        result = (frame.astype(np.float32) * self._vignette).clip(0, 255)
        return result.astype(np.uint8)

    def apply_chromatic_aberration(self, frame: np.ndarray,
                                   shift: int = 2) -> np.ndarray:
        """Shift R and B channels to simulate chromatic aberration."""
        result = frame.copy()
        h, w = frame.shape[:2]

        if shift >= w:
            return result

        # Shift blue channel left
        result[:, :w - shift, 0] = frame[:, shift:, 0]
        # Shift red channel right (index 2 in BGR)
        result[:, shift:, 2] = frame[:, :w - shift, 2]

        return result

    def apply_scanlines(self, frame: np.ndarray) -> np.ndarray:
        """Apply CRT-style horizontal scanlines."""
        result = (frame.astype(np.float32) * self._scanlines).clip(0, 255)
        return result.astype(np.uint8)

    def apply_all(self, frame: np.ndarray) -> np.ndarray:
        """Apply all post-processing effects in order."""
        frame = self.apply_chromatic_aberration(frame, shift=2)
        frame = self.apply_scanlines(frame)
        frame = self.apply_vignette(frame)
        return frame

    def resize(self, width: int, height: int):
        """Rebuild masks if resolution changes."""
        if width != self.w or height != self.h:
            self.w = width
            self.h = height
            self._build_vignette_mask()
            self._build_scanline_mask()
