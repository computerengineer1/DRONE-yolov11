"""
Drone Reconnaissance & Targeting System — Global Configuration
"""

import torch

# ──────────────────────────── Device ────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"  # FP16 only on GPU

# ──────────────────────────── Model ─────────────────────────────
MODEL_PATH = "yolo11s.pt"  # Small model — much better accuracy than nano
CONFIDENCE_THRESHOLD = 0.20  # Low threshold for challenging drone footage
IOU_THRESHOLD = 0.40         # Allow overlapping detections
INPUT_SIZE = (1280, 720)
UPSCALE_TARGET = 1280        # Upscale low-res frames to this before inference
FRAME_SKIP = 2               # Detect every Nth frame; Kalman-predict on others
EMA_ALPHA = 0.35             # Bbox smoothing (0=frozen, 1=raw detection)

# ──────────────────────────── Tactical Priority ─────────────────
THREAT_PRIORITY = {
    "MERKAVA MBT": 5, "AIRCRAFT": 4, "PERSONNEL": 3,
    "NAVAL ASSET": 3, "LIGHT VEHICLE": 2, "EQUIPMENT": 1, "OBJECT": 0,
}

# ──────────────────────────── Class Mapping ─────────────────────
# Map ALL 80 COCO classes → military reconnaissance labels.
# This ensures YOLO detects EVERYTHING visible in the frame.

_PERSONNEL_IDS = {0}                          # person
_HEAVY_VEHICLE_IDS = {2, 5, 7}                # car, bus, truck → tank proxies
_LIGHT_VEHICLE_IDS = {1, 3, 6}                # bicycle, motorcycle, train
_AIRCRAFT_IDS = {4}                           # airplane
_NAVAL_IDS = {8}                              # boat
_EQUIPMENT_IDS = {24, 25, 26, 27, 28, 39, 67} # backpack, umbrella, handbag, tie, suitcase, bottle, cell phone

CLASS_MAP = {}
for _cid in range(80):
    if _cid in _PERSONNEL_IDS:
        CLASS_MAP[_cid] = "PERSONNEL"
    elif _cid in _HEAVY_VEHICLE_IDS:
        CLASS_MAP[_cid] = "MERKAVA MBT"
    elif _cid in _LIGHT_VEHICLE_IDS:
        CLASS_MAP[_cid] = "LIGHT VEHICLE"
    elif _cid in _AIRCRAFT_IDS:
        CLASS_MAP[_cid] = "AIRCRAFT"
    elif _cid in _NAVAL_IDS:
        CLASS_MAP[_cid] = "NAVAL ASSET"
    elif _cid in _EQUIPMENT_IDS:
        CLASS_MAP[_cid] = "EQUIPMENT"
    else:
        CLASS_MAP[_cid] = "OBJECT"

# None = detect ALL 80 COCO classes (no pre-filtering)
TARGET_CLASSES = None

# ──────────────────────────── Colors (BGR for OpenCV) ──────────
NEON_CYAN = (255, 255, 0)       # Neon blue/cyan
NEON_BLUE = (255, 180, 0)       # Bright blue
TACTICAL_RED = (0, 0, 255)      # Lock-on red
TACTICAL_GREEN = (0, 255, 0)    # Tracking green
TACTICAL_YELLOW = (0, 255, 255) # Acquiring yellow
HUD_TEXT = (200, 200, 200)      # Light grey for text
DARK_BG = (15, 15, 20)          # Near-black background
GRID_COLOR = (80, 60, 0)        # Subtle grid lines

# ──────────────────────────── Tracking ──────────────────────────
LOCK_FRAMES_THRESHOLD = 18   # Frames to acquire target lock
MAX_TRACKER_AGE = 30          # Max frames without detection before removal
IOU_MATCH_THRESHOLD = 0.20   # Lower threshold for better association

# ──────────────────────────── Kalman Filter ─────────────────────
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEASUREMENT_NOISE = 1e-1

# ──────────────────────────── Distance Estimation ───────────────
# Reference real-world heights (meters) per class
REFERENCE_HEIGHTS = {
    "PERSONNEL": 1.75,
    "MERKAVA MBT": 2.6,
    "LIGHT VEHICLE": 1.5,
    "AIRCRAFT": 4.0,
    "NAVAL ASSET": 3.0,
    "EQUIPMENT": 0.5,
    "OBJECT": 1.5,
    "INFRASTRUCTURE": 3.5,
}
VIRTUAL_FOCAL_LENGTH = 800  # Approximate focal length in pixels

# ──────────────────────────── HUD Animation ─────────────────────
SCAN_SPEED = 3          # Pixels per frame for scanning line
SCAN_BAND_HEIGHT = 50   # Height of the scanning gradient band
COMPASS_VISIBLE_RANGE = 120  # Degrees visible on compass strip

# ──────────────────────────── UI ────────────────────────────────
WINDOW_TITLE = "DRONE RECON — Tactical Reconnaissance System v1.0"
WINDOW_SIZE = "1440x860"
SIDEBAR_WIDTH = 260
