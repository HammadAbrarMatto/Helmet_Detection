"""
config.py — Central configuration for the Helmet Detection System.
Modify these settings to customize paths, thresholds, and behaviour.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH    = os.path.join(BASE_DIR, "Weights", "best.pt")
MEDIA_DIR       = os.path.join(BASE_DIR, "Media")
VIOLATIONS_DIR  = os.path.join(BASE_DIR, "violations")
VIOLATIONS_CSV  = os.path.join(BASE_DIR, "violations.csv")

# ─── Model ────────────────────────────────────────────────────────────────────
CLASS_NAMES         = ["With Helmet", "Without Helmet"]
CONFIDENCE_THRESHOLD = 0.40   # Detections below this are ignored

# ─── Video / Webcam ───────────────────────────────────────────────────────────
FRAME_SKIP          = 2        # Process every Nth frame (1 = every frame)
DISPLAY_WIDTH       = 960
DISPLAY_HEIGHT      = 540
INFERENCE_WIDTH     = 640
INFERENCE_HEIGHT    = 480

# ─── Violation logging ────────────────────────────────────────────────────────
VIOLATION_COOLDOWN  = 3        # Minimum seconds between saved violation frames
CSV_HEADERS         = ["Timestamp", "Detection", "Confidence", "Source", "Image"]
