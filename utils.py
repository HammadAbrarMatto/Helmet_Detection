"""
utils.py — Shared utilities: violation logging, frame saving, drawing helpers.
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import cvzone

from config import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    CSV_HEADERS,
    VIOLATIONS_CSV,
    VIOLATIONS_DIR,
)


# ─── Initialisation ───────────────────────────────────────────────────────────

def ensure_directories() -> None:
    """Create output directories and CSV header on first run."""
    Path(VIOLATIONS_DIR).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(VIOLATIONS_CSV):
        with open(VIOLATIONS_CSV, mode="w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADERS)


# ─── Violation helpers ────────────────────────────────────────────────────────

def save_violation_frame(frame, source: str) -> str:
    """
    Save a violation frame to disk.

    Args:
        frame: The OpenCV image (numpy array).
        source: A short label, e.g. ``"webcam"`` or ``"video"``.

    Returns:
        The filename (not full path) of the saved image.
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"violation_{source}_{timestamp_str}.jpg"
    filepath = os.path.join(VIOLATIONS_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filename


def log_violation(detection_label: str, confidence: float, source: str, image_name: str) -> None:
    """Append one violation row to violations.csv."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(VIOLATIONS_CSV, mode="a", newline="") as f:
        csv.writer(f).writerow([timestamp, detection_label, f"{confidence:.2f}", source, image_name])


# ─── Drawing helpers ──────────────────────────────────────────────────────────

# Colours (BGR)
_COLOR_SAFE      = (0, 200, 0)    # green  – With Helmet
_COLOR_VIOLATION = (0, 0, 220)    # red    – Without Helmet
_COLOR_TEXT_BG   = (30, 30, 30)


def draw_detection(frame, x1: int, y1: int, x2: int, y2: int, cls: int, conf: float) -> None:
    """Draw a bounding box and label for a single detection."""
    label = CLASS_NAMES[cls]
    color = _COLOR_SAFE if cls == 0 else _COLOR_VIOLATION
    w, h = x2 - x1, y2 - y1

    cvzone.cornerRect(frame, (x1, y1, w, h), colorC=color, colorR=color)
    cvzone.putTextRect(
        frame,
        f"{label}  {conf:.2f}",
        (max(0, x1), max(35, y1 - 5)),
        scale=0.9,
        thickness=1,
        colorR=color,
        colorB=_COLOR_TEXT_BG,
    )


def process_detections(frame, results, source: str, violation_state: dict) -> int:
    """
    Iterate YOLO results, draw boxes, and save violations with cooldown.

    Args:
        frame:           The current OpenCV frame (mutated in-place).
        results:         YOLO result list.
        source:          Label for the violation log (e.g. ``"webcam"``).
        violation_state: Mutable dict with keys ``"count"`` and ``"last_saved"``.

    Returns:
        Number of new violations logged this call.
    """
    new_violations = 0
    current_time = time.time()

    from config import VIOLATION_COOLDOWN  # avoid circular at module level

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            draw_detection(frame, x1, y1, x2, y2, cls, conf)

            if cls == 1 and (current_time - violation_state["last_saved"]) > VIOLATION_COOLDOWN:
                img_name = save_violation_frame(frame, source)
                log_violation(CLASS_NAMES[cls], conf, source, img_name)
                violation_state["count"] += 1
                violation_state["last_saved"] = current_time
                new_violations += 1

    return new_violations
