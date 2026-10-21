"""Frame annotation using plain OpenCV (no cvzone dependency)."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .types import Detection

COLOR_SAFE = (0, 200, 0)        # BGR green
COLOR_VIOLATION = (0, 0, 220)   # BGR red
COLOR_TEXT = (255, 255, 255)


def _corner_rect(img: np.ndarray, box: tuple[int, int, int, int], color, length: int = 18, t: int = 4) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    length = max(4, min(length, (x2 - x1) // 3, (y2 - y1) // 3))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, t)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, t)


def draw_detection(img: np.ndarray, det: Detection) -> None:
    color = COLOR_VIOLATION if det.is_violation else COLOR_SAFE
    _corner_rect(img, det.box, color)
    text = f"{det.label} {det.conf:.2f}" + (f" #{det.track_id}" if det.track_id is not None else "")
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x = max(0, min(det.x1, img.shape[1] - tw - 6))  # keep the label inside the frame
    y = max(th + base + 4, det.y1)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)


def draw_detections(img: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    for det in detections:
        draw_detection(img, det)
    return img


def draw_hud(img: np.ndarray, *, fps: float | None, total_violations: int) -> np.ndarray:
    lines = [f"Violations: {total_violations}"]
    if fps is not None:
        lines.insert(0, f"FPS: {fps:.1f}")
    for i, line in enumerate(lines):
        y = 24 + i * 24
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img
