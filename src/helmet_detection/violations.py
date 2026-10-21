"""Violation bookkeeping: de-duplication, snapshots and the CSV audit log."""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import CSV_HEADERS, Settings
from .types import Detection


class ViolationLogger:
    """Decides which violations are *new* and records them.

    * With tracking enabled, every track ID is logged **once** – a rider who stays
      in frame for 10 seconds is one violation, not 300.
    * Without track IDs, a global cooldown prevents near-duplicate snapshots.
    """

    def __init__(self, settings: Settings, clock: Callable[[], float] = time.time):
        self.settings = settings
        self._clock = clock
        self._seen_tracks: set[int] = set()
        self._last_saved = float("-inf")
        self.count = 0
        settings.violations_dir.mkdir(parents=True, exist_ok=True)
        if not settings.violations_csv.exists():
            with settings.violations_csv.open("w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADERS)

    def reset(self) -> None:
        self._seen_tracks.clear()
        self._last_saved = float("-inf")

    def _is_new(self, det: Detection, now: float) -> bool:
        if det.track_id is not None:
            if det.track_id in self._seen_tracks:
                return False
            self._seen_tracks.add(det.track_id)
            return True
        return (now - self._last_saved) >= self.settings.violation_cooldown_s

    def process(self, frame: np.ndarray, detections: Iterable[Detection], source: str) -> list[Detection]:
        """Log new violations in *detections*; returns the ones that were logged."""
        now = self._clock()
        new = [d for d in detections if d.is_violation and self._is_new(d, now)]
        if not new:
            return []
        image_name = self._save_snapshot(frame, source)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.settings.violations_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            for d in new:
                writer.writerow([stamp, d.label, f"{d.conf:.2f}", d.track_id if d.track_id is not None else "",
                                 source, image_name])
        self._last_saved = now
        self.count += len(new)
        return new

    def _save_snapshot(self, frame: np.ndarray, source: str) -> str:
        safe_source = "".join(c if c.isalnum() else "_" for c in Path(str(source)).stem)[:40] or "src"
        name = f"violation_{safe_source}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(self.settings.violations_dir / name), frame)
        return name


def read_log(settings: Settings) -> list[list[str]]:
    if not settings.violations_csv.exists():
        return []
    with settings.violations_csv.open(newline="") as f:
        return list(csv.reader(f))
