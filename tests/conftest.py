from __future__ import annotations

import numpy as np
import pytest

from helmet_detection.config import Settings
from helmet_detection.types import Detection


class FakeBoxes:
    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.conf = np.asarray(conf, dtype=float)
        self.cls = np.asarray(cls, dtype=float)
        self.id = None if ids is None else np.asarray(ids, dtype=float)

    def __len__(self):
        return len(self.conf)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeDetector:
    """Stands in for HelmetDetector without loading YOLO."""

    def __init__(self, settings: Settings, detections: list[Detection]):
        self.settings = settings
        self._dets = detections

    def detect(self, frame):
        return list(self._dets)

    track = detect

    def reset_tracker(self):
        pass


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.output_dir = tmp_path / "out"
    s.conf_threshold = 0.4
    s.violation_cooldown_s = 3
    return s


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def make_det(cls=1, conf=0.9, track_id=None, settings: Settings | None = None) -> Detection:
    s = settings or Settings()
    return Detection(10, 10, 80, 120, cls, conf, s.label(cls), track_id, s.is_violation(cls))
