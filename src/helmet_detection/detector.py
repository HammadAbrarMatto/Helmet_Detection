"""Thin wrapper around an Ultralytics YOLO model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings
from .types import Detection


def parse_results(results: Iterable[Any], settings: Settings) -> list[Detection]:
    """Convert Ultralytics ``Results`` objects into plain :class:`Detection` objects."""
    detections: list[Detection] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        classes = _to_numpy(boxes.cls).astype(int)
        ids = getattr(boxes, "id", None)
        ids = _to_numpy(ids).astype(int) if ids is not None else [None] * len(confs)

        for (x1, y1, x2, y2), conf, cls, tid in zip(xyxy, confs, classes, ids, strict=False):
            if conf < settings.conf_threshold:
                continue
            detections.append(
                Detection(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    cls=int(cls), conf=float(conf), label=settings.label(int(cls)),
                    track_id=None if tid is None else int(tid),
                    is_violation=settings.is_violation(int(cls)),
                )
            )
    return detections


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


class HelmetDetector:
    """Loads YOLO weights once and runs detection / tracking on frames."""

    def __init__(self, settings: Settings | None = None, weights: str | Path | None = None):
        from ultralytics import YOLO  # heavy import kept local

        self.settings = settings or Settings()
        path = Path(weights or self.settings.weights)
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{path}'. Train a model "
                "(`helmet-detect train ...`) or copy best.pt into the weights/ folder."
            )
        self.model = YOLO(str(path))
        names = getattr(self.model, "names", None)
        if isinstance(names, dict) and names:
            # Trust the class names stored inside the checkpoint.
            self.settings.class_names = tuple(names[i] for i in sorted(names))

    def _kwargs(self) -> dict[str, Any]:
        s = self.settings
        kw: dict[str, Any] = dict(conf=s.conf_threshold, iou=s.iou_threshold, imgsz=s.imgsz,
                                  device=s.device, verbose=False)
        if s.half:
            kw["half"] = True  # FP16 on CUDA
        return kw

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return parse_results(self.model.predict(frame, **self._kwargs()), self.settings)

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Detect + assign persistent track IDs across frames (ByteTrack/BoT-SORT)."""
        results = self.model.track(frame, persist=True, tracker=self.settings.tracker, **self._kwargs())
        return parse_results(results, self.settings)

    def reset_tracker(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        for t in getattr(predictor, "trackers", []) or []:
            t.reset()
