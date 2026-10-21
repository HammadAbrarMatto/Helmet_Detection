"""Source-agnostic processing used by the CLI, the GUI and the web app."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detector import HelmetDetector
from .drawing import draw_detections, draw_hud
from .types import Detection
from .violations import ViolationLogger


@dataclass
class FrameResult:
    frame: np.ndarray
    detections: list[Detection]
    new_violations: list[Detection]
    frame_index: int = 0
    fps: float | None = None
    stats: dict = field(default_factory=dict)


def process_frame(
    detector: HelmetDetector,
    logger: ViolationLogger | None,
    frame: np.ndarray,
    source: str,
    *,
    track: bool = False,
) -> FrameResult:
    detections = detector.track(frame) if track else detector.detect(frame)
    draw_detections(frame, detections)
    new = logger.process(frame, detections, source) if logger else []
    return FrameResult(frame=frame, detections=detections, new_violations=new)


def process_image(detector: HelmetDetector, logger: ViolationLogger | None, path: str | Path) -> FrameResult:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not read image: {path}")
    if logger:
        logger.reset()  # each image is independent
    return process_frame(detector, logger, frame, str(path), track=False)


def parse_source(source: str | int) -> str | int:
    """'0' → 0 (webcam index); files / RTSP / HTTP URLs are passed through."""
    if isinstance(source, int):
        return source
    return int(source) if str(source).isdigit() else source


def iter_stream(
    detector: HelmetDetector,
    logger: ViolationLogger | None,
    source: str | int,
    *,
    frame_skip: int | None = None,
    track: bool | None = None,
    max_width: int | None = 1280,
) -> Iterator[FrameResult]:
    """Yield annotated frames from a video file, webcam index, or RTSP/HTTP stream."""
    s = detector.settings
    frame_skip = max(1, frame_skip or s.frame_skip)
    track = s.track if track is None else track
    src = parse_source(source)
    label = "webcam" if isinstance(src, int) else Path(str(src)).name

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")
    if track:
        detector.reset_tracker()
    if logger:
        logger.reset()

    idx, t_prev, fps = 0, time.perf_counter(), None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % frame_skip:
                continue
            if max_width and frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            result = process_frame(detector, logger, frame, label, track=track)
            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = inst if fps is None else 0.9 * fps + 0.1 * inst
            t_prev = now
            draw_hud(result.frame, fps=fps, total_violations=logger.count if logger else 0)
            result.frame_index, result.fps = idx, fps
            yield result
    finally:
        cap.release()


def video_properties(source: str | int) -> tuple[float, int]:
    cap = cv2.VideoCapture(parse_source(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n
