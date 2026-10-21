from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest
from conftest import FakeBoxes, FakeDetector, FakeResult, make_det

from helmet_detection.cli import build_parser
from helmet_detection.detector import parse_results
from helmet_detection.drawing import draw_detections, draw_hud
from helmet_detection.pipeline import iter_stream, parse_source, process_image
from helmet_detection.violations import ViolationLogger, read_log


def test_parse_results_filters_low_confidence_and_maps_labels(settings):
    boxes = FakeBoxes([[0, 0, 10, 10], [5, 5, 50, 50], [1, 1, 2, 2]], [0.9, 0.8, 0.1], [0, 1, 1], ids=[3, 4, 5])
    dets = parse_results([FakeResult(boxes)], settings)
    assert [d.label for d in dets] == ["With Helmet", "Without Helmet"]
    assert [d.track_id for d in dets] == [3, 4]
    assert [d.is_violation for d in dets] == [False, True]


def test_parse_results_without_track_ids(settings):
    dets = parse_results([FakeResult(FakeBoxes([[0, 0, 5, 5]], [0.7], [1]))], settings)
    assert dets[0].track_id is None


def test_parse_results_handles_empty(settings):
    assert parse_results([FakeResult(FakeBoxes(np.zeros((0, 4)), [], []))], settings) == []


def test_logger_logs_each_track_once(settings, frame):
    logger = ViolationLogger(settings)
    det = make_det(track_id=7, settings=settings)
    assert len(logger.process(frame, [det], "cam")) == 1
    assert logger.process(frame, [det], "cam") == []
    assert len(logger.process(frame, [make_det(track_id=8, settings=settings)], "cam")) == 1
    assert logger.count == 2
    rows = read_log(settings)
    assert rows[0][0] == "Timestamp" and len(rows) == 3
    assert len(list(settings.violations_dir.glob("*.jpg"))) == 2


def test_logger_cooldown_without_tracks(settings, frame):
    now = [100.0]
    logger = ViolationLogger(settings, clock=lambda: now[0])
    det = make_det(settings=settings)
    assert logger.process(frame, [det], "v")
    now[0] += 1
    assert logger.process(frame, [det], "v") == []
    now[0] += 5
    assert logger.process(frame, [det], "v")


def test_logger_ignores_safe_riders(settings, frame):
    logger = ViolationLogger(settings)
    assert logger.process(frame, [make_det(cls=0, settings=settings)], "x") == []
    with settings.violations_csv.open() as f:
        assert len(list(csv.reader(f))) == 1


def test_drawing_modifies_frame(frame, settings):
    out = draw_detections(frame.copy(), [make_det(settings=settings), make_det(cls=0, track_id=2, settings=settings)])
    draw_hud(out, fps=12.3, total_violations=4)
    assert out.any()


def test_process_image_end_to_end(tmp_path, settings, frame):
    img = tmp_path / "riders.jpg"
    cv2.imwrite(str(img), frame)
    detector = FakeDetector(settings, [make_det(settings=settings), make_det(cls=0, settings=settings)])
    res = process_image(detector, ViolationLogger(settings), img)
    assert len(res.detections) == 2 and len(res.new_violations) == 1


def test_process_image_missing_file(settings):
    with pytest.raises(ValueError):
        process_image(FakeDetector(settings, []), None, "nope.jpg")


def test_iter_stream_on_video(tmp_path, settings, frame):
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for _ in range(6):
        w.write(frame)
    w.release()
    detector = FakeDetector(settings, [make_det(track_id=1, settings=settings)])
    logger = ViolationLogger(settings)
    results = list(iter_stream(detector, logger, str(path), frame_skip=2, track=True))
    assert len(results) == 3
    assert logger.count == 1  # same track id across frames → one violation


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source("rtsp://x/y") == "rtsp://x/y"


def test_cli_parser():
    args = build_parser().parse_args(["video", "a.mp4", "--no-show", "--save", "--conf", "0.5"])
    assert args.path == "a.mp4" and not args.show and args.save and args.conf == 0.5
    args = build_parser().parse_args(["train", "--data", "d.yaml"])
    assert args.model.startswith("yolo")
