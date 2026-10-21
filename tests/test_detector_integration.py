"""Runs the real Ultralytics stack on an untrained 2-class YOLO built from YAML.

No weights are downloaded; this only checks that the wrapper integrates correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

ultralytics = pytest.importorskip("ultralytics")


@pytest.fixture(scope="module")
def tiny_weights(tmp_path_factory):
    from ultralytics import YOLO

    model = YOLO("yolo11n.yaml")
    model.model.names = {0: "With Helmet", 1: "Without Helmet"}
    path = tmp_path_factory.mktemp("w") / "tiny.pt"
    model.save(str(path))
    return path


def test_detector_runs_detect_and_track(tiny_weights, settings):
    from helmet_detection.detector import HelmetDetector

    settings.conf_threshold = 0.0
    det = HelmetDetector(settings, weights=tiny_weights)
    frame = np.random.default_rng(0).integers(0, 255, (320, 320, 3), dtype=np.uint8)
    assert isinstance(det.detect(frame), list)
    assert isinstance(det.track(frame), list)


def test_detector_missing_weights(settings, tmp_path):
    from helmet_detection.detector import HelmetDetector

    with pytest.raises(FileNotFoundError):
        HelmetDetector(settings, weights=tmp_path / "missing.pt")
