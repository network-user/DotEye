"""Тесты детекторов: фабрика, деградация, motion-бэкенд."""

from __future__ import annotations

import numpy as np

from doteye import detector as det


def test_build_falls_back_to_motion_without_yolo() -> None:
    d = det.build_detector("auto", "no-model.pt", "cpu", 0.5, False, "", "")
    assert isinstance(d, det.MotionDetector)
    d.close()


def test_motion_ignores_first_frames() -> None:
    d = det.MotionDetector(warmup=3)
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    for _ in range(3):
        assert d.detect(frame) == []
    d.close()


def test_yunet_missing_model_raises() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        det.YuNetDetector("definitely-missing.onnx", 0.5)
