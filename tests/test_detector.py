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


def test_merge_person_boxes_filters_tiny_squares() -> None:
    shape = (480, 640)
    # Большой человек + совсем мелкий осколок в углу.
    boxes = [(10, 10, 300, 470), (500, 400, 508, 408)]
    assert det._merge_person_boxes(boxes, shape) == [(10, 10, 300, 470)]


def test_merge_person_boxes_merges_overlapping() -> None:
    shape = (480, 640)
    boxes = [(100, 50, 300, 470), (100, 50, 280, 450)]
    assert det._merge_person_boxes(boxes, shape) == [(100, 50, 300, 470)]


def test_merge_person_boxes_all_tiny_is_empty() -> None:
    shape = (480, 640)
    assert det._merge_person_boxes([(0, 0, 5, 5), (100, 100, 110, 110)], shape) == []


def test_merge_person_boxes_keeps_distinct() -> None:
    shape = (480, 640)
    # Два отдельных человека без перекрытия остаются двумя боксами.
    boxes = [(10, 10, 150, 470), (400, 10, 630, 470)]
    assert det._merge_person_boxes(boxes, shape) == boxes
