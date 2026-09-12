"""Тесты каталога YOLO-моделей."""

from __future__ import annotations

from doteye import models


def test_catalog_not_empty_and_has_default() -> None:
    assert models.DEFAULT_MODEL in models.YOLO_MODELS
    assert len(models.YOLO_MODELS) >= 3


def test_describe_known_and_unknown() -> None:
    text = models.describe("yolov8s.pt")
    assert "YOLOv8s" in text
    assert "Скорость" in text

    assert "не из каталога" in models.describe("custom.pt")


def test_comparison_mentions_all_models() -> None:
    text = models.comparison()
    for info in models.YOLO_MODELS.values():
        assert info.title in text


def test_get_model() -> None:
    assert models.get_model("yolov8x.pt") is not None
    assert models.get_model("nope.pt") is None
