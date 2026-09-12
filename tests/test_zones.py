"""Тесты зон кадра."""

from __future__ import annotations

from doteye.zones import Zone, dump_zones, filter_boxes, parse_zones


def test_parse_and_dump_roundtrip() -> None:
    raw = '[{"name":"дверь","x1":0,"y1":0.2,"x2":0.4,"y2":1}]'
    zones = parse_zones(raw)
    assert len(zones) == 1
    assert zones[0].name == "дверь"
    again = parse_zones(dump_zones(zones))
    assert again[0].name == "дверь"


def test_parse_bad_json() -> None:
    assert parse_zones("not-json") == []
    assert parse_zones("") == []


def test_filter_without_zones_keeps_all() -> None:
    boxes = [(0, 0, 10, 10)]
    out = filter_boxes(boxes, [], (32, 32, 3))
    assert out == [(boxes[0], None)]


def test_filter_drops_outside() -> None:
    zone = Zone("право", 0.6, 0.0, 1.0, 1.0)
    left = (0, 0, 8, 8)
    right = (22, 0, 30, 8)
    out = filter_boxes([left, right], [zone], (32, 32, 3))
    assert [b for b, _ in out] == [right]
    assert out[0][1] == "право"
