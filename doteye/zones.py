"""Зоны кадра (ROI) в относительных координатах 0..1."""

from __future__ import annotations

import json
from dataclasses import dataclass

from doteye.tracker import Box, iou


@dataclass(frozen=True)
class Zone:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def clamp(self) -> Zone:
        x1, x2 = sorted((max(0.0, min(1.0, self.x1)), max(0.0, min(1.0, self.x2))))
        y1, y2 = sorted((max(0.0, min(1.0, self.y1)), max(0.0, min(1.0, self.y2))))
        return Zone(self.name, x1, y1, x2, y2)

    def to_pixels(self, width: int, height: int) -> Box:
        z = self.clamp()
        return (
            int(z.x1 * width),
            int(z.y1 * height),
            int(z.x2 * width),
            int(z.y2 * height),
        )


def parse_zones(raw: str | None) -> list[Zone]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    zones: list[Zone] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "зона").strip() or "зона"
        try:
            zone = Zone(
                name=name,
                x1=float(item["x1"]),
                y1=float(item["y1"]),
                x2=float(item["x2"]),
                y2=float(item["y2"]),
            ).clamp()
        except (KeyError, TypeError, ValueError):
            continue
        zones.append(zone)
    return zones


def dump_zones(zones: list[Zone]) -> str:
    return json.dumps(
        [{"name": z.name, "x1": z.x1, "y1": z.y1, "x2": z.x2, "y2": z.y2} for z in zones],
        ensure_ascii=False,
    )


def box_center_in_zone(box: Box, zone: Zone, width: int, height: int) -> bool:
    zx1, zy1, zx2, zy2 = zone.to_pixels(width, height)
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def filter_boxes(
    boxes: list[Box], zones: list[Zone], frame_shape: tuple[int, ...]
) -> list[tuple[Box, str | None]]:
    """Оставить боксы, чей центр попал в зону. Без зон - все боксы."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if not zones:
        return [(box, None) for box in boxes]
    out: list[tuple[Box, str | None]] = []
    for box in boxes:
        hit: str | None = None
        best = 0.0
        for zone in zones:
            zbox = zone.to_pixels(width, height)
            score = iou(box, zbox)
            if box_center_in_zone(box, zone, width, height) or score > 0.05:
                if score >= best:
                    best = score
                    hit = zone.name
        if hit is not None:
            out.append((box, hit))
    return out
