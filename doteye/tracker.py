"""IoU-трекер: событие на вход и выход, а не на каждый кадр с человеком."""

from __future__ import annotations

from dataclasses import dataclass, field


Box = tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


@dataclass
class Track:
    id: int
    box: Box
    misses: int = 0
    hits: int = 1
    person_name: str | None = None
    confidence: float = 0.0
    zone: str | None = None
    identified: bool = False


@dataclass
class TrackUpdate:
    entered: list[Track] = field(default_factory=list)
    active: list[Track] = field(default_factory=list)
    exited: list[Track] = field(default_factory=list)


class IoUTracker:
    """Жадный матчинг боксов по IoU. Без внешних зависимостей."""

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max(1, int(max_misses))
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    def active_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def update(self, boxes: list[Box]) -> TrackUpdate:
        pairs: list[tuple[float, int, int]] = []
        track_ids = list(self._tracks)
        for tid in track_ids:
            for j, box in enumerate(boxes):
                score = iou(self._tracks[tid].box, box)
                if score >= self.iou_threshold:
                    pairs.append((score, tid, j))
        pairs.sort(reverse=True)

        used_t: set[int] = set()
        used_j: set[int] = set()
        matched: list[tuple[int, int]] = []
        for _, tid, j in pairs:
            if tid in used_t or j in used_j:
                continue
            used_t.add(tid)
            used_j.add(j)
            matched.append((tid, j))

        entered: list[Track] = []
        active: list[Track] = []
        for tid, j in matched:
            track = self._tracks[tid]
            track.box = boxes[j]
            track.misses = 0
            track.hits += 1
            active.append(track)

        for j, box in enumerate(boxes):
            if j in used_j:
                continue
            track = Track(id=self._next_id, box=box)
            self._next_id += 1
            self._tracks[track.id] = track
            entered.append(track)
            active.append(track)

        exited: list[Track] = []
        for tid in list(self._tracks):
            if tid in used_t or any(t.id == tid for t in entered):
                continue
            track = self._tracks[tid]
            track.misses += 1
            if track.misses > self.max_misses:
                exited.append(track)
                del self._tracks[tid]

        return TrackUpdate(entered=entered, active=active, exited=exited)
