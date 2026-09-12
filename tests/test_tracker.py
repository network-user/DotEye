"""Тесты IoU-трекера: вход, сопровождение, выход."""

from __future__ import annotations

from doteye.tracker import IoUTracker, iou


def test_iou_overlap_and_disjoint() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.0 < iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1.0


def test_enter_then_active_then_exit() -> None:
    tr = IoUTracker(iou_threshold=0.3, max_misses=1)
    u = tr.update([(0, 0, 10, 10)])
    assert len(u.entered) == 1
    assert tr.active_count == 1

    u = tr.update([(1, 1, 11, 11)])
    assert u.entered == []
    assert len(u.active) == 1

    u = tr.update([])
    assert u.exited == []
    u = tr.update([])
    assert len(u.exited) == 1
    assert tr.active_count == 0


def test_two_people_independent() -> None:
    tr = IoUTracker(max_misses=2)
    u = tr.update([(0, 0, 8, 8), (40, 0, 48, 8)])
    assert len(u.entered) == 2
    u = tr.update([(0, 0, 8, 8)])
    assert len(u.active) == 1
    assert tr.active_count == 2
