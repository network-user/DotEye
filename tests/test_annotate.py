"""Тесты кропа и подписи кадра."""

from __future__ import annotations

import numpy as np

from doteye.annotate import annotate, crop_box, encode_jpeg


def test_crop_and_annotate() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    crop = crop_box(frame, (5, 5, 15, 15), pad=0.0, min_size=1)
    assert crop is not None
    assert crop.shape[0] == 10
    vis = annotate(frame, [((5, 5, 15, 15), "alice", (0, 255, 0))])
    assert vis.shape == frame.shape
    jpeg = encode_jpeg(vis, 80)
    assert jpeg is not None
    assert jpeg[:2] == b"\xff\xd8"
