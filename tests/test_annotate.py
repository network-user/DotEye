"""Тесты кропа и подписи кадра."""

from __future__ import annotations

import numpy as np

from doteye.annotate import annotate, crop_box, encode_jpeg, redact_frame


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


def test_redact_frame_hides_background_and_pixelates_person() -> None:
    frame = np.full((40, 40, 3), 200, dtype=np.uint8)
    frame[10:30, 10:30] = np.random.default_rng(4).integers(0, 255, (20, 20, 3), dtype=np.uint8)
    redacted = redact_frame(frame, [(10, 10, 30, 30)], blocks=4)
    assert np.all(redacted[:10] == 20)
    assert np.unique(redacted[10:30, 10:30].reshape(-1, 3), axis=0).shape[0] <= 16
