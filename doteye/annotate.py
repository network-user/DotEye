"""Кроп бокса и подпись кадра для Telegram."""

from __future__ import annotations

import cv2
import numpy as np

from doteye.tracker import Box


def crop_box(frame: np.ndarray, box: Box, pad: float = 0.2, min_size: int = 64) -> np.ndarray | None:
    """Вырезать бокс с запасом. None, если кадр пустой."""
    if frame.size == 0:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(bw * pad), int(bh * pad)
    extra = 0
    if min(bw, bh) < min_size:
        extra = (min_size - min(bw, bh)) // 2
    xa = max(0, x1 - px - extra)
    ya = max(0, y1 - py - extra)
    xb = min(w, x2 + px + extra)
    yb = min(h, y2 + py + extra)
    if xb <= xa or yb <= ya:
        return None
    return frame[ya:yb, xa:xb].copy()


def annotate(
    frame: np.ndarray,
    items: list[tuple[Box, str, tuple[int, int, int] | None]],
) -> np.ndarray:
    """Копия кадра с прямоугольниками и подписями."""
    vis = frame.copy()
    for box, label, color in items:
        bgr = color or (40, 200, 40)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), bgr, 2)
        text = label or ""
        if not text:
            continue
        ty = y1 - 8 if y1 > 18 else y1 + 16
        cv2.putText(
            vis, text[:48], (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA,
        )
    return vis


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None
