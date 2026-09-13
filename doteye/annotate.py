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


def redact_frame(frame: np.ndarray, boxes: list[Box], blocks: int = 12) -> np.ndarray:
    """Скрыть окружение и грубо пикселизировать обнаруженные объекты.

    В результирующем кадре остаются только пикселизированные силуэты внутри
    боксов. Это безопаснее размытия: ни лицо, ни детали помещения не попадают
    в Telegram даже при доступе к истории чата.
    """
    if frame.size == 0:
        return frame.copy()
    if blocks < 2:
        raise ValueError("blocks должен быть не меньше 2")
    height, width = frame.shape[:2]
    # Однотонный фон не сохраняет контуры, предметы и текст из помещения.
    result = np.full_like(frame, (20, 20, 20))
    for raw_box in boxes:
        x1, y1, x2, y2 = (int(value) for value in raw_box)
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        small_width = min(blocks, crop.shape[1])
        small_height = min(blocks, crop.shape[0])
        miniature = cv2.resize(crop, (small_width, small_height), interpolation=cv2.INTER_AREA)
        result[y1:y2, x1:x2] = cv2.resize(
            miniature, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST,
        )
    return result


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None
