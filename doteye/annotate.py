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


def pixelate(frame: np.ndarray, blocks: int) -> np.ndarray:
    """Пикселизировать весь переданный фрагмент без изменения его размера."""
    if frame.size == 0:
        return frame.copy()
    if blocks < 2:
        raise ValueError("blocks должен быть не меньше 2")
    height, width = frame.shape[:2]
    miniature = cv2.resize(
        frame, (min(blocks, width), min(blocks, height)), interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(miniature, (width, height), interpolation=cv2.INTER_NEAREST)


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
        result[y1:y2, x1:x2] = pixelate(frame[y1:y2, x1:x2], blocks)
    return result


def privacy_frame(frame: np.ndarray, boxes: list[Box], mode: str, blocks: int) -> np.ndarray:
    """Подготовить кадр для отправки без ложного обещания защиты лица.

    Режим face использует каскад OpenCV внутри силуэта человека. Если лицо не
    найдено, пикселизируется весь силуэт - это безопаснее отправки лица.
    """
    if mode == "off":
        return frame.copy()
    if mode == "all":
        return pixelate(frame, blocks)
    if mode == "silhouette":
        return redact_frame(frame, boxes, blocks)
    result = frame.copy()
    if mode == "person":
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box)
            h, w = result.shape[:2]
            x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
            y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
            if x2 > x1 and y2 > y1:
                result[y1:y2, x1:x2] = pixelate(result[y1:y2, x1:x2], blocks)
        return result
    if mode == "face":
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box)
            h, w = result.shape[:2]
            x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
            y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
            crop = result[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            faces = cascade.detectMultiScale(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 1.1, 4)
            if len(faces) == 0:
                result[y1:y2, x1:x2] = pixelate(crop, blocks)
                continue
            for fx, fy, fw, fh in faces:
                result[y1 + fy:y1 + fy + fh, x1 + fx:x1 + fx + fw] = pixelate(
                    crop[fy:fy + fh, fx:fx + fw], blocks
                )
        return result
    return redact_frame(frame, boxes, blocks)


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None
