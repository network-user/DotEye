"""Детектор присутствия.

Отвечает только за вопрос "есть ли человек в кадре" и где именно.
Распознавание личности - отдельный слой (recognizer.py).

Бэкенды (переключаются через DOTEYE_DETECTOR / команду /detector):
  yolo   - ultralytics YOLO (точнее всего, требует torch)
  yunet  - OpenCV FaceDetectorYN (ONNX-модель лица, нужен DOTEYE_FACE_MODEL)
  motion - MOG2-детектор движения (грубое "присутствие", без моделей)
  auto   - первый доступный из yolo -> yunet -> motion

Инференс может работать локально или на отдельном сервере (remote).
При падении remote можно уйти на локальный fallback.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cv2
import numpy as np

from doteye.tracker import iou

if TYPE_CHECKING:
    from doteye.crypto import Crypto

CLASS_PERSON = 0


def _merge_person_boxes(
    boxes: list[tuple[int, int, int, int]],
    shape: tuple[int, ...],
    min_area_fraction: float = 0.004,
    merge_iou: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """Убрать осколки и объединить близкие боксы одного человека.

    YOLO без жёсткого NMS выдаёт кучу мелких квадратов по частям тела. Здесь
    отсеиваются мизерные области (менее min_area_fraction кадра) и склеиваются
    сильно перекрывающиеся соседи, чтобы в кадре остался один цельный
    прямоугольник на человека.
    """
    height, width = int(shape[0]), int(shape[1])
    min_area = width * height * max(0.0, min_area_fraction)
    filtered: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        filtered.append((int(x1), int(y1), int(x2), int(y2)))
    if not filtered:
        return []

    merged: list[tuple[int, int, int, int]] = []
    consumed: set[int] = set()
    for i, box in enumerate(filtered):
        if i in consumed:
            continue
        group = [box]
        consumed.add(i)
        for j in range(i + 1, len(filtered)):
            if j in consumed:
                continue
            if iou(box, filtered[j]) >= merge_iou:
                group.append(filtered[j])
                consumed.add(j)
        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[2] for b in group)
        y2 = max(b[3] for b in group)
        merged.append((x1, y1, x2, y2))
    return merged


class Detector(ABC):
    last_error: str | None = None

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Вернуть список боксов людей: (x1, y1, x2, y2)."""

    @abstractmethod
    def close(self) -> None: ...

    @property
    def backend(self) -> str:
        return type(self).__name__


class LocalDetector(Detector):
    """YOLO в текущем процессе (CPU/GPU)."""

    def __init__(
        self,
        model_path: str,
        device: str,
        min_conf: float,
        imgsz: int = 640,
        nms_iou: float = 0.6,
        person_min_area: float = 0.004,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._device = device
        self._min_conf = min_conf
        self._imgsz = int(imgsz)
        self._nms_iou = max(0.0, min(1.0, float(nms_iou)))
        self._person_min_area = max(0.0, min(0.5, float(person_min_area)))

    @property
    def backend(self) -> str:
        return "yolo"

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        kwargs: dict = {
            "source": frame,
            "classes": [CLASS_PERSON],
            "conf": self._min_conf,
            "verbose": False,
            "device": self._device,
            "imgsz": self._imgsz,
            "iou": self._nms_iou,
            "max_det": 20,
        }
        if self._device == "cuda":
            kwargs["half"] = True
        results = self._model.predict(**kwargs)
        boxes: list[tuple[int, int, int, int]] = []
        for r in results:
            for b in r.boxes.xyxy.cpu().numpy():
                boxes.append(tuple(int(v) for v in b))
        self.last_error = None
        return _merge_person_boxes(
            boxes, frame.shape[:2], self._person_min_area, self._nms_iou,
        )

    def close(self) -> None:
        del self._model


class YuNetDetector(Detector):
    """Детектор лиц OpenCV (YuNet ONNX). Работает без torch.

    Модель не входит в пакет: скачать face_detection_yunet_*.onnx и указать
    путь в DOTEYE_FACE_MODEL.
    """

    def __init__(self, model_path: str, min_conf: float) -> None:
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"модель лица не найдена: {model_path!r}")
        self._min_conf = min_conf
        self._det = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold=min_conf
        )

    @property
    def backend(self) -> str:
        return "yunet"

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame)
        boxes: list[tuple[int, int, int, int]] = []
        if faces is None:
            return boxes
        for f in faces:
            x, y, bw, bh = (int(v) for v in f[:4])
            boxes.append((x, y, x + bw, y + bh))
        return boxes

    def close(self) -> None:
        pass


class MotionDetector(Detector):
    """Грубый fallback: MOG2 находит движущиеся объекты, без моделей и torch.

    Подходит для presence-режима "в кадре что-то/кто-то появилось".
    """

    _BG = "mog2"

    def __init__(self, min_conf: float = 0.5, min_area: int = 2500, warmup: int = 5) -> None:
        self._min_conf = min_conf
        self._min_area = min_area
        self._warmup = warmup
        self._seen = 0
        self._bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25)

    @property
    def backend(self) -> str:
        return "motion"

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        mask = self._bg.apply(frame)
        self._seen += 1
        if self._seen <= self._warmup:
            return []
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        for c in contours:
            if cv2.contourArea(c) < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
        return boxes

    def close(self) -> None:
        pass


class MotionGate:
    """Дешёвый фильтр: нет движения и нет треков - YOLO можно не гонять."""

    def __init__(self, min_pixels: int = 1500) -> None:
        self._prev: np.ndarray | None = None
        self._min_pixels = min_pixels

    def moved(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev is None:
            self._prev = gray
            return True
        diff = cv2.absdiff(self._prev, gray)
        self._prev = gray
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        return int(cv2.countNonZero(th)) >= self._min_pixels


class RemoteDetector(Detector):
    """Передаёт кадры на отдельный сервер обработки (см. remote.py).

    Кадр шифруется AES-256-GCM и уходит POST-ом; сервер возвращает боксы.
    Если сервер недоступен, при наличии fallback идёт локальный детектор,
    иначе пустой список (пайплайн не падает). last_error при этом заполнен.
    """

    def __init__(
        self,
        remote_url: str,
        crypto: "Crypto | None" = None,
        timeout: float = 10.0,
        fallback: Detector | None = None,
        insecure: bool = False,
        ca_cert: str = "",
    ) -> None:
        from doteye.remote import RemoteClient

        self._url = remote_url
        self._crypto = crypto
        self._fallback = fallback
        self.using_fallback = False
        self.last_error: str | None = None
        self._client = (
            RemoteClient(remote_url, crypto, timeout, insecure=insecure, ca_cert=ca_cert)
            if crypto and remote_url
            else None
        )

    @property
    def backend(self) -> str:
        if self.using_fallback and self._fallback is not None:
            return f"remote-fallback:{self._fallback.backend}"
        return "remote"

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._client is None:
            self.last_error = "нет ключа или URL remote"
            return self._fallback_detect(frame)
        try:
            boxes = self._client.detect(frame)
            self.last_error = None
            self.using_fallback = False
            return boxes
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            print(f"[detector] remote недоступен: {exc}")
            return self._fallback_detect(frame)

    def _fallback_detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._fallback is None:
            self.using_fallback = False
            return []
        self.using_fallback = True
        return self._fallback.detect(frame)

    def close(self) -> None:
        if self._fallback is not None:
            self._fallback.close()
        if self._client is not None:
            self._client.close()


def yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


def _build_local(kind: str, model_path: str, device: str, min_conf: float,
                 face_model: str, imgsz: int, nms_iou: float = 0.6,
                 person_min_area: float = 0.004) -> Detector:
    kind = (kind or "auto").lower()

    if kind in ("auto", "yolo") and yolo_available():
        try:
            return LocalDetector(
                model_path, device, min_conf, imgsz=imgsz,
                nms_iou=nms_iou, person_min_area=person_min_area,
            )
        except Exception as exc:
            print(f"[detector] YOLO не загрузился ({exc})")
    elif kind == "yolo":
        print("[detector] ultralytics не установлен")

    if kind in ("auto", "yunet"):
        try:
            return YuNetDetector(face_model or os.getenv("DOTEYE_FACE_MODEL", ""), min_conf)
        except Exception as exc:
            if kind == "yunet":
                print(f"[detector] YuNet недоступен ({exc})")
            else:
                print(f"[detector] YuNet недоступен ({exc}); fallback на motion")

    print("[detector] motion-режим (MOG2): детекция движения, не людей")
    return MotionDetector(min_conf)


def build_detector(kind: str, model_path: str, device: str, min_conf: float,
                   remote: bool, remote_url: str, face_model: str = "",
                   crypto: "Crypto | None" = None, imgsz: int = 640,
                   remote_fallback: bool = False,
                   remote_insecure: bool = False,
                   remote_ca_cert: str = "",
                   nms_iou: float = 0.6,
                   person_min_area: float = 0.004) -> Detector:
    """Собрать детектор. kind: auto | yolo | yunet | motion.

    remote перекрывает локальный бэкенд. auto идёт по цепочке yolo -> yunet -> motion.
    """
    if remote:
        fallback: Detector | None = None
        if remote_fallback:
            fallback = _build_local(
                kind, model_path, device, min_conf, face_model, imgsz,
                nms_iou, person_min_area,
            )
        if not remote_url:
            print("[detector] remote включён, URL пуст")
            if fallback is not None:
                return fallback
        return RemoteDetector(
            remote_url, crypto, fallback=fallback, insecure=remote_insecure,
            ca_cert=remote_ca_cert,
        )

    return _build_local(kind, model_path, device, min_conf, face_model, imgsz,
                        nms_iou, person_min_area)
