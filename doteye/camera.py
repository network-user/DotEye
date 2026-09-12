"""Источники видеокадров.

CameraSource - единый интерфейс над разными источниками:
  - вебка ноутбука (cv2.VideoCapture(0))
  - IP-камера / телефон с приложением типа IP Webcam (RTSP или http MJPEG)
Реализации только читают кадры, обработка вынесена в pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class CameraSource(ABC):
    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Вернуть следующий кадр (BGR) или None, если кадр недоступен."""

    @abstractmethod
    def close(self) -> None: ...


class OpenCVCamera(CameraSource):
    """Вебка или любой источник, открываемый через cv2.VideoCapture."""

    def __init__(self, source: str | int = 0) -> None:
        self._cap = cv2.VideoCapture(source)

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class RTSPCamera(OpenCVCamera):
    """IP-камера / телефон по RTSP.

    Пример адреса для телефона с приложением IP Webcam:
        http://<ip>:8080/video
    Пример RTSP-камеры:
        rtsp://user:pass@<ip>:554/stream
    """

    def __init__(self, url: str) -> None:
        super().__init__(url)


class MJpegCamera(OpenCVCamera):
    """HTTP MJPEG stream (телефон / дешёвая камера)."""

    def __init__(self, url: str) -> None:
        super().__init__(url)


def build_camera(source: str | int) -> CameraSource:
    """Фабрика: выбирает реализацию по типу source."""
    if isinstance(source, int):
        return OpenCVCamera(source)
    s = str(source).strip()
    if s.lower().startswith("rtsp://"):
        return RTSPCamera(s)
    if s.startswith(("http://", "https://")):
        return MJpegCamera(s)
    if s.isdigit():
        return OpenCVCamera(int(s))
    return OpenCVCamera(s)
