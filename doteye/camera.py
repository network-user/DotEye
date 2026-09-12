"""Источники видеокадров.

CameraSource - единый интерфейс над разными источниками:
  - вебка ноутбука (cv2.VideoCapture(0))
  - IP-камера / телефон с приложением типа IP Webcam (RTSP или http MJPEG)
Несколько источников задаются через | :  0|rtsp://host/stream
Реализации только читают кадры, обработка вынесена в pipeline.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np


def parse_sources(source: str | int) -> list[str]:
    raw = str(source).strip()
    if not raw:
        return ["0"]
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    return parts or ["0"]


class CameraSource(ABC):
    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Вернуть следующий кадр (BGR) или None, если кадр недоступен."""

    @abstractmethod
    def close(self) -> None: ...

    @property
    def healthy(self) -> bool:
        return True

    @property
    def last_error(self) -> str | None:
        return None


class OpenCVCamera(CameraSource):
    """Вебка или любой источник, открываемый через cv2.VideoCapture."""

    def __init__(self, source: str | int = 0) -> None:
        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self._fail = 0
        self._last_ok = 0.0
        self._error: str | None = None
        self._reconnects = 0
        if isinstance(source, str) and source.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self._open()

    def _open(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self._source)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def read(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.isOpened():
            self._open()
            self._reconnects += 1
        if self._cap is None:
            self._error = "камера не открылась"
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._fail += 1
            self._error = "нет кадра"
            if self._fail >= 2:
                self._open()
                self._reconnects += 1
                self._fail = 0
                if self._cap is not None:
                    ok, frame = self._cap.read()
                    if ok and frame is not None:
                        self._last_ok = time.time()
                        self._error = None
                        return frame
            return None
        self._fail = 0
        self._error = None
        self._last_ok = time.time()
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def healthy(self) -> bool:
        return self._last_ok > 0 and (time.time() - self._last_ok) < 15.0

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def reconnects(self) -> int:
        return self._reconnects


class RTSPCamera(OpenCVCamera):
    """IP-камера / телефон по RTSP."""


class MJpegCamera(OpenCVCamera):
    """HTTP MJPEG stream (телефон / дешёвая камера)."""


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


def build_cameras(source: str | int) -> list[tuple[str, CameraSource]]:
    """Несколько источников: метка + камера."""
    return [(item, build_camera(item)) for item in parse_sources(source)]
