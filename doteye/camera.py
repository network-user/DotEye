"""Источники видеокадров.

CameraSource - единый интерфейс над разными источниками:
  - вебка ноутбука (cv2.VideoCapture(0))
  - IP-камера / телефон с приложением типа IP Webcam (RTSP или http MJPEG)
Несколько источников задаются через | :  0|rtsp://host/stream
Реализации только читают кадры, обработка вынесена в pipeline.
"""

from __future__ import annotations

import os
import sys
import threading
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
        self._next_open_at = 0.0
        if isinstance(source, str) and source.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self._open()

    def _open(self) -> None:
        if self._cap is not None:
            self._cap.release()
        # These properties are ignored by old OpenCV builds, but avoid an
        # indefinitely blocked RTSP open/read where the backend supports them.
        self._cap = self._create_capture()
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000)
        except Exception:
            pass

    def _create_capture(self) -> cv2.VideoCapture:
        """Открыть источник, выбрав рабочий backend.

        На Windows backend по умолчанию (MSMF) умеет открыть устройство, но
        затем не отдаёт кадры на части USB/встроенных камер. DSHOW читает их
        надёжнее, поэтому для локальных индексов пробуем его первым и
        откатываемся на стандартное открытие, если кадра всё равно нет.
        """
        if isinstance(self._source, int) and sys.platform == "win32":
            cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
            cap.release()
        return cv2.VideoCapture(self._source)

    def read(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.isOpened():
            if time.monotonic() < self._next_open_at:
                return None
            self._open()
        if self._cap is None:
            self._error = "камера не открылась"
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._fail += 1
            self._error = "нет кадра"
            if self._fail >= 2:
                # Exponential backoff prevents a broken stream from spinning
                # VideoCapture open attempts at full CPU/network speed.
                self._next_open_at = time.monotonic() + min(30.0, 2.0 ** min(self._fail, 5))
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                self._reconnects += 1
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


class BufferedCamera(CameraSource):
    """Неблокирующая обёртка над камерой с одним фоновым reader-потоком.

    OpenCV может зависнуть внутри ``read`` на недоступном RTSP/MJPEG потоке.
    Изоляция чтения гарантирует, что такая камера не останавливает обработку
    остальных камер. Храним только последний кадр: очередь кадров здесь лишь
    добавила бы задержку и не помогла бы детекции в реальном времени.
    """

    def __init__(self, camera: CameraSource, retry_min: float = 0.05, retry_max: float = 5.0) -> None:
        self._camera = camera
        self._retry_min = max(0.01, retry_min)
        self._retry_max = max(self._retry_min, retry_max)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame: np.ndarray | None = None
        self._last_frame_at = 0.0
        self._error: str | None = None
        self._thread = threading.Thread(target=self._reader, name="doteye-camera-reader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        pause = self._retry_min
        while not self._stop.is_set():
            try:
                frame = self._camera.read()
            except Exception as exc:
                frame = None
                error = f"ошибка чтения: {type(exc).__name__}"
            else:
                error = getattr(self._camera, "last_error", None)
            if frame is not None:
                with self._lock:
                    self._frame = frame
                    self._last_frame_at = time.time()
                    self._error = None
                pause = self._retry_min
                continue
            with self._lock:
                self._error = error or "нет кадра"
            self._stop.wait(pause)
            pause = min(self._retry_max, pause * 2)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def close(self) -> None:
        self._stop.set()
        # release may unblock VideoCapture.read on common OpenCV backends.
        self._camera.close()
        self._thread.join(timeout=1.0)

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._last_frame_at > 0 and (time.time() - self._last_frame_at) < 15.0

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def reconnects(self) -> int:
        return int(getattr(self._camera, "reconnects", 0))


class RTSPCamera(OpenCVCamera):
    """IP-камера / телефон по RTSP."""


class MJpegCamera(OpenCVCamera):
    """HTTP MJPEG stream (телефон / дешёвая камера)."""


def build_camera(source: str | int) -> CameraSource:
    """Фабрика: выбирает реализацию по типу source."""
    if isinstance(source, int):
        return BufferedCamera(OpenCVCamera(source))
    s = str(source).strip()
    if s.lower().startswith("rtsp://"):
        return BufferedCamera(RTSPCamera(s))
    if s.startswith(("http://", "https://")):
        return BufferedCamera(MJpegCamera(s))
    if s.isdigit():
        return BufferedCamera(OpenCVCamera(int(s)))
    return BufferedCamera(OpenCVCamera(s))


def build_cameras(source: str | int) -> list[tuple[str, CameraSource]]:
    """Несколько источников: метка + камера."""
    return [(item, build_camera(item)) for item in parse_sources(source)]
