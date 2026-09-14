"""Тесты камеры: выбор backend для локальных устройств на Windows.

cv2 не открывается по-настоящему: проверяем логику выбора backend через
подмену cv2.VideoCapture, чтобы тест не зависел от наличия физической камеры.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from doteye import camera as camera_module
from doteye.camera import OpenCVCamera


class _FakeCapture:
    def __init__(self, *, opened: bool = True, frame: np.ndarray | None = None) -> None:
        self._opened = opened
        self._frame = frame

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if self._frame is None:
            return False, None
        return True, self._frame

    def set(self, *_args) -> bool:
        return True

    def release(self) -> None:
        self._opened = False


def test_windows_local_camera_prefers_dshow(monkeypatch) -> None:
    if sys.platform != "win32":
        pytest.skip("DSHOW только на Windows")
    calls: list[tuple] = []
    good_frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def fake_capture(source, api=None):
        calls.append((source, api))
        if api == camera_module.cv2.CAP_DSHOW:
            return _FakeCapture(opened=True, frame=good_frame)
        return _FakeCapture(opened=True, frame=None)

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", fake_capture)
    cam = OpenCVCamera(0)
    assert calls[0] == (0, camera_module.cv2.CAP_DSHOW)
    assert cam.read() is good_frame
    cam.close()


def test_windows_local_camera_falls_back_when_dshow_empty(monkeypatch) -> None:
    if sys.platform != "win32":
        pytest.skip("DSHOW только на Windows")
    calls: list[tuple] = []
    good_frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def fake_capture(source, api=None):
        calls.append((source, api))
        if api == camera_module.cv2.CAP_DSHOW:
            return _FakeCapture(opened=True, frame=None)
        return _FakeCapture(opened=True, frame=good_frame)

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", fake_capture)
    cam = OpenCVCamera(0)
    assert calls[0][1] == camera_module.cv2.CAP_DSHOW
    assert len(calls) == 2 and calls[1][1] is None
    assert cam.read() is good_frame
    cam.close()


def test_network_source_uses_default_backend(monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_capture(source, api=None):
        calls.append((source, api))
        return _FakeCapture(opened=True, frame=np.zeros((2, 2, 3), dtype=np.uint8))

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", fake_capture)
    cam = OpenCVCamera("rtsp://cam.local/stream")
    assert calls == [("rtsp://cam.local/stream", None)]
    cam.close()
