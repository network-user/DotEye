"""Runtime-настройки: значения из env плюс переопределения из Telegram-чата.

Команды бота пишут в Storage (таблица settings), пайплайн читает оттуда
на каждом шаге. Так смена режима/камеры из чата применяется к уже
запущенному пайплайну без перезапуска процесса.

Приоритет: значение из Storage, иначе env-дефолт из Settings.
"""

from __future__ import annotations

from doteye.config import Settings
from doteye.storage import Storage


class Runtime:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self._settings = settings
        self._storage = storage

    # -- переопределяемые из чата --------------------------------------

    def _get_str(self, key: str, default: str) -> str:
        value = self._storage.get(key)
        return value if value not in (None, "") else default

    def _get_float(self, key: str, default: float) -> float:
        value = self._storage.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @property
    def detect_mode(self) -> str:
        return self._get_str("detect_mode", self._settings.detect_mode)

    @detect_mode.setter
    def detect_mode(self, value: str) -> None:
        self._storage.set("detect_mode", value)

    @property
    def camera_source(self) -> str:
        return self._get_str("camera_source", self._settings.camera_source)

    @camera_source.setter
    def camera_source(self, value: str) -> None:
        self._storage.set("camera_source", value)

    @property
    def detector_backend(self) -> str:
        return self._get_str("detector_backend", self._settings.detector_backend)

    @detector_backend.setter
    def detector_backend(self, value: str) -> None:
        self._storage.set("detector_backend", value)

    @property
    def min_confidence(self) -> float:
        return self._get_float("min_confidence", self._settings.min_confidence)

    @min_confidence.setter
    def min_confidence(self, value: float) -> None:
        self._storage.set("min_confidence", str(value))

    @property
    def cooldown_seconds(self) -> float:
        return self._get_float("cooldown_seconds", self._settings.cooldown_seconds)

    @cooldown_seconds.setter
    def cooldown_seconds(self, value: float) -> None:
        self._storage.set("cooldown_seconds", str(value))

    @property
    def detection_interval(self) -> float:
        return self._get_float("detection_interval", self._settings.detection_interval)

    @detection_interval.setter
    def detection_interval(self, value: float) -> None:
        self._storage.set("detection_interval", str(value))

    @property
    def face_threshold(self) -> float:
        return self._get_float("face_threshold", self._settings.face_threshold)

    @face_threshold.setter
    def face_threshold(self, value: float) -> None:
        self._storage.set("face_threshold", str(value))

    # -- только из env (не меняются на ходу) ---------------------------

    @property
    def model_path(self) -> str:
        return self._settings.model_path

    @property
    def device(self) -> str:
        return self._settings.device

    @property
    def remote_processing(self) -> bool:
        return self._settings.remote_processing

    @property
    def remote_url(self) -> str:
        return self._settings.remote_url

    @property
    def jpeg_quality(self) -> int:
        return self._settings.jpeg_quality

    @property
    def face_model(self) -> str:
        return self._settings.face_model

    @property
    def events_limit(self) -> int:
        return self._settings.events_limit
