"""Runtime-настройки: значения из env плюс переопределения из Telegram-чата.

Команды бота пишут в Storage (таблица settings), пайплайн читает оттуда
на каждом шаге. Так смена режима/камеры из чата применяется к уже
запущенному пайплайну без перезапуска процесса.

Приоритет: значение из Storage, иначе env-дефолт из Settings.
"""

from __future__ import annotations

from datetime import datetime

from doteye.config import Settings
from doteye.storage import Storage


def in_quiet_hours(spec: str, now: datetime | None = None) -> bool:
    """True, если now попадает в интервал HH:MM-HH:MM (через полночь можно)."""
    text = (spec or "").strip().lower()
    if not text or text in ("0", "off", "выкл", "-"):
        return False
    if "-" not in text:
        return False
    start_s, end_s = text.split("-", 1)
    try:
        sh, sm = (int(x) for x in start_s.split(":"))
        eh, em = (int(x) for x in end_s.split(":"))
        if not (0 <= sh <= 23 and 0 <= eh <= 23 and 0 <= sm <= 59 and 0 <= em <= 59):
            return False
    except ValueError:
        return False
    now = now or datetime.now().astimezone()
    minutes = now.hour * 60 + now.minute
    start = sh * 60 + sm
    end = eh * 60 + em
    if start == end:
        return True
    if start < end:
        return start <= minutes < end
    return minutes >= start or minutes < end


class Runtime:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self._settings = settings
        self._storage = storage

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

    def _get_int(self, key: str, default: int) -> int:
        return int(self._get_float(key, float(default)))

    def _get_bool(self, key: str, default: bool) -> bool:
        value = self._storage.get(key)
        if value is None:
            return default
        return value.strip().lower() in ("1", "true", "yes", "on", "вкл")

    # -- переопределяемые из чата --------------------------------------

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

    @property
    def model_path(self) -> str:
        return self._get_str("model_path", self._settings.model_path)

    @model_path.setter
    def model_path(self, value: str) -> None:
        self._storage.set("model_path", value)

    @property
    def device(self) -> str:
        return self._get_str("device", self._settings.device)

    @device.setter
    def device(self, value: str) -> None:
        self._storage.set("device", value)

    @property
    def armed(self) -> bool:
        return self._get_bool("armed", self._settings.armed)

    @armed.setter
    def armed(self, value: bool) -> None:
        self._storage.set("armed", "1" if value else "0")

    @property
    def quiet_hours(self) -> str:
        return self._get_str("quiet_hours", self._settings.quiet_hours)

    @quiet_hours.setter
    def quiet_hours(self, value: str) -> None:
        self._storage.set("quiet_hours", value)

    @property
    def notify_exit(self) -> bool:
        return self._get_bool("notify_exit", self._settings.notify_exit)

    @notify_exit.setter
    def notify_exit(self, value: bool) -> None:
        self._storage.set("notify_exit", "1" if value else "0")

    @property
    def remote_processing(self) -> bool:
        return self._get_bool("remote_processing", self._settings.remote_processing)

    @remote_processing.setter
    def remote_processing(self, value: bool) -> None:
        self._storage.set("remote_processing", "1" if value else "0")

    @property
    def remote_url(self) -> str:
        return self._get_str("remote_url", self._settings.remote_url)

    @remote_url.setter
    def remote_url(self, value: str) -> None:
        self._storage.set("remote_url", value)

    @property
    def remote_fallback(self) -> bool:
        return self._get_bool("remote_fallback", self._settings.remote_fallback)

    @remote_fallback.setter
    def remote_fallback(self, value: bool) -> None:
        self._storage.set("remote_fallback", "1" if value else "0")

    @property
    def remote_insecure(self) -> bool:
        return self._get_bool("remote_insecure", self._settings.remote_insecure)

    @property
    def imgsz(self) -> int:
        return max(160, self._get_int("imgsz", self._settings.imgsz))

    @imgsz.setter
    def imgsz(self, value: int) -> None:
        self._storage.set("imgsz", str(int(value)))

    @property
    def track_max_misses(self) -> int:
        return max(1, self._get_int("track_max_misses", self._settings.track_max_misses))

    @property
    def zones_json(self) -> str:
        return self._get_str("zones", self._settings.zones)

    @zones_json.setter
    def zones_json(self, value: str) -> None:
        self._storage.set("zones", value)

    @property
    def events_max(self) -> int:
        return max(10, self._get_int("events_max", self._settings.events_max))

    @property
    def events_ttl_days(self) -> float:
        return self._get_float("events_ttl_days", self._settings.events_ttl_days)

    def is_quiet(self, now: datetime | None = None) -> bool:
        return in_quiet_hours(self.quiet_hours, now)

    def should_notify(self, now: datetime | None = None) -> bool:
        return self.armed and not self.is_quiet(now)

    # -- только из env (не меняются на ходу) ---------------------------

    @property
    def jpeg_quality(self) -> int:
        return self._settings.jpeg_quality

    @property
    def face_model(self) -> str:
        return self._settings.face_model

    @property
    def events_limit(self) -> int:
        return self._settings.events_limit
