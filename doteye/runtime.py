"""Runtime-настройки: значения из env плюс переопределения из Telegram-чата.

Команды бота пишут в Storage (таблица settings), пайплайн читает оттуда
на каждом шаге. Так смена режима/камеры из чата применяется к уже
запущенному пайплайну без перезапуска процесса.

Приоритет: значение из Storage, иначе env-дефолт из Settings.
"""

from __future__ import annotations

import json
from datetime import datetime

from doteye.config import (
    ConfigurationError,
    Settings,
    validate_camera_name,
    validate_camera_source,
    validate_quiet_hours,
    validate_remote_url,
)
from doteye.storage import Storage

_CAMERA_OVERRIDE_KEYS = {
    "cooldown_seconds": "float",
    "notify_enter": "bool",
    "notify_exit": "bool",
    "enabled": "bool",
    "quiet_hours": "quiet",
    "detect_mode": "mode",
}


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


def _normalize_camera_override(key: str, value: object) -> object | None:
    """Привести значение per-camera override к безопасному типу.

    None означает «сбросить override, вернуться к глобальному значению».
    """
    kind = _CAMERA_OVERRIDE_KEYS.get(key)
    if kind is None:
        raise ConfigurationError(f"камера: неизвестная настройка {key}")
    if value is None:
        return None
    if kind == "float":
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"камера.{key}: нужно число") from exc
        if not 0 <= number <= 86400:
            raise ConfigurationError(f"камера.{key}: значение должно быть 0..86400")
        return number
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on", "вкл", "да"):
            return True
        if text in ("0", "false", "no", "off", "выкл", "нет"):
            return False
        raise ConfigurationError(f"камера.{key}: ожидается да/нет")
    if kind == "quiet":
        return validate_quiet_hours(str(value))
    if kind == "mode":
        text = str(value).strip().lower()
        if text not in {"presence", "identity"}:
            raise ConfigurationError(f"камера.{key}: presence или identity")
        return text
    raise ConfigurationError(f"камера.{key}: неизвестный тип настройки")


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
            result = float(value)
            return result if result == result and result not in (float("inf"), float("-inf")) else default
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
        value = self._get_str("detect_mode", self._settings.detect_mode)
        return value if value in {"presence", "identity"} else self._settings.detect_mode

    @detect_mode.setter
    def detect_mode(self, value: str) -> None:
        if value not in {"presence", "identity"}:
            raise ConfigurationError("detect_mode: presence или identity")
        self._storage.set("detect_mode", value)

    @property
    def camera_source(self) -> str:
        value = self._get_str("camera_source", self._settings.camera_source)
        try:
            return validate_camera_source(
                value,
                allowed_hosts=self._settings.allowed_url_hosts,
                environment=self._settings.environment,
            )
        except ConfigurationError:
            return self._settings.camera_source

    @camera_source.setter
    def camera_source(self, value: str) -> None:
        self._storage.set("camera_source", validate_camera_source(
            value,
            allowed_hosts=self._settings.allowed_url_hosts,
            environment=self._settings.environment,
        ))

    def camera_sources(self) -> list[str]:
        return [part.strip() for part in self.camera_source.split("|") if part.strip()]

    @property
    def camera_parallel(self) -> bool:
        return self._settings.camera_parallel

    @property
    def camera_names(self) -> list[str]:
        """Имена камер по порядку источников. Хранятся CSV в settings."""
        count = len(self.camera_sources())
        raw = self._get_str("camera_names", "")
        names = [part.strip() for part in raw.split(",")] if raw else []
        default = list(self._settings.camera_names)
        names.extend([""] * (count - len(names)))
        defaults = list(default) + [""] * max(0, count - len(default))
        out: list[str] = []
        for index in range(count):
            candidate = names[index] if index < len(names) else ""
            source = candidate or (defaults[index] if index < len(defaults) else "")
            try:
                out.append(validate_camera_name(source))
            except ConfigurationError:
                out.append("")
        return out

    def camera_name(self, index: int) -> str:
        names = self.camera_names
        return names[index] if 0 <= index < len(names) else ""

    @camera_names.setter
    def camera_names(self, value: list[str] | tuple[str, ...]) -> None:
        count = len(self.camera_sources())
        cleaned = [validate_camera_name(item) for item in list(value)[:count]]
        cleaned.extend([""] * (count - len(cleaned)))
        self._storage.set("camera_names", ",".join(cleaned))

    def set_camera_name(self, index: int, name: str) -> None:
        names = list(self.camera_names)
        if not 0 <= index < len(names):
            raise ConfigurationError("камера с таким номером не найдена")
        names[index] = validate_camera_name(name)
        self.camera_names = names

    def _camera_overrides(self) -> dict[str, dict[str, object]]:
        raw = self._get_str("camera_overrides", "")
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        out: dict[str, dict[str, object]] = {}
        for key, value in parsed.items():
            if isinstance(value, dict):
                out[str(key)] = dict(value)
        return out

    def camera_override(self, index: int) -> dict[str, object]:
        return dict(self._camera_overrides().get(str(index), {}))

    def set_camera_override(self, index: int, key: str, value: object) -> None:
        if key not in _CAMERA_OVERRIDE_KEYS:
            raise ConfigurationError(f"камера: неизвестная настройка {key}")
        rounded = _normalize_camera_override(key, value)
        data = self._camera_overrides()
        entry = data.setdefault(str(index), {})
        if rounded is None:
            entry.pop(key, None)
        else:
            entry[key] = rounded
        if not entry:
            data.pop(str(index), None)
        self._storage.set("camera_overrides", json.dumps(data, ensure_ascii=False))

    def clear_camera_override(self, index: int, key: str) -> None:
        data = self._camera_overrides()
        entry = data.get(str(index))
        if not entry:
            return
        entry.pop(key, None)
        if not entry:
            data.pop(str(index), None)
        self._storage.set("camera_overrides", json.dumps(data, ensure_ascii=False))

    def camera_cooldown(self, index: int) -> float:
        value = self.camera_override(index).get("cooldown_seconds")
        if isinstance(value, (int, float)):
            return float(value)
        return self.cooldown_seconds

    def camera_notify_enter(self, index: int) -> bool:
        value = self.camera_override(index).get("notify_enter")
        return bool(value) if isinstance(value, bool) else self.should_notify()

    def camera_notify_exit(self, index: int) -> bool:
        value = self.camera_override(index).get("notify_exit")
        if isinstance(value, bool):
            return value
        return self.notify_exit

    def camera_enabled(self, index: int) -> bool:
        value = self.camera_override(index).get("enabled")
        return bool(value) if isinstance(value, bool) else True

    def camera_quiet(self, index: int, now: datetime | None = None) -> bool:
        value = self.camera_override(index).get("quiet_hours")
        spec = (str(value).strip() if value else "") or self.quiet_hours
        return in_quiet_hours(spec, now)

    def camera_detect_mode(self, index: int) -> str:
        value = self.camera_override(index).get("detect_mode")
        text = str(value or "")
        return text if text in {"presence", "identity"} else self.detect_mode

    def camera_label(self, index: int, source: str | None = None) -> str:
        """Человекочитаемая метка камеры для уведомлений и событий."""
        name = self.camera_name(index)
        if name:
            return name
        if source is None:
            sources = self.camera_sources()
            source = sources[index] if 0 <= index < len(sources) else ""
        return source or f"камера {index + 1}"

    @property
    def detector_backend(self) -> str:
        value = self._get_str("detector_backend", self._settings.detector_backend)
        return value if value in {"auto", "yolo", "yunet", "motion"} else self._settings.detector_backend

    @detector_backend.setter
    def detector_backend(self, value: str) -> None:
        if value not in {"auto", "yolo", "yunet", "motion"}:
            raise ConfigurationError("detector_backend: auto, yolo, yunet или motion")
        self._storage.set("detector_backend", value)

    @property
    def nms_iou(self) -> float:
        value = self._get_float("nms_iou", self._settings.nms_iou)
        return value if 0.0 <= value <= 1.0 else self._settings.nms_iou

    @nms_iou.setter
    def nms_iou(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ConfigurationError("nms_iou: значение должно быть в диапазоне 0..1")
        self._storage.set("nms_iou", str(value))

    @property
    def person_min_area(self) -> float:
        value = self._get_float("person_min_area", self._settings.person_min_area)
        return value if 0.0 <= value <= 0.5 else self._settings.person_min_area

    @person_min_area.setter
    def person_min_area(self, value: float) -> None:
        if not 0.0 <= value <= 0.5:
            raise ConfigurationError("person_min_area: значение должно быть в диапазоне 0..0.5")
        self._storage.set("person_min_area", str(value))

    @property
    def min_confidence(self) -> float:
        value = self._get_float("min_confidence", self._settings.min_confidence)
        return value if 0 <= value <= 1 else self._settings.min_confidence

    @min_confidence.setter
    def min_confidence(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ConfigurationError("min_confidence: значение должно быть в диапазоне 0..1")
        self._storage.set("min_confidence", str(value))

    @property
    def cooldown_seconds(self) -> float:
        value = self._get_float("cooldown_seconds", self._settings.cooldown_seconds)
        return value if 0 <= value <= 86400 else self._settings.cooldown_seconds

    @cooldown_seconds.setter
    def cooldown_seconds(self, value: float) -> None:
        if not 0 <= value <= 86400:
            raise ConfigurationError("cooldown_seconds: значение должно быть в диапазоне 0..86400")
        self._storage.set("cooldown_seconds", str(value))

    @property
    def detection_interval(self) -> float:
        value = self._get_float("detection_interval", self._settings.detection_interval)
        return value if 0.05 <= value <= 3600 else self._settings.detection_interval

    @detection_interval.setter
    def detection_interval(self, value: float) -> None:
        if not 0.05 <= value <= 3600:
            raise ConfigurationError("detection_interval: значение должно быть в диапазоне 0.05..3600")
        self._storage.set("detection_interval", str(value))

    @property
    def face_threshold(self) -> float:
        value = self._get_float("face_threshold", self._settings.face_threshold)
        return value if 0 < value <= 2 else self._settings.face_threshold

    @face_threshold.setter
    def face_threshold(self, value: float) -> None:
        if not 0 < value <= 2:
            raise ConfigurationError("face_threshold: значение должно быть в диапазоне (0..2]")
        self._storage.set("face_threshold", str(value))

    @property
    def model_path(self) -> str:
        return self._get_str("model_path", self._settings.model_path)

    @model_path.setter
    def model_path(self, value: str) -> None:
        self._storage.set("model_path", value)

    @property
    def device(self) -> str:
        value = self._get_str("device", self._settings.device)
        return value if value in {"cpu", "cuda", "mps"} else self._settings.device

    @device.setter
    def device(self, value: str) -> None:
        if value not in {"cpu", "cuda", "mps"}:
            raise ConfigurationError("device: cpu, cuda или mps")
        self._storage.set("device", value)

    @property
    def armed(self) -> bool:
        return self._get_bool("armed", self._settings.armed)

    @armed.setter
    def armed(self, value: bool) -> None:
        self._storage.set("armed", "1" if value else "0")

    @property
    def quiet_hours(self) -> str:
        value = self._get_str("quiet_hours", self._settings.quiet_hours)
        try:
            return validate_quiet_hours(value)
        except ConfigurationError:
            return self._settings.quiet_hours

    @quiet_hours.setter
    def quiet_hours(self, value: str) -> None:
        self._storage.set("quiet_hours", validate_quiet_hours(value))

    @property
    def notify_exit(self) -> bool:
        return self._get_bool("notify_exit", self._settings.notify_exit)

    @notify_exit.setter
    def notify_exit(self, value: bool) -> None:
        self._storage.set("notify_exit", "1" if value else "0")

    @property
    def mute_known_present(self) -> bool:
        """Не напоминать о человеке, который уже узнан и остаётся в кадре.

        Вход узнанного фиксируется один раз, а на повторные появления (после
        краткого сбоя трекинга или на других камерах) событие гасится, пока
        тот же человек физически не покинет кадр.
        """
        return self._get_bool("mute_known_present", self._settings.mute_known_present)

    @mute_known_present.setter
    def mute_known_present(self, value: bool) -> None:
        self._storage.set("mute_known_present", "1" if value else "0")

    @property
    def privacy_outbound(self) -> bool:
        """Отправлять в Telegram только обезличенные кадры."""
        return self._get_bool("privacy_outbound", self._settings.privacy_outbound)

    @privacy_outbound.setter
    def privacy_outbound(self, value: bool) -> None:
        self._storage.set("privacy_outbound", "1" if value else "0")

    @property
    def privacy_mode(self) -> str:
        """Как подготавливать кадры для Telegram.

        Старый флаг privacy_outbound остаётся совместимым: при его включении
        без явного режима применяется самый безопасный вариант silhouette.
        """
        value = self._get_str("privacy_mode", self._settings.privacy_mode).lower()
        if value in {"off", "person", "face", "silhouette", "all"}:
            return value
        return "silhouette" if self.privacy_outbound else "off"

    @privacy_mode.setter
    def privacy_mode(self, value: str) -> None:
        if value not in {"off", "person", "face", "silhouette", "all"}:
            raise ConfigurationError("privacy_mode: off, person, face, silhouette или all")
        self._storage.set("privacy_mode", value)
        self.privacy_outbound = value != "off"

    @property
    def privacy_blocks(self) -> int:
        value = self._get_int("privacy_blocks", self._settings.privacy_blocks)
        return value if 2 <= value <= 64 else 12

    @privacy_blocks.setter
    def privacy_blocks(self, value: int) -> None:
        if not 2 <= value <= 64:
            raise ConfigurationError("privacy_blocks: значение должно быть в диапазоне 2..64")
        self._storage.set("privacy_blocks", str(value))

    @property
    def remote_processing(self) -> bool:
        enabled = self._get_bool("remote_processing", self._settings.remote_processing)
        return enabled and bool(self.remote_url)

    @remote_processing.setter
    def remote_processing(self, value: bool) -> None:
        if value and not self.remote_url:
            raise ConfigurationError("remote_processing требует валидный remote_url")
        self._storage.set("remote_processing", "1" if value else "0")

    @property
    def remote_url(self) -> str:
        value = self._get_str("remote_url", self._settings.remote_url)
        if not value:
            return ""
        try:
            return validate_remote_url(
                value,
                allowed_hosts=self._settings.allowed_url_hosts,
                environment=self._settings.environment,
            )
        except ConfigurationError:
            return ""

    @remote_url.setter
    def remote_url(self, value: str) -> None:
        self._storage.set("remote_url", validate_remote_url(
            value,
            allowed_hosts=self._settings.allowed_url_hosts,
            environment=self._settings.environment,
        ))

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
    def remote_ca_cert(self) -> str:
        """PEM сертификат, которому доверяет клиент remote-инференса."""
        return self._settings.remote_ca_cert

    @property
    def imgsz(self) -> int:
        value = self._get_int("imgsz", self._settings.imgsz)
        return value if 160 <= value <= 4096 else self._settings.imgsz

    @imgsz.setter
    def imgsz(self, value: int) -> None:
        self._storage.set("imgsz", str(int(value)))

    @property
    def track_max_misses(self) -> int:
        value = self._get_int("track_max_misses", self._settings.track_max_misses)
        return value if 1 <= value <= 100 else self._settings.track_max_misses

    @property
    def zones_json(self) -> str:
        return self._get_str("zones", self._settings.zones)

    @zones_json.setter
    def zones_json(self, value: str) -> None:
        self._storage.set("zones", value)

    @property
    def events_max(self) -> int:
        value = self._get_int("events_max", self._settings.events_max)
        return value if 10 <= value <= 1_000_000 else self._settings.events_max

    @property
    def events_ttl_days(self) -> float:
        value = self._get_float("events_ttl_days", self._settings.events_ttl_days)
        return value if 0 <= value <= 3650 else self._settings.events_ttl_days

    def is_quiet(self, now: datetime | None = None) -> bool:
        return in_quiet_hours(self.quiet_hours, now)

    def should_notify(self, now: datetime | None = None) -> bool:
        return self.armed and not self.is_quiet(now)

    # -- голос и тревога ----------------------------------------------

    @property
    def voice_enabled(self) -> bool:
        return self._get_bool("voice_enabled", self._settings.voice_enabled)

    @voice_enabled.setter
    def voice_enabled(self, value: bool) -> None:
        self._storage.set("voice_enabled", "1" if value else "0")

    @property
    def voice_alarm_enabled(self) -> bool:
        return self._get_bool("voice_alarm_enabled", self._settings.voice_alarm_enabled)

    @voice_alarm_enabled.setter
    def voice_alarm_enabled(self, value: bool) -> None:
        self._storage.set("voice_alarm_enabled", "1" if value else "0")

    @property
    def voice_siren_enabled(self) -> bool:
        return self._get_bool("voice_siren_enabled", self._settings.voice_siren_enabled)

    @voice_siren_enabled.setter
    def voice_siren_enabled(self, value: bool) -> None:
        self._storage.set("voice_siren_enabled", "1" if value else "0")

    @property
    def voice_speech_enabled(self) -> bool:
        return self._get_bool("voice_speech_enabled", self._settings.voice_speech_enabled)

    @voice_speech_enabled.setter
    def voice_speech_enabled(self, value: bool) -> None:
        self._storage.set("voice_speech_enabled", "1" if value else "0")

    @property
    def voice_welcome(self) -> bool:
        return self._get_bool("voice_welcome", self._settings.voice_welcome)

    @voice_welcome.setter
    def voice_welcome(self, value: bool) -> None:
        self._storage.set("voice_welcome", "1" if value else "0")

    @property
    def voice_goodbye(self) -> bool:
        return self._get_bool("voice_goodbye", self._settings.voice_goodbye)

    @voice_goodbye.setter
    def voice_goodbye(self, value: bool) -> None:
        self._storage.set("voice_goodbye", "1" if value else "0")

    @property
    def voice_presence(self) -> bool:
        return self._get_bool("voice_presence", self._settings.voice_presence)

    @voice_presence.setter
    def voice_presence(self, value: bool) -> None:
        self._storage.set("voice_presence", "1" if value else "0")

    @property
    def voice_armed_announce(self) -> bool:
        return self._get_bool(
            "voice_armed_announce", self._settings.voice_armed_announce
        )

    @voice_armed_announce.setter
    def voice_armed_announce(self, value: bool) -> None:
        self._storage.set("voice_armed_announce", "1" if value else "0")

    @property
    def voice_alarm_on_presence(self) -> bool:
        return self._get_bool(
            "voice_alarm_on_presence", self._settings.voice_alarm_on_presence
        )

    @voice_alarm_on_presence.setter
    def voice_alarm_on_presence(self, value: bool) -> None:
        self._storage.set("voice_alarm_on_presence", "1" if value else "0")

    @property
    def voice_mute_quiet(self) -> bool:
        return self._get_bool("voice_mute_quiet", self._settings.voice_mute_quiet)

    @voice_mute_quiet.setter
    def voice_mute_quiet(self, value: bool) -> None:
        self._storage.set("voice_mute_quiet", "1" if value else "0")

    @property
    def voice_repeat_seconds(self) -> float:
        return max(2.0, self._get_float(
            "voice_repeat_seconds", self._settings.voice_repeat_seconds
        ))

    @voice_repeat_seconds.setter
    def voice_repeat_seconds(self, value: float) -> None:
        self._storage.set("voice_repeat_seconds", str(max(2.0, float(value))))

    @property
    def voice_timeout_seconds(self) -> float:
        return max(0.0, self._get_float(
            "voice_timeout_seconds", self._settings.voice_timeout_seconds
        ))

    @voice_timeout_seconds.setter
    def voice_timeout_seconds(self, value: float) -> None:
        self._storage.set("voice_timeout_seconds", str(max(0.0, float(value))))

    @property
    def voice_grace_seconds(self) -> float:
        return max(0.0, self._get_float(
            "voice_grace_seconds", self._settings.voice_grace_seconds
        ))

    @voice_grace_seconds.setter
    def voice_grace_seconds(self, value: float) -> None:
        self._storage.set("voice_grace_seconds", str(max(0.0, float(value))))

    @property
    def voice_clear_on(self) -> str:
        return self._get_str("voice_clear_on", self._settings.voice_clear_on)

    @voice_clear_on.setter
    def voice_clear_on(self, value: str) -> None:
        cleaned = (value or "").strip().lower()
        if cleaned not in ("both", "known", "exit"):
            raise ConfigurationError("voice_clear_on: both, known или exit")
        self._storage.set("voice_clear_on", cleaned)

    @property
    def voice_volume(self) -> float:
        return max(0.0, min(1.0, self._get_float(
            "voice_volume", self._settings.voice_volume
        )))

    @voice_volume.setter
    def voice_volume(self, value: float) -> None:
        self._storage.set("voice_volume", str(max(0.0, min(1.0, float(value)))))

    @property
    def voice_rate(self) -> float:
        return max(0.4, min(2.5, self._get_float(
            "voice_rate", self._settings.voice_rate
        )))

    @voice_rate.setter
    def voice_rate(self, value: float) -> None:
        self._storage.set("voice_rate", str(max(0.4, min(2.5, float(value)))))

    @property
    def voice_tts_voice(self) -> str:
        return self._get_str("voice_tts_voice", self._settings.voice_tts_voice)

    @voice_tts_voice.setter
    def voice_tts_voice(self, value: str) -> None:
        text = (value or "").strip()
        if len(text) > 256:
            raise ConfigurationError("voice_tts_voice: идентификатор длиннее 256 символов")
        self._storage.set("voice_tts_voice", text)

    @property
    def voice_alarm_tts_voice(self) -> str:
        return self._get_str("voice_alarm_tts_voice", self._settings.voice_alarm_tts_voice)

    @voice_alarm_tts_voice.setter
    def voice_alarm_tts_voice(self, value: str) -> None:
        self._set_voice_id("voice_alarm_tts_voice", value)

    @property
    def voice_welcome_tts_voice(self) -> str:
        return self._get_str("voice_welcome_tts_voice", self._settings.voice_welcome_tts_voice)

    @voice_welcome_tts_voice.setter
    def voice_welcome_tts_voice(self, value: str) -> None:
        self._set_voice_id("voice_welcome_tts_voice", value)

    def _set_voice_id(self, key: str, value: str) -> None:
        text = (value or "").strip()
        if len(text) > 256:
            raise ConfigurationError(f"{key}: идентификатор длиннее 256 символов")
        self._storage.set(key, text)

    @property
    def voice_cooldown_seconds(self) -> float:
        return max(0.0, self._get_float(
            "voice_cooldown_seconds", self._settings.voice_cooldown_seconds
        ))

    @voice_cooldown_seconds.setter
    def voice_cooldown_seconds(self, value: float) -> None:
        self._storage.set("voice_cooldown_seconds", str(max(0.0, float(value))))

    def voice_phrase(self, key: str) -> str:
        from doteye.voice import DEFAULT_PHRASES

        default = DEFAULT_PHRASES.get(key, "")
        return self._get_str(f"voice_phrase_{key}", default)

    def set_voice_phrase(self, key: str, value: str) -> None:
        from doteye.voice import DEFAULT_PHRASES

        if key not in DEFAULT_PHRASES:
            raise ConfigurationError(f"voice_phrase: неизвестный ключ {key}")
        text = " ".join((value or "").split())
        if not text or len(text) > 400:
            raise ConfigurationError("voice_phrase: текст пустой или длиннее 400 символов")
        self._storage.set(f"voice_phrase_{key}", text)

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
