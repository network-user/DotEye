"""Конфигурация DotEye.

Один источник правды: значения из переменных окружения с дефолтами.
Токены и секреты читаются из .env через python-dotenv, в конфиг не пишутся.
"""

from __future__ import annotations

import os
import ipaddress
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default) == "1"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} должен быть целым числом") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} должен быть числом") from exc


class ConfigurationError(ValueError):
    """Настройка небезопасна или не соответствует поддерживаемому формату."""


_DEVELOPMENT_ENVIRONMENTS = {"development", "dev", "test"}
_CAMERA_SCHEMES = {"rtsp", "rtsps", "http", "https"}
_REMOTE_SCHEMES = {"https", "http"}
_MAX_URL_LENGTH = 2048
_MAX_CAMERA_SOURCES = 4


def parse_allowed_hosts(value: str) -> frozenset[str]:
    """Вернуть точный allowlist хостов без wildcard-подстановок."""
    hosts = {item.strip().casefold().rstrip(".") for item in value.split(",") if item.strip()}
    for host in hosts:
        if "/" in host or "://" in host or any(char.isspace() for char in host):
            raise ConfigurationError("DOTEYE_ALLOWED_URL_HOSTS содержит некорректный хост")
    return frozenset(hosts)


def _is_loopback_host(host: str) -> bool:
    if host.casefold().rstrip(".") == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_url(
    value: str,
    *,
    allowed_hosts: frozenset[str],
    purpose: str,
    environment: str,
) -> str:
    """Проверить URL сетевого источника до записи в runtime-хранилище."""
    text = value.strip()
    if not text or len(text) > _MAX_URL_LENGTH:
        raise ConfigurationError(f"{purpose}: URL пустой или превышает {_MAX_URL_LENGTH} символов")
    parsed = urlsplit(text)
    schemes = _CAMERA_SCHEMES if purpose == "camera" else _REMOTE_SCHEMES
    scheme = parsed.scheme.casefold()
    if scheme not in schemes or not parsed.hostname:
        expected = ", ".join(sorted(schemes))
        raise ConfigurationError(f"{purpose}: допустимые схемы: {expected}")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError(f"{purpose}: credentials в URL запрещены")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError(f"{purpose}: недопустимый порт") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ConfigurationError(f"{purpose}: недопустимый порт")
    host = parsed.hostname.casefold().rstrip(".")
    if host not in allowed_hosts and not _is_loopback_host(host):
        raise ConfigurationError(f"{purpose}: хост отсутствует в DOTEYE_ALLOWED_URL_HOSTS")
    if purpose == "remote" and scheme == "http" and not (
        environment in _DEVELOPMENT_ENVIRONMENTS or _is_loopback_host(host)
    ):
        raise ConfigurationError("remote: HTTP разрешён только для loopback или development")
    return text


def validate_camera_source(value: str, *, allowed_hosts: frozenset[str], environment: str) -> str:
    """Проверить список камер: индекс устройства или allowlisted URL."""
    text = value.strip()
    if not text:
        return "0"
    if len(text) > _MAX_URL_LENGTH * _MAX_CAMERA_SOURCES:
        raise ConfigurationError("camera: список источников слишком длинный")
    sources = [item.strip() for item in text.split("|") if item.strip()]
    if not sources or len(sources) > _MAX_CAMERA_SOURCES:
        raise ConfigurationError(f"camera: допустимо от 1 до {_MAX_CAMERA_SOURCES} источников")
    normalized: list[str] = []
    for source in sources:
        if source.isdecimal():
            if int(source) > 99:
                raise ConfigurationError("camera: индекс устройства должен быть от 0 до 99")
            normalized.append(str(int(source)))
        else:
            normalized.append(validate_url(
                source, allowed_hosts=allowed_hosts, purpose="camera", environment=environment,
            ))
    return "|".join(normalized)


def validate_remote_url(value: str, *, allowed_hosts: frozenset[str], environment: str) -> str:
    return validate_url(value, allowed_hosts=allowed_hosts, purpose="remote", environment=environment)


def validate_quiet_hours(value: str) -> str:
    text = value.strip()
    if not text or text.casefold() in {"0", "off", "выкл", "-"}:
        return ""
    if "-" not in text:
        raise ConfigurationError("quiet_hours: ожидается формат HH:MM-HH:MM")
    start, end = text.split("-", 1)
    try:
        hours = [int(part) for item in (start, end) for part in item.split(":")]
    except ValueError as exc:
        raise ConfigurationError("quiet_hours: ожидается формат HH:MM-HH:MM") from exc
    if len(hours) != 4 or not (0 <= hours[0] <= 23 and 0 <= hours[2] <= 23 and 0 <= hours[1] <= 59 and 0 <= hours[3] <= 59):
        raise ConfigurationError("quiet_hours: время вне диапазона")
    return f"{hours[0]:02d}:{hours[1]:02d}-{hours[2]:02d}:{hours[3]:02d}"


@dataclass(frozen=True)
class Settings:
    environment: str = field(default_factory=lambda: os.getenv("DOTEYE_ENV", "production").casefold())
    allowed_url_hosts: frozenset[str] = field(
        default_factory=lambda: parse_allowed_hosts(os.getenv("DOTEYE_ALLOWED_URL_HOSTS", ""))
    )
    # Telegram
    bot_token: str = field(default_factory=lambda: os.getenv("DOTEYE_BOT_TOKEN", ""))
    admin_ids: list[int] = field(
        default_factory=lambda: [
            int(x) for x in os.getenv("DOTEYE_ADMIN_IDS", "").split(",") if x.strip()
        ]
    )
    # Пустой список админов по умолчанию никого не пускает. Для локальной
    # отладки: DOTEYE_ALLOW_OPEN_ACCESS=1
    allow_open_access: bool = field(default_factory=lambda: _env_bool("DOTEYE_ALLOW_OPEN_ACCESS", "0"))

    # Режимы обработки
    detect_mode: str = field(default_factory=lambda: os.getenv("DOTEYE_DETECT_MODE", "presence"))

    # Камеры: один источник или несколько через |
    camera_source: str = field(default_factory=lambda: os.getenv("DOTEYE_CAMERA_SOURCE", "0"))

    # Детектор
    detector_backend: str = field(default_factory=lambda: os.getenv("DOTEYE_DETECTOR", "auto"))
    remote_processing: bool = field(default_factory=lambda: _env_bool("DOTEYE_REMOTE_PROCESSING", "0"))
    remote_url: str = field(default_factory=lambda: os.getenv("DOTEYE_REMOTE_URL", ""))
    remote_fallback: bool = field(default_factory=lambda: _env_bool("DOTEYE_REMOTE_FALLBACK", "1"))
    remote_insecure: bool = field(default_factory=lambda: _env_bool("DOTEYE_REMOTE_INSECURE", "0"))

    # Модели и данные
    model_path: str = field(default_factory=lambda: os.getenv("DOTEYE_MODEL_PATH", "yolov8n.pt"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DOTEYE_DATA_DIR", BASE_DIR / "data")))
    db_path: Path | None = field(
        default_factory=lambda: Path(os.environ["DOTEYE_DB_PATH"])
        if os.getenv("DOTEYE_DB_PATH") else None
    )

    # Инференс
    device: str = field(default_factory=lambda: os.getenv("DOTEYE_DEVICE", "cpu"))
    min_confidence: float = field(default_factory=lambda: _env_float("DOTEYE_MIN_CONFIDENCE", 0.5))
    detection_interval: float = field(default_factory=lambda: _env_float("DOTEYE_DETECTION_INTERVAL", 1.0))
    cooldown_seconds: float = field(default_factory=lambda: _env_float("DOTEYE_COOLDOWN_SECONDS", 30.0))
    jpeg_quality: int = field(default_factory=lambda: _env_int("DOTEYE_JPEG_QUALITY", 85))
    privacy_outbound: bool = field(
        default_factory=lambda: _env_bool("DOTEYE_PRIVACY_OUTBOUND", "1")
    )
    events_limit: int = field(default_factory=lambda: _env_int("DOTEYE_EVENTS_LIMIT", 10))
    imgsz: int = field(default_factory=lambda: _env_int("DOTEYE_IMGSZ", 640))
    track_max_misses: int = field(default_factory=lambda: _env_int("DOTEYE_TRACK_MAX_MISSES", 3))

    # Охрана и тихие часы (локальное время, формат HH:MM-HH:MM)
    armed: bool = field(default_factory=lambda: _env_bool("DOTEYE_ARMED", "1"))
    quiet_hours: str = field(default_factory=lambda: os.getenv("DOTEYE_QUIET_HOURS", ""))
    notify_exit: bool = field(default_factory=lambda: _env_bool("DOTEYE_NOTIFY_EXIT", "1"))

    # Хранение событий
    events_max: int = field(default_factory=lambda: _env_int("DOTEYE_EVENTS_MAX", 200))
    events_ttl_days: float = field(default_factory=lambda: _env_float("DOTEYE_EVENTS_TTL_DAYS", 7))

    # Зоны кадра: JSON [{"name":"дверь","x1":0,"y1":0,"x2":0.5,"y2":1}]
    zones: str = field(default_factory=lambda: os.getenv("DOTEYE_ZONES", ""))

    # Распознавание лиц (identity mode)
    face_threshold: float = field(default_factory=lambda: _env_float("DOTEYE_FACE_THRESHOLD", 0.4))
    face_model: str = field(default_factory=lambda: os.getenv("DOTEYE_FACE_MODEL", ""))

    # Голос и тревога (локальные динамики)
    voice_enabled: bool = field(default_factory=lambda: _env_bool("DOTEYE_VOICE", "1"))
    voice_alarm_enabled: bool = field(default_factory=lambda: _env_bool("DOTEYE_VOICE_ALARM", "1"))
    voice_siren_enabled: bool = field(default_factory=lambda: _env_bool("DOTEYE_VOICE_SIREN", "1"))
    voice_speech_enabled: bool = field(default_factory=lambda: _env_bool("DOTEYE_VOICE_SPEECH", "1"))
    voice_welcome: bool = field(default_factory=lambda: _env_bool("DOTEYE_VOICE_WELCOME", "1"))
    voice_goodbye: bool = field(default_factory=lambda: _env_bool("DOTEYE_VOICE_GOODBYE", "0"))
    voice_presence: bool = field(default_factory=lambda: _env_bool("DOTEYE_VOICE_PRESENCE", "1"))
    voice_armed_announce: bool = field(default_factory=lambda: _env_bool("DOTEYE_VOICE_ARMED_ANNOUNCE", "1"))
    voice_alarm_on_presence: bool = field(
        default_factory=lambda: _env_bool("DOTEYE_VOICE_ALARM_ON_PRESENCE", "0")
    )
    voice_mute_quiet: bool = field(default_factory=lambda: _env_bool("DOTEYE_VOICE_MUTE_QUIET", "1"))
    voice_repeat_seconds: float = field(
        default_factory=lambda: _env_float("DOTEYE_VOICE_REPEAT_SECONDS", 8.0)
    )
    voice_timeout_seconds: float = field(
        default_factory=lambda: _env_float("DOTEYE_VOICE_TIMEOUT_SECONDS", 0.0)
    )
    voice_grace_seconds: float = field(
        default_factory=lambda: _env_float("DOTEYE_VOICE_GRACE_SECONDS", 4.0)
    )
    voice_clear_on: str = field(default_factory=lambda: os.getenv("DOTEYE_VOICE_CLEAR_ON", "both"))
    voice_volume: float = field(default_factory=lambda: _env_float("DOTEYE_VOICE_VOLUME", 0.8))
    voice_rate: float = field(default_factory=lambda: _env_float("DOTEYE_VOICE_RATE", 1.0))
    voice_tts_voice: str = field(default_factory=lambda: os.getenv("DOTEYE_VOICE_TTS_VOICE", ""))
    voice_alarm_tts_voice: str = field(
        default_factory=lambda: os.getenv("DOTEYE_VOICE_ALARM_TTS_VOICE", "")
    )
    voice_welcome_tts_voice: str = field(
        default_factory=lambda: os.getenv("DOTEYE_VOICE_WELCOME_TTS_VOICE", "")
    )
    voice_cooldown_seconds: float = field(
        default_factory=lambda: _env_float("DOTEYE_VOICE_COOLDOWN_SECONDS", 20.0)
    )

    # Безопасность
    crypto_key_env: str = field(default_factory=lambda: os.getenv("DOTEYE_CRYPTO_KEY", ""))

    def __post_init__(self) -> None:
        if self.db_path is None:
            object.__setattr__(self, "db_path", self.data_dir / "doteye.db")
        if self.environment not in {"production", "staging", *_DEVELOPMENT_ENVIRONMENTS}:
            raise ConfigurationError("DOTEYE_ENV: production, staging, development, dev или test")
        if self.allow_open_access and self.environment not in _DEVELOPMENT_ENVIRONMENTS:
            raise ConfigurationError("DOTEYE_ALLOW_OPEN_ACCESS разрешён только при DOTEYE_ENV=development")
        if self.detect_mode not in {"presence", "identity"}:
            raise ConfigurationError("DOTEYE_DETECT_MODE: presence или identity")
        if self.detector_backend not in {"auto", "yolo", "yunet", "motion"}:
            raise ConfigurationError("DOTEYE_DETECTOR: auto, yolo, yunet или motion")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ConfigurationError("DOTEYE_DEVICE: cpu, cuda или mps")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigurationError("DOTEYE_MIN_CONFIDENCE должен быть в диапазоне 0..1")
        if not 0.05 <= self.detection_interval <= 3600:
            raise ConfigurationError("DOTEYE_DETECTION_INTERVAL должен быть в диапазоне 0.05..3600")
        if not 0 <= self.cooldown_seconds <= 86400:
            raise ConfigurationError("DOTEYE_COOLDOWN_SECONDS должен быть в диапазоне 0..86400")
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigurationError("DOTEYE_JPEG_QUALITY должен быть в диапазоне 1..100")
        if not 1 <= self.events_limit <= 100:
            raise ConfigurationError("DOTEYE_EVENTS_LIMIT должен быть в диапазоне 1..100")
        if not 160 <= self.imgsz <= 4096:
            raise ConfigurationError("DOTEYE_IMGSZ должен быть в диапазоне 160..4096")
        if not 1 <= self.track_max_misses <= 100:
            raise ConfigurationError("DOTEYE_TRACK_MAX_MISSES должен быть в диапазоне 1..100")
        if not 10 <= self.events_max <= 1_000_000 or not 0 <= self.events_ttl_days <= 3650:
            raise ConfigurationError("лимиты хранения событий вне безопасного диапазона")
        if not 0 < self.face_threshold <= 2:
            raise ConfigurationError("DOTEYE_FACE_THRESHOLD должен быть в диапазоне (0..2]")
        if self.voice_clear_on not in {"both", "known", "exit"}:
            raise ConfigurationError("DOTEYE_VOICE_CLEAR_ON: both, known или exit")
        if not 2 <= self.voice_repeat_seconds <= 3600:
            raise ConfigurationError("DOTEYE_VOICE_REPEAT_SECONDS должен быть в диапазоне 2..3600")
        if not 0 <= self.voice_timeout_seconds <= 86400:
            raise ConfigurationError("DOTEYE_VOICE_TIMEOUT_SECONDS должен быть в диапазоне 0..86400")
        if not 0 <= self.voice_grace_seconds <= 3600:
            raise ConfigurationError("DOTEYE_VOICE_GRACE_SECONDS должен быть в диапазоне 0..3600")
        if not 0 <= self.voice_volume <= 1:
            raise ConfigurationError("DOTEYE_VOICE_VOLUME должен быть в диапазоне 0..1")
        if not 0.4 <= self.voice_rate <= 2.5:
            raise ConfigurationError("DOTEYE_VOICE_RATE должен быть в диапазоне 0.4..2.5")
        if not 0 <= self.voice_cooldown_seconds <= 86400:
            raise ConfigurationError("DOTEYE_VOICE_COOLDOWN_SECONDS должен быть в диапазоне 0..86400")
        object.__setattr__(self, "camera_source", validate_camera_source(
            self.camera_source, allowed_hosts=self.allowed_url_hosts, environment=self.environment,
        ))
        object.__setattr__(self, "quiet_hours", validate_quiet_hours(self.quiet_hours))
        if self.remote_url:
            object.__setattr__(self, "remote_url", validate_remote_url(
                self.remote_url, allowed_hosts=self.allowed_url_hosts, environment=self.environment,
            ))
        if self.remote_processing and not self.remote_url:
            raise ConfigurationError("DOTEYE_REMOTE_PROCESSING требует DOTEYE_REMOTE_URL")
        if self.remote_insecure and self.environment not in _DEVELOPMENT_ENVIRONMENTS:
            raise ConfigurationError("DOTEYE_REMOTE_INSECURE разрешён только при DOTEYE_ENV=development")

    @property
    def has_token(self) -> bool:
        return bool(self.bot_token)

    @property
    def has_admins(self) -> bool:
        return bool(self.admin_ids)


def get_settings() -> Settings:
    return Settings()
