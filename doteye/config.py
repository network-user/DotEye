"""Конфигурация DotEye.

Один источник правды: значения из переменных окружения с дефолтами.
Токены и секреты читаются из .env через python-dotenv, в конфиг не пишутся.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default) == "1"


@dataclass(frozen=True)
class Settings:
    # Telegram
    bot_token: str = field(default_factory=lambda: os.getenv("DOTEYE_BOT_TOKEN", ""))
    admin_ids: list[int] = field(
        default_factory=lambda: [
            int(x) for x in os.getenv("DOTEYE_ADMIN_IDS", "").split(",") if x.strip()
        ]
    )
    # Пустой список админов по умолчанию никого не пускает. Для локальной
    # отладки: DOTEYE_ALLOW_OPEN_ACCESS=1
    allow_open_access: bool = _env_bool("DOTEYE_ALLOW_OPEN_ACCESS", "0")

    # Режимы обработки
    detect_mode: str = os.getenv("DOTEYE_DETECT_MODE", "presence")  # presence | identity

    # Камеры: один источник или несколько через |
    camera_source: str = os.getenv("DOTEYE_CAMERA_SOURCE", "0")

    # Детектор
    detector_backend: str = os.getenv("DOTEYE_DETECTOR", "auto")
    remote_processing: bool = _env_bool("DOTEYE_REMOTE_PROCESSING", "0")
    remote_url: str = os.getenv("DOTEYE_REMOTE_URL", "")
    remote_fallback: bool = _env_bool("DOTEYE_REMOTE_FALLBACK", "1")
    remote_insecure: bool = _env_bool("DOTEYE_REMOTE_INSECURE", "0")

    # Модели и данные
    model_path: str = os.getenv("DOTEYE_MODEL_PATH", "yolov8n.pt")
    data_dir: Path = Path(os.getenv("DOTEYE_DATA_DIR", BASE_DIR / "data"))
    db_path: Path = Path(os.getenv("DOTEYE_DB_PATH", BASE_DIR / "data" / "doteye.db"))

    # Инференс
    device: str = os.getenv("DOTEYE_DEVICE", "cpu")
    min_confidence: float = float(os.getenv("DOTEYE_MIN_CONFIDENCE", "0.5"))
    detection_interval: float = float(os.getenv("DOTEYE_DETECTION_INTERVAL", "1.0"))
    cooldown_seconds: float = float(os.getenv("DOTEYE_COOLDOWN_SECONDS", "30.0"))
    jpeg_quality: int = int(os.getenv("DOTEYE_JPEG_QUALITY", "85"))
    events_limit: int = int(os.getenv("DOTEYE_EVENTS_LIMIT", "10"))
    imgsz: int = int(os.getenv("DOTEYE_IMGSZ", "640"))
    track_max_misses: int = int(os.getenv("DOTEYE_TRACK_MAX_MISSES", "3"))

    # Охрана и тихие часы (локальное время, формат HH:MM-HH:MM)
    armed: bool = _env_bool("DOTEYE_ARMED", "1")
    quiet_hours: str = os.getenv("DOTEYE_QUIET_HOURS", "")
    notify_exit: bool = _env_bool("DOTEYE_NOTIFY_EXIT", "1")

    # Хранение событий
    events_max: int = int(os.getenv("DOTEYE_EVENTS_MAX", "500"))
    events_ttl_days: float = float(os.getenv("DOTEYE_EVENTS_TTL_DAYS", "14"))

    # Зоны кадра: JSON [{"name":"дверь","x1":0,"y1":0,"x2":0.5,"y2":1}]
    zones: str = os.getenv("DOTEYE_ZONES", "")

    # Распознавание лиц (identity mode)
    face_threshold: float = float(os.getenv("DOTEYE_FACE_THRESHOLD", "0.4"))
    face_model: str = os.getenv("DOTEYE_FACE_MODEL", "")

    # Безопасность
    crypto_key_env: str = os.getenv("DOTEYE_CRYPTO_KEY", "")

    @property
    def has_token(self) -> bool:
        return bool(self.bot_token)

    @property
    def has_admins(self) -> bool:
        return bool(self.admin_ids)


def get_settings() -> Settings:
    return Settings()
