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


@dataclass(frozen=True)
class Settings:
    # Telegram
    bot_token: str = field(default_factory=lambda: os.getenv("DOTEYE_BOT_TOKEN", ""))
    admin_ids: list[int] = field(
        default_factory=lambda: [
            int(x) for x in os.getenv("DOTEYE_ADMIN_IDS", "").split(",") if x.strip()
        ]
    )

    # Режимы обработки
    detect_mode: str = os.getenv("DOTEYE_DETECT_MODE", "presence")  # presence | identity

    # Камеры
    camera_source: str = os.getenv("DOTEYE_CAMERA_SOURCE", "0")  # 0 = вебка, либо RTSP/http

    # Детектор
    detector_backend: str = os.getenv("DOTEYE_DETECTOR", "auto")  # auto | yolo | hog
    remote_processing: bool = os.getenv("DOTEYE_REMOTE_PROCESSING", "0") == "1"
    remote_url: str = os.getenv("DOTEYE_REMOTE_URL", "")

    # Модели и данные
    model_path: str = os.getenv("DOTEYE_MODEL_PATH", "yolov8n.pt")
    data_dir: Path = Path(os.getenv("DOTEYE_DATA_DIR", BASE_DIR / "data"))
    db_path: Path = Path(os.getenv("DOTEYE_DB_PATH", BASE_DIR / "data" / "doteye.db"))

    # Инференс
    device: str = os.getenv("DOTEYE_DEVICE", "cpu")  # cpu | cuda | mps
    min_confidence: float = float(os.getenv("DOTEYE_MIN_CONFIDENCE", "0.5"))
    detection_interval: float = float(os.getenv("DOTEYE_DETECTION_INTERVAL", "1.0"))
    cooldown_seconds: float = float(os.getenv("DOTEYE_COOLDOWN_SECONDS", "30.0"))
    jpeg_quality: int = int(os.getenv("DOTEYE_JPEG_QUALITY", "85"))
    events_limit: int = int(os.getenv("DOTEYE_EVENTS_LIMIT", "10"))

    # Распознавание лиц (identity mode)
    face_threshold: float = float(os.getenv("DOTEYE_FACE_THRESHOLD", "0.4"))
    face_model: str = os.getenv("DOTEYE_FACE_MODEL", "")  # YuNet ONNX для детектора yunet

    # Безопасность
    crypto_key_env: str = os.getenv("DOTEYE_CRYPTO_KEY", "")  # 32-байтовый base64 ключ

    @property
    def has_token(self) -> bool:
        return bool(self.bot_token)

    @property
    def has_admins(self) -> bool:
        return bool(self.admin_ids)


def get_settings() -> Settings:
    return Settings()
