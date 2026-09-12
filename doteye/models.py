"""Каталог YOLO-моделей с человеческими описаниями для админ-панели.

Модель скачивается ultralytics автоматически при первом запуске (по имени
файла). Здесь только метаданные для выбора в чате: размер, скорость,
точность и для чего модель подходит.

Данные приблизительные (порядок величин на CPU/GPU), нужны для выбора,
а не для бенчмарка.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str          # имя файла для ultralytics
    title: str         # подпись в панели
    size_mb: int       # примерный размер весов
    params_m: float    # миллионов параметров
    speed: str         # примерная скорость
    accuracy: str      # условная точность (mAP)
    best_for: str      # для чего подходит


# Порядок в списке - от лёгкой к тяжёлой.
YOLO_MODELS: dict[str, ModelInfo] = {
    "yolov8n.pt": ModelInfo(
        name="yolov8n.pt",
        title="YOLOv8n (nano)",
        size_mb=6,
        params_m=3.2,
        speed="~40-80 FPS на CPU",
        accuracy="базовая (mAP ~37)",
        best_for="слабое железо, Raspberry Pi, постоянный поток с одной камеры",
    ),
    "yolov8s.pt": ModelInfo(
        name="yolov8s.pt",
        title="YOLOv8s (small)",
        size_mb=22,
        params_m=11.2,
        speed="~15-30 FPS на CPU",
        accuracy="хорошая (mAP ~44)",
        best_for="баланс скорости и качества, обычный ноутбук без GPU",
    ),
    "yolov8m.pt": ModelInfo(
        name="yolov8m.pt",
        title="YOLOv8m (medium)",
        size_mb=50,
        params_m=25.9,
        speed="~5-10 FPS на CPU, 60+ на GPU",
        accuracy="высокая (mAP ~50)",
        best_for="ПК с GPU, точная детекция людей в сложных условиях",
    ),
    "yolov8l.pt": ModelInfo(
        name="yolov8l.pt",
        title="YOLOv8l (large)",
        size_mb=84,
        params_m=43.7,
        speed="~2-5 FPS на CPU, 40+ на GPU",
        accuracy="очень высокая (mAP ~53)",
        best_for="сервер/GPU, где важна точность, а не частота кадров",
    ),
    "yolov8x.pt": ModelInfo(
        name="yolov8x.pt",
        title="YOLOv8x (extra)",
        size_mb=131,
        params_m=68.2,
        speed="медленно на CPU, 30+ на GPU",
        accuracy="максимальная (mAP ~54)",
        best_for="максимальная точность на мощном GPU, редко для реального времени",
    ),
}

DEFAULT_MODEL = "yolov8n.pt"


def get_model(name: str) -> ModelInfo | None:
    return YOLO_MODELS.get(name)


def describe(name: str) -> str:
    """Текст для карточки модели в чате."""
    info = YOLO_MODELS.get(name)
    if info is None:
        return f"{name}: не из каталога, загрузится как есть."
    return (
        f"*{info.title}*\n"
        f"Размер: ~{info.size_mb} МБ ({info.params_m}M параметров)\n"
        f"Скорость: {info.speed}\n"
        f"Точность: {info.accuracy}\n"
        f"Когда брать: {info.best_for}"
    )


def comparison() -> str:
    """Общая таблица: чем мощнее модель, чем отличается."""
    lines = ["Модели, от лёгкой к тяжёлой:", ""]
    for info in YOLO_MODELS.values():
        lines.append(
            f"• {info.title}: {info.size_mb} МБ, {info.params_m}M параметров\n"
            f"   скорость {info.speed}, точность {info.accuracy}"
        )
    lines.append("")
    lines.append(
        "Правило простое: каждая следующая крупнее, точнее и медленнее. "
        "На CPU бери n/s, с GPU - m и выше. Смена модели перезапускает детектор."
    )
    return "\n".join(lines)
