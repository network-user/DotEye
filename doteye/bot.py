"""Telegram-бот на aiogram 3.

Настройка целиком внутри чата: охрана, тихие часы, камеры, зоны,
списки людей и событий, «это кто?» на неизвестном входе.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

import cv2
import numpy as np
from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    TelegramObject,
)

from doteye import models
from doteye.annotate import crop_box, encode_jpeg, privacy_frame
from doteye.config import ConfigurationError, Settings
from doteye.crypto import Crypto
from doteye.pipeline import Pipeline
from doteye.recognizer import Recognizer
from doteye.runtime import Runtime
from doteye.storage import Storage
from doteye.voice import CLEAR_ON_VALUES, DEFAULT_PHRASES, PHRASE_TITLES, VoiceEngine
from doteye.zones import Zone, dump_zones, parse_zones

HELP = (
    "DotEye - кто зашёл в комнату.\n\n"
    "Управление\n"
    "/panel - админ-панель (кнопки)\n"
    "/status - текущие настройки\n\n"
    "Видео и детекция\n"
    "/camera - источник кадров (несколько через | )\n"
    "/detector - auto | yolo | yunet | motion\n"
    "/model - выбрать YOLO-модель\n"
    "/device - cpu | cuda | mps\n"
    "/mode - переключить: «Обнаружение людей» / «Распознавание лиц»\n"
    "/confidence - порог детекции (0..1)\n"
    "/cooldown - пауза повторного входа, сек\n\n"
    "Люди\n"
    "/people - известные люди\n"
    "/add - добавить человека (имя + фото)\n"
    "/remove - удалить человека\n\n"
    "События и голос\n"
    "/events - последние события\n"
    "/audit - журнал действий без содержимого сообщений\n"
    "/say - сказать вслух через динамики\n"
    "/alarm - ручная тревога (/alarm off - снять)\n\n"
    "Прочее\n"
    "/help - эта справка\n"
    "/cancel - отменить текущий ввод"
)

DETECTORS = ["auto", "yolo", "yunet", "motion"]
DEVICES = ["cpu", "cuda", "mps"]
EVENTS_PAGE = 3
PEOPLE_PAGE = 6
OUTBOX_POLL_SECONDS = 1.0
OUTBOX_BATCH_SIZE = 20
OUTBOX_MAX_RETRY_SECONDS = 3600
TEST_SNAPSHOT_TTL_SECONDS = 60


class CameraForm(StatesGroup):
    source = State()
    name = State()


class AddPersonForm(StatesGroup):
    name = State()
    photo = State()


class RemovePersonForm(StatesGroup):
    name = State()


class NumberForm(StatesGroup):
    confidence = State()
    cooldown = State()
    value = State()


class TextForm(StatesGroup):
    value = State()


class PhotoWaitForm(StatesGroup):
    photo = State()


def _is_admin(message: Message, settings: Settings) -> bool:
    if not settings.has_admins:
        return bool(settings.allow_open_access)
    return message.from_user is not None and message.from_user.id in settings.admin_ids


def _is_admin_user(user_id: int | None, settings: Settings) -> bool:
    if not settings.has_admins:
        return bool(settings.allow_open_access)
    return user_id is not None and user_id in settings.admin_ids


def _face(recognizer: Recognizer | None, pipeline: Pipeline | None) -> Recognizer | None:
    if pipeline is not None and getattr(pipeline, "_recognizer", None) is not None:
        return pipeline._recognizer
    return recognizer


def _next_item(items: list[str], current: str) -> str:
    idx = items.index(current) if current in items else 0
    return items[(idx + 1) % len(items)]


def _cancel_requested(message: Message) -> bool:
    text = (message.text or "").strip().casefold()
    return text in ("отмена", "/cancel")


def _back_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Назад", callback_data="panel:open")]
    ])


def _main_menu() -> ReplyKeyboardMarkup:
    """Постоянная навигация Telegram для частых действий."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Панель"), KeyboardButton(text="Статус")],
            [KeyboardButton(text="События"), KeyboardButton(text="Люди")],
            [KeyboardButton(text="Голос и тревога"), KeyboardButton(text="Превью")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Выберите действие",
    )


def _camera_title(source: str) -> str:
    """Короткое безопасное название источника для кнопок и панели."""
    if source.isdecimal():
        return f"Локальная камера #{source}"
    parsed = urlsplit(source)
    host = parsed.hostname or "сетевой источник"
    scheme = parsed.scheme.upper() or "URL"
    return f"{scheme}: {host}"


def _camera_source_text(source: str) -> str:
    """Источник без учётных данных и query-параметров."""
    if source.isdecimal():
        return f"устройство #{source}"
    parsed = urlsplit(source)
    if not parsed.scheme:
        return source
    host = parsed.hostname or ""
    try:
        port_value = parsed.port
    except ValueError:
        port_value = None
    port = f":{port_value}" if port_value else ""
    path = parsed.path or "/"
    return f"{parsed.scheme}://{host}{port}{path}"


def _camera_keyboard(runtime: Runtime) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for index, source in enumerate(runtime.camera_sources(), start=0):
        label = runtime.camera_label(index, source)
        rows.append([InlineKeyboardButton(
            text=f"Камера {index + 1}: {label}",
            callback_data=f"camera:view:{index}",
        )])
    rows.extend([
        [InlineKeyboardButton(text="Изменить список камер", callback_data="camera:edit")],
        [InlineKeyboardButton(text="Инструкция и помощь", callback_data="camera:help")],
        [InlineKeyboardButton(text="◀ Назад", callback_data="panel:open")],
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _camera_info_text(
    source: str, index: int, pipeline: Pipeline | None, runtime: Runtime,
) -> str:
    info = pipeline.camera_info(source) if pipeline is not None else None
    if pipeline is None:
        status = "проверка недоступна: пайплайн выключен"
        frame = "нет"
        reconnects = 0
        error = None
    elif info is None:
        status = "ожидает применения настройки"
        frame = "нет"
        reconnects = 0
        error = None
    else:
        status = "доступна" if info["healthy"] else "нет связи или кадра"
        frame = "есть" if info["has_frame"] else "ещё нет"
        reconnects = int(info["reconnects"])
        error = info["last_error"]
    lines = [
        f"Камера {index + 1}: {runtime.camera_label(index, source)}",
        f"Тип: {_camera_title(source)}",
        f"Источник: {_camera_source_text(source)}",
        f"Статус: {status}",
        f"Последний кадр: {frame}",
    ]
    if reconnects:
        lines.append(f"Переподключений: {reconnects}")
    if error:
        lines.append(f"Ошибка: {error}")
    if not runtime.camera_enabled(index):
        lines.append("Состояние: выключена")
    cooldown = runtime.camera_cooldown(index)
    lines.append(f"Кулдаун: {cooldown:g} сек")
    lines.append(
        "Уведомления: вход "
        + ("вкл" if runtime.camera_notify_enter(index) else "выкл")
        + ", выход " + ("вкл" if runtime.camera_notify_exit(index) else "выкл")
    )
    quiet = runtime.camera_quiet(index)
    if quiet:
        lines.append("Сейчас тихие часы: уведомления приостановлены")
    mode = runtime.camera_detect_mode(index)
    lines.append(f"Режим: {'распознавание лиц' if mode == 'identity' else 'обнаружение людей'}")
    return "\n".join(lines)


def _camera_detail_keyboard(index: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Тестовый снимок", callback_data=f"camera:test:{index}")],
        [InlineKeyboardButton(text="Настройки камеры", callback_data=f"camera:conf:{index}")],
        [InlineKeyboardButton(text="Переименовать", callback_data=f"camera:rename:{index}")],
        [InlineKeyboardButton(text="Изменить список", callback_data="camera:edit")],
        [InlineKeyboardButton(text="Если не работает", callback_data="camera:help")],
        [InlineKeyboardButton(text="◀ К списку камер", callback_data="panel:camera")],
    ])


def _camera_conf_text(index: int, source: str, runtime: Runtime) -> str:
    enabled = runtime.camera_enabled(index)
    return (
        f"Камера {index + 1}: {runtime.camera_label(index, source)}\n\n"
        f"Обработка: {'включена' if enabled else 'выключена'}\n"
        f"Кулдаун: {runtime.camera_cooldown(index):g} сек\n"
        f"Уведомлять о входе: {'да' if runtime.camera_notify_enter(index) else 'нет'}\n"
        f"Уведомлять о выходе: {'да' if runtime.camera_notify_exit(index) else 'нет'}\n"
        f"Свои тихие часы: {runtime.camera_override(index).get('quiet_hours') or 'общие'}\n"
        f"Режим: {runtime.camera_detect_mode(index)}"
    )


def _camera_conf_keyboard(index: int, runtime: Runtime) -> InlineKeyboardMarkup:
    enabled = runtime.camera_enabled(index)
    notify_enter = runtime.camera_notify_enter(index)
    notify_exit = runtime.camera_notify_exit(index)
    mode = runtime.camera_detect_mode(index)
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text=f"Обработка: {'вкл' if enabled else 'выкл'}",
            callback_data=f"camera:tog:{index}:enabled",
        )],
        [InlineKeyboardButton(text="Кулдаун", callback_data=f"camera:num:{index}:cooldown")],
        [InlineKeyboardButton(
            text=f"Вход: {'вкл' if notify_enter else 'выкл'}",
            callback_data=f"camera:tog:{index}:notify_enter",
        )],
        [InlineKeyboardButton(
            text=f"Выход: {'вкл' if notify_exit else 'выкл'}",
            callback_data=f"camera:tog:{index}:notify_exit",
        )],
        [InlineKeyboardButton(
            text=f"Режим: {mode}",
            callback_data=f"camera:mode:{index}",
        )],
        [InlineKeyboardButton(text="Сбросить настройки", callback_data=f"camera:reset:{index}")],
        [InlineKeyboardButton(text="◀ К камере", callback_data=f"camera:view:{index}")],
    ])


async def _delete_test_snapshot(message: Message) -> None:
    await asyncio.sleep(TEST_SNAPSHOT_TTL_SECONDS)
    try:
        await message.delete()
    except Exception:
        pass


class AccessMiddleware(BaseMiddleware):
    """Пускает дальше только админов и прокидывает зависимости в хендлеры."""

    def __init__(self, settings: Settings, storage: Storage, runtime: Runtime,
                 crypto: Crypto | None, recognizer: Recognizer | None,
                 pipeline: Pipeline | None, voice: VoiceEngine | None = None) -> None:
        self._settings = settings
        self._storage = storage
        self._runtime = runtime
        self._crypto = crypto
        self._recognizer = recognizer
        self._pipeline = pipeline
        self._voice = voice

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        data["settings"] = self._settings
        data["storage"] = self._storage
        data["runtime"] = self._runtime
        data["crypto"] = self._crypto
        data["recognizer"] = self._recognizer
        data["pipeline"] = self._pipeline
        data["voice"] = self._voice

        if isinstance(event, Message) and not _is_admin(event, self._settings):
            await event.answer("Нет доступа.")
            return None
        if isinstance(event, CallbackQuery) and not _is_admin_user(
            event.from_user.id if event.from_user else None, self._settings
        ):
            await event.answer("Нет доступа.", show_alert=True)
            return None
        self._write_audit(event)
        return await handler(event, data)

    def _write_audit(self, event: TelegramObject) -> None:
        """Пишет только тип операции, никогда текст, имя, id или изображение."""
        if self._crypto is None:
            return
        if isinstance(event, CallbackQuery):
            parts = (event.data or "").split(":")
            label = "callback:" + ":".join(parts[:2])
        elif isinstance(event, Message):
            text = (event.text or "").strip()
            label = "command:" + text.split(maxsplit=1)[0] if text.startswith("/") else "message_input"
            if getattr(event, "photo", None) or getattr(event, "document", None):
                label = "image_input"
        else:
            return
        try:
            self._storage.add_audit_entry(self._crypto.encrypt(label.encode("utf-8")))
        except Exception as exc:  # noqa: BLE001
            print(f"[audit] write: {exc}")


router = Router()


def _on(flag: bool) -> str:
    return "вкл" if flag else "выкл"


_MODE_LABELS = {
    "presence": "Обнаружение людей",
    "identity": "Распознавание лиц",
}
_MODE_HINTS = {
    "presence": "замечает любого человека, но не определяет, кто это",
    "identity": "сравнивает лицо с вашим списком людей",
}


def _mode_text(mode: str) -> str:
    label = _MODE_LABELS.get(mode, mode)
    hint = _MODE_HINTS.get(mode, "")
    return f"{label} ({mode}) - {hint}" if hint else label


def _ico(flag: bool) -> str:
    """Эмодзи-индикатор вкл/выкл для быстрого считывания."""
    return "🟢" if flag else "⚪"


def _short_phrase(text: str, limit: int = 36) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _panel_keyboard(runtime: Runtime, voice: VoiceEngine | None = None) -> InlineKeyboardMarkup:
    mode_next = "identity" if runtime.detect_mode == "presence" else "presence"
    if voice is not None and voice.alarming:
        voice_s = "ТРЕВОГА"
        voice_ico = "●"
    else:
        voice_s = "голос"
        voice_ico = _ico(runtime.voice_enabled)
    rows = [
        [
            InlineKeyboardButton(
                text=f"{_ico(runtime.armed)} Охрана {_on(runtime.armed)}",
                callback_data="panel:arm",
            ),
            InlineKeyboardButton(
                text=f"Режим: {_MODE_LABELS[runtime.detect_mode]}",
                callback_data="panel:mode",
            ),
        ],
        [
            InlineKeyboardButton(text="События", callback_data="panel:events"),
            InlineKeyboardButton(text="Люди", callback_data="panel:people"),
        ],
        [
            InlineKeyboardButton(text="Превью", callback_data="panel:snapshot"),
            InlineKeyboardButton(text="Камера", callback_data="panel:camera"),
        ],
        [
            InlineKeyboardButton(
                text=f"{voice_ico} {voice_s}", callback_data="voice:open"
            ),
            InlineKeyboardButton(text="Здоровье", callback_data="panel:health"),
        ],
        [
            InlineKeyboardButton(text="Тонкая настройка", callback_data="panel:tune"),
            InlineKeyboardButton(text="Статус", callback_data="panel:status"),
        ],
        [
            InlineKeyboardButton(text="Справка", callback_data="panel:help"),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _tune_keyboard(runtime: Runtime) -> InlineKeyboardMarkup:
    det_next = _next_item(DETECTORS, runtime.detector_backend)
    dev_next = _next_item(DEVICES, runtime.device)
    quiet = runtime.quiet_hours or "выкл"
    remote = _on(runtime.remote_processing)
    privacy = _PRIVACY_LABELS[runtime.privacy_mode]
    exit_s = _on(runtime.notify_exit)
    rows = [
        [
            InlineKeyboardButton(
                text=f"Детектор: {runtime.detector_backend} → {det_next}",
                callback_data="panel:detector",
            ),
            InlineKeyboardButton(
                text=f"Устройство: {runtime.device} → {dev_next}",
                callback_data="panel:device",
            ),
        ],
        [
            InlineKeyboardButton(text="Модель", callback_data="panel:model"),
            InlineKeyboardButton(text="Различия", callback_data="panel:model_help"),
        ],
        [
            InlineKeyboardButton(
                text=f"Порог {runtime.min_confidence:g}",
                callback_data="panel:confidence",
            ),
            InlineKeyboardButton(
                text=f"Кулдаун {runtime.cooldown_seconds:g}с",
                callback_data="panel:cooldown",
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"NMS IoU {runtime.nms_iou:g}",
                callback_data="panel:nms_iou",
            ),
            InlineKeyboardButton(
                text=f"Мин. площадь {runtime.person_min_area:g}",
                callback_data="panel:person_min_area",
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"Интервал {runtime.detection_interval:g}с",
                callback_data="panel:interval",
            ),
            InlineKeyboardButton(
                text=f"Лицо {runtime.face_threshold:g}",
                callback_data="panel:face",
            ),
        ],
        [
            InlineKeyboardButton(text=f"Тихие: {quiet}", callback_data="panel:quiet"),
            InlineKeyboardButton(text=f"Выход: {exit_s}", callback_data="panel:exit"),
        ],
        [
            InlineKeyboardButton(
                text=f"Приватность: {privacy}", callback_data="panel:privacy"
            ),
            InlineKeyboardButton(text=f"Remote: {remote}", callback_data="panel:remote"),
        ],
        [
            InlineKeyboardButton(text="Зоны", callback_data="panel:zones"),
            InlineKeyboardButton(text="Журнал действий", callback_data="panel:audit"),
        ],
        [
            InlineKeyboardButton(text="Голос", callback_data="voice:open"),
            InlineKeyboardButton(text="◀ Назад", callback_data="panel:open"),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


_PRIVACY_MODES = ["off", "person", "face", "silhouette", "all"]
_PRIVACY_LABELS = {
    "off": "выкл", "person": "человек", "face": "лицо",
    "silhouette": "силуэт", "all": "весь кадр",
}


def _privacy_keyboard(runtime: Runtime) -> InlineKeyboardMarkup:
    rows = []
    for mode in _PRIVACY_MODES:
        current = " ✓" if runtime.privacy_mode == mode else ""
        rows.append([InlineKeyboardButton(
            text=_PRIVACY_LABELS[mode] + current, callback_data=f"privacy:mode:{mode}",
        )])
    rows.extend([
        [InlineKeyboardButton(
            text=f"Пикселизация: {runtime.privacy_blocks} блоков",
            callback_data="privacy:blocks",
        )],
        [InlineKeyboardButton(text="◀ К настройкам", callback_data="panel:tune")],
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _model_keyboard(runtime: Runtime) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for name, info in models.YOLO_MODELS.items():
        mark = " [текущая]" if name == runtime.model_path else ""
        rows.append([
            InlineKeyboardButton(
                text=f"{info.title}{mark}", callback_data=f"model:set:{name}"
            )
        ])
    rows.append([InlineKeyboardButton(text="Назад", callback_data="panel:open")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _people_keyboard(storage: Storage, page: int = 0) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    total = storage.count_people()
    for row in storage.list_people_paged(PEOPLE_PAGE, page * PEOPLE_PAGE):
        n = int(row["face_count"] or 0)
        rows.append([
            InlineKeyboardButton(
                text=f"{row['name']} ({n} фото)",
                callback_data=f"person:view:{row['id']}",
            )
        ])
    rows.append([InlineKeyboardButton(text="Добавить", callback_data="person:add")])
    nav = _people_nav(page, total)
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton(text="Назад", callback_data="panel:open")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _people_nav(page: int, total: int) -> list[InlineKeyboardButton]:
    buttons: list[InlineKeyboardButton] = []
    pages = max(1, (total + PEOPLE_PAGE - 1) // PEOPLE_PAGE)
    if page > 0:
        buttons.append(InlineKeyboardButton(text="←", callback_data=f"people:page:{page - 1}"))
    if page + 1 < pages:
        buttons.append(InlineKeyboardButton(text="→", callback_data=f"people:page:{page + 1}"))
    return buttons


def _person_view_keyboard(person_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Ещё фото", callback_data=f"person:photo:{person_id}")],
        [InlineKeyboardButton(text="Удалить", callback_data=f"person:del:{person_id}")],
        [InlineKeyboardButton(text="Назад", callback_data="panel:people")],
    ])


def _person_confirm_keyboard(person_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="Удалить навсегда",
                callback_data=f"person:delconfirm:{person_id}",
            )
        ],
        [InlineKeyboardButton(text="Отмена", callback_data=f"person:view:{person_id}")],
    ])


def _capture_face_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Сканировать лицо с камеры", callback_data="person:capture")],
        [InlineKeyboardButton(text="Отмена", callback_data="person:add_cancel")],
    ])


def _events_nav(page: int, total: int) -> InlineKeyboardMarkup:
    buttons: list[InlineKeyboardButton] = []
    pages = max(1, (total + EVENTS_PAGE - 1) // EVENTS_PAGE)
    if page > 0:
        buttons.append(InlineKeyboardButton(text="← Новее", callback_data=f"events:page:{page - 1}"))
    buttons.append(InlineKeyboardButton(text=f"{page + 1}/{pages}", callback_data="events:noop"))
    if (page + 1) * EVENTS_PAGE < total:
        buttons.append(InlineKeyboardButton(text="Старее →", callback_data=f"events:page:{page + 1}"))
    buttons.append(InlineKeyboardButton(text="Назад", callback_data="panel:open"))
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def _who_keyboard(event_id: int, storage: Storage) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for row in storage.list_people():
        rows.append([
            InlineKeyboardButton(
                text=row["name"],
                callback_data=f"event:set:{event_id}:{row['id']}",
            )
        ])
    rows.append([
        InlineKeyboardButton(text="Новый человек", callback_data=f"event:new:{event_id}")
    ])
    rows.append([InlineKeyboardButton(text="Назад", callback_data="panel:open")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _event_keyboard(event_id: int, unknown_enter: bool, storage: Storage) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if unknown_enter:
        rows.append([InlineKeyboardButton(text="Это кто?", callback_data=f"event:who:{event_id}")])
    rows.append([InlineKeyboardButton(text="Добавить заметку", callback_data=f"event:note:{event_id}")])
    rows.append([InlineKeyboardButton(text="Вернуться в панель", callback_data="panel:open")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _zones_keyboard(runtime: Runtime) -> InlineKeyboardMarkup:
    zones = parse_zones(runtime.zones_json)
    rows: list[list[InlineKeyboardButton]] = []
    for i, zone in enumerate(zones):
        rows.append([
            InlineKeyboardButton(
                text=f"{zone.name} ({zone.x1:g},{zone.y1:g})-({zone.x2:g},{zone.y2:g})",
                callback_data=f"zone:del:{i}",
            )
        ])
    rows.append([InlineKeyboardButton(text="Добавить зону", callback_data="zone:add")])
    rows.append([InlineKeyboardButton(text="Назад", callback_data="panel:open")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


_CLEAR_ON_LABELS = {
    "both": "свой или уход",
    "known": "только свой",
    "exit": "уход незнакомца",
}


def _voice_text(runtime: Runtime, voice: VoiceEngine | None) -> str:
    alarm = "ТРЕВОГА" if (voice is not None and voice.alarming) else "тихо"
    backend = voice.backend if voice is not None else "нет"
    audio = "доступен" if voice is not None and voice.audio_available else "не найден"
    last = f"\nПоследняя фраза: {voice.last_phrase}" if voice and voice.last_phrase else ""
    err = f"\nОшибка: {voice.last_error}" if voice and voice.last_error else ""
    return (
        "Голос и тревога.\n"
        "Незнакомец в режиме identity: сирена и фраза из динамиков. "
        "Тревога звучит до кнопки «Снять тревогу» в Telegram или до "
        "распознавания человека из списка в камере. После снятия звучит "
        "подтверждение. "
        "«Сказать вслух» работает всегда, даже при выключенных авто-фразах.\n\n"
        f"Состояние: {alarm}\n"
        f"Движок: {backend}\n"
        f"Динамик: {audio}\n"
        f"Авто: {_on(runtime.voice_enabled)}, "
        f"тревога {_on(runtime.voice_alarm_enabled)}, "
        f"сирена {_on(runtime.voice_siren_enabled)}, "
        f"речь {_on(runtime.voice_speech_enabled)}\n"
        f"Снятие: {_CLEAR_ON_LABELS.get(runtime.voice_clear_on, runtime.voice_clear_on)}\n"
        f"Не спамить по «своему»: {_on(runtime.mute_known_present)}"
        f"{last}{err}"
    )


def _voice_keyboard(runtime: Runtime, voice: VoiceEngine | None) -> InlineKeyboardMarkup:
    alarming = voice is not None and voice.alarming
    vol = int(round(runtime.voice_volume * 100))
    rows = [
        [
            InlineKeyboardButton(
                text=f"Авто: {_on(runtime.voice_enabled)}",
                callback_data="voice:tog:voice_enabled",
            ),
            InlineKeyboardButton(
                text=f"Тревога: {_on(runtime.voice_alarm_enabled)}",
                callback_data="voice:tog:voice_alarm_enabled",
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"Сирена: {_on(runtime.voice_siren_enabled)}",
                callback_data="voice:tog:voice_siren_enabled",
            ),
            InlineKeyboardButton(
                text=f"Речь: {_on(runtime.voice_speech_enabled)}",
                callback_data="voice:tog:voice_speech_enabled",
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"Привет: {_on(runtime.voice_welcome)}",
                callback_data="voice:tog:voice_welcome",
            ),
            InlineKeyboardButton(
                text=f"Пока: {_on(runtime.voice_goodbye)}",
                callback_data="voice:tog:voice_goodbye",
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"Присутствие: {_on(runtime.voice_presence)}",
                callback_data="voice:tog:voice_presence",
            ),
            InlineKeyboardButton(
                text=f"Охрана голосом: {_on(runtime.voice_armed_announce)}",
                callback_data="voice:tog:voice_armed_announce",
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"Тревога в presence: {_on(runtime.voice_alarm_on_presence)}",
                callback_data="voice:tog:voice_alarm_on_presence",
            ),
            InlineKeyboardButton(
                text=f"Глушить в тихие: {_on(runtime.voice_mute_quiet)}",
                callback_data="voice:tog:voice_mute_quiet",
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"Не спамить по «своему»: {_on(runtime.mute_known_present)}",
                callback_data="voice:tog:mute_known_present",
            ),
        ],
        [InlineKeyboardButton(
            text="Снятие: вручную или распознанный человек",
            callback_data="voice:clear_info",
        )],
        [
            InlineKeyboardButton(
                text=f"Повтор {runtime.voice_repeat_seconds:g}с",
                callback_data="voice:num:voice_repeat_seconds",
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"Пауза ухода {runtime.voice_grace_seconds:g}с",
                callback_data="voice:num:voice_grace_seconds",
            ),
            InlineKeyboardButton(
                text=f"Кулдаун {runtime.voice_cooldown_seconds:g}с",
                callback_data="voice:num:voice_cooldown_seconds",
            ),
        ],
        [
            InlineKeyboardButton(text=f"Громкость {vol}%", callback_data="voice:num:voice_volume_pct"),
            InlineKeyboardButton(
                text=f"Скорость {runtime.voice_rate:g}",
                callback_data="voice:num:voice_rate",
            ),
        ],
        [
            InlineKeyboardButton(text="Фразы", callback_data="voice:phrases"),
            InlineKeyboardButton(text="Голоса TTS", callback_data="voice:voices"),
        ],
        [
            InlineKeyboardButton(text="Сказать вслух", callback_data="voice:say"),
            InlineKeyboardButton(text="Тест сирены", callback_data="voice:test"),
        ],
        [
            InlineKeyboardButton(
                text="Снять тревогу" if alarming else "Тревога вручную",
                callback_data="voice:dismiss" if alarming else "voice:trigger",
            ),
        ],
        [InlineKeyboardButton(text="Назад", callback_data="panel:open")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _phrase_keyboard(runtime: Runtime) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for key, title in PHRASE_TITLES.items():
        preview = _short_phrase(runtime.voice_phrase(key))
        rows.append([
            InlineKeyboardButton(
                text=f"{title}: {preview}",
                callback_data=f"voice:ph:{key}",
            ),
            InlineKeyboardButton(text="▶", callback_data=f"voice:phrase_preview:{key}"),
        ])
    rows.append([InlineKeyboardButton(text="Назад", callback_data="voice:open")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


_VOICE_SLOTS = {
    "default": ("Обычные фразы", "voice_tts_voice"),
    "alarm": ("Тревога", "voice_alarm_tts_voice"),
    "welcome": ("Приветствие и прощание", "voice_welcome_tts_voice"),
}


def _voice_slots_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=title, callback_data=f"voice:slot:{slot}")]
        for slot, (title, _field) in _VOICE_SLOTS.items()
    ] + [[InlineKeyboardButton(text="Назад", callback_data="voice:open")]])


def _voices_keyboard(
    runtime: Runtime, voices: list[tuple[str, str]], slot: str
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    _title, field = _VOICE_SLOTS[slot]
    current = str(getattr(runtime, field))
    for i, (vid, title) in enumerate(voices[:15]):
        mark = " [текущий]" if vid == current else ""
        preview = " ▶" if i == 0 else ""
        rows.append([
            InlineKeyboardButton(
                text=f"{_short_phrase(title, 40)}{mark}{preview}",
                callback_data=f"voice:vset:{slot}:{i}",
            ),
            InlineKeyboardButton(
                text="🔊", callback_data=f"voice:prev:{slot}:{i}"
            ),
        ])
    rows.append([
        InlineKeyboardButton(text="Использовать обычный голос", callback_data=f"voice:vset:{slot}:-1")
    ])
    rows.append([InlineKeyboardButton(text="К профилям", callback_data="voice:voices")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _status_text(runtime: Runtime, recognizer: Recognizer | None,
                 pipeline: Pipeline | None,
                 voice: VoiceEngine | None = None) -> str:
    face = _face(recognizer, pipeline)
    face_s = "● доступно" if face is not None and face.available() else "○ dummy"
    remote_s = _ico(runtime.remote_processing)
    running = pipeline is not None and pipeline.running
    running_s = _ico(running)
    info = models.get_model(runtime.model_path)
    model_title = info.title if info else runtime.model_path
    quiet = runtime.quiet_hours or "выкл"
    lines = [
        "DotEye - статус",
        "",
        "Охрана",
        f"{_ico(runtime.armed)} Охрана: {_on(runtime.armed)}",
        f"Тихие часы: {quiet}",
        f"{_ico(runtime.notify_exit)} Уведомлять выход: {_on(runtime.notify_exit)}",
        f"{_ico(runtime.privacy_outbound)} Приватные кадры: {_on(runtime.privacy_outbound)}",
        "",
        "Детекция",
        f"{running_s} Пайплайн: {'работает' if running else 'остановлен'}",
        f"Режим: {_mode_text(runtime.detect_mode)}",
        f"Камера: {runtime.camera_source}",
        f"Детектор: {runtime.detector_backend} (device={runtime.device})",
        f"Модель: {model_title}",
        f"imgsz: {runtime.imgsz} · порог: {runtime.min_confidence:g}",
        f"Кулдаун: {runtime.cooldown_seconds:g}с · интервал: {runtime.detection_interval:g}с",
        f"Порог лица: {runtime.face_threshold:g} · распознавание: {face_s}",
        f"{remote_s} Remote: {'вкл' if runtime.remote_processing else 'выкл'} ({runtime.remote_url or 'нет URL'})",
        "",
        "Голос и тревога",
        f"{_ico(runtime.voice_enabled)} Авто: {_on(runtime.voice_enabled)} · тревога {_on(runtime.voice_alarm_enabled)}",
        f"{_ico(runtime.voice_siren_enabled)} Сирена: {_on(runtime.voice_siren_enabled)} · речь {_on(runtime.voice_speech_enabled)}",
        f"Привет/пока: {_on(runtime.voice_welcome)}/{_on(runtime.voice_goodbye)}",
    ]
    engine = voice if voice is not None else (
        pipeline.voice if pipeline is not None else None
    )
    if engine is not None and pipeline is None:
        lines.append(engine.status_line())
    if pipeline is not None:
        lines.append("")
        lines.append("Здоровье")
        lines.append(pipeline.health_text())
    return "\n".join(lines)


def _event_box(row: Any) -> tuple[int, int, int, int] | None:
    raw = row["boxes"] if row["boxes"] else ""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        item = data[0]
        return int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        return None


def _event_boxes(row: Any) -> list[tuple[int, int, int, int]]:
    """Извлечь валидные боксы события, не доверяя данным старой БД."""
    try:
        values = json.loads(row["boxes"] or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    result: list[tuple[int, int, int, int]] = []
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, dict):
            continue
        try:
            result.append(tuple(int(value[key]) for key in ("x1", "y1", "x2", "y2")))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _embed_jpeg(
    jpeg: bytes, recognizer: Recognizer, box: tuple[int, int, int, int] | None
) -> bytes | None:
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    target = crop_box(frame, box) if box is not None else frame
    if target is None:
        target = frame
    return recognizer.embed(target)


def _assign_event(
    event_id: int, person_id: int, storage: Storage,
    crypto: Crypto | None, recognizer: Recognizer | None,
) -> str:
    row = storage.get_event(event_id)
    if row is None:
        return "Событие не найдено."
    person = storage.get_person_by_id(person_id)
    if person is None:
        return "Человек не найден."
    storage.update_event_person(event_id, person_id)
    face = recognizer
    if crypto is None or face is None or not face.available() or not row["frame"]:
        return f"Событие привязано к «{person['name']}» (без нового фото)."
    try:
        jpeg = crypto.decrypt(row["frame"])
    except Exception:
        return f"Событие привязано к «{person['name']}», кадр не расшифрован."
    embedding = _embed_jpeg(jpeg, face, _event_box(row))
    if embedding is None:
        return f"Событие привязано к «{person['name']}», лицо на кадре не найдено."
    storage.add_embedding(person_id, crypto.encrypt(embedding))
    return f"«{person['name']}»: эталон с события добавлен."


async def _send_events_page(
    message: Message, storage: Storage, crypto: Crypto | None,
    runtime: Runtime, page: int,
) -> None:
    if crypto is None:
        await message.answer("DOTEYE_CRYPTO_KEY не задан, кадры недоступны.")
        return
    total = storage.count_events()
    if total == 0:
        await message.answer("Событий пока нет.", reply_markup=_back_kb())
        return
    page = max(0, page)
    rows = storage.recent_events(EVENTS_PAGE, page * EVENTS_PAGE)
    if not rows:
        await message.answer("На этой странице пусто.", reply_markup=_events_nav(0, total))
        return
    await message.answer(
        f"События {page * EVENTS_PAGE + 1}–{page * EVENTS_PAGE + len(rows)} из {total}",
        reply_markup=_events_nav(page, total),
    )
    for row in rows:
        kind = row["event_type"] or "enter"
        conf = f", conf={row['confidence']:.2f}" if row["confidence"] else ""
        cam = f"\nкамера {row['camera_source']}" if row["camera_source"] else ""
        zone = f", зона {row['zone']}" if row["zone"] else ""
        note = ""
        if row["note"]:
            try:
                note_text = crypto.decrypt(bytes(row["note"])).decode("utf-8")
                note = f"\nзаметка: {note_text}"
            except Exception:
                note = "\nзаметка: недоступна"
        if kind == "enter" and row["person_name"]:
            summary = f"Обнаружен: {row['person_name']}"
        elif kind == "enter":
            summary = "Обнаружен незнакомый человек"
        elif row["person_name"]:
            summary = f"Вышел: {row['person_name']}"
        else:
            summary = "Незнакомый человек вышел"
        caption = f"{row['detected_at']}\n{summary}{conf}{cam}{zone}"
        caption += note
        markup = _event_keyboard(int(row["id"]), row["person_name"] is None and kind == "enter", storage)
        if not row["frame"]:
            await message.answer(caption, reply_markup=markup)
            continue
        try:
            jpeg = crypto.decrypt(row["frame"])
        except Exception:
            await message.answer(caption + "\n(кадр не расшифрован)")
            continue
        if runtime.privacy_mode != "off":
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await message.answer(caption + "\n(кадр повреждён)", reply_markup=markup)
                continue
            jpeg = encode_jpeg(privacy_frame(
                frame, _event_boxes(row), runtime.privacy_mode, runtime.privacy_blocks,
            ), runtime.jpeg_quality)
            if jpeg is None:
                await message.answer(caption + "\n(кадр не подготовлен)", reply_markup=markup)
                continue
        await message.answer_photo(
            BufferedInputFile(jpeg, filename=f"event_{row['id']}.jpg"),
            caption=caption, reply_markup=markup,
        )


async def _send_audit_log(message: Message, storage: Storage, crypto: Crypto | None) -> None:
    """Показать журнал без возможности вывести незашифрованные данные."""
    if crypto is None:
        await message.answer("Журнал недоступен: ключ шифрования не задан.")
        return
    rows = storage.recent_audit_entries(AUDIT_PAGE)
    if not rows:
        await message.answer("Журнал действий пока пуст.", reply_markup=_back_kb())
        return
    lines = ["Журнал действий. Содержимое сообщений, имена и фото не записываются."]
    for row in rows:
        try:
            action = crypto.decrypt(bytes(row["payload"])).decode("utf-8")
        except Exception:
            action = "повреждённая зашифрованная запись"
        lines.append(f"{row['created_at']} - {action}")
    await message.answer("\n".join(lines), reply_markup=_back_kb())


# -- команды ------------------------------------------------------------

@router.message(CommandStart())
async def cmd_start(message: Message, settings: Settings) -> None:
    admin = "админ" if _is_admin(message, settings) else "гость"
    await message.answer(
        f"DotEye на связи ({admin}).\n\n"
        "Основные действия - в нижнем меню, подробные - в панели.\n"
        "Подключи камеру через /camera, выбери режим.\n\n"
        "/help - все команды\n"
        "/cancel - отменить ввод",
        reply_markup=_main_menu(),
    )


@router.message(Command("panel"))
async def cmd_panel(message: Message, runtime: Runtime,
                    voice: VoiceEngine | None = None) -> None:
    await message.answer("Постоянное меню включено.", reply_markup=_main_menu())
    await message.answer(
        "Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime, voice)
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP)


@router.message(Command("cancel"))
@router.message(F.text.casefold() == "отмена")
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    current = await state.get_state()
    if current is None:
        await message.answer("Нечего отменять. /help")
        return
    await state.clear()
    await message.answer("Отменено.")


@router.message(Command("status"))
async def cmd_status(message: Message, runtime: Runtime,
                     recognizer: Recognizer | None, pipeline: Pipeline | None,
                     voice: VoiceEngine | None = None) -> None:
    await message.answer(_status_text(runtime, recognizer, pipeline, voice))


@router.message(F.text == "Панель")
async def menu_panel(message: Message, runtime: Runtime,
                     voice: VoiceEngine | None = None) -> None:
    await message.answer("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime, voice))


@router.message(F.text == "Статус")
async def menu_status(message: Message, runtime: Runtime,
                      recognizer: Recognizer | None, pipeline: Pipeline | None,
                      voice: VoiceEngine | None = None) -> None:
    await message.answer(_status_text(runtime, recognizer, pipeline, voice))


@router.message(F.text == "События")
async def menu_events(message: Message, storage: Storage, crypto: Crypto | None,
                      runtime: Runtime) -> None:
    await _send_events_page(message, storage, crypto, runtime, 0)


@router.message(F.text == "Люди")
async def menu_people(message: Message, storage: Storage) -> None:
    await message.answer("Люди:", reply_markup=_people_keyboard(storage, 0))


@router.message(F.text == "Голос и тревога")
async def menu_voice(message: Message, runtime: Runtime,
                     voice: VoiceEngine | None = None) -> None:
    await message.answer(_voice_text(runtime, voice), reply_markup=_voice_keyboard(runtime, voice))


@router.message(F.text == "Превью")
async def menu_snapshot(message: Message, pipeline: Pipeline | None) -> None:
    if pipeline is None:
        await message.answer("Пайплайн выключен.")
        return
    jpeg = await asyncio.to_thread(pipeline.snapshot)
    if jpeg is None:
        await message.answer("Кадр пока недоступен.")
        return
    await message.answer_photo(BufferedInputFile(jpeg, filename="snapshot.jpg"))


@router.callback_query(F.data == "panel:open")
async def cb_open(cq: CallbackQuery, runtime: Runtime,
                  voice: VoiceEngine | None = None) -> None:
    await cq.message.edit_text(
        "Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime, voice)
    )
    await cq.answer()


@router.callback_query(F.data == "panel:tune")
async def cb_tune(cq: CallbackQuery, runtime: Runtime) -> None:
    await cq.message.edit_text(
        "Тонкая настройка детекции, камер, зон и уведомлений:",
        reply_markup=_tune_keyboard(runtime),
    )
    await cq.answer()


@router.callback_query(F.data == "panel:status")
async def cb_status(cq: CallbackQuery, runtime: Runtime,
                    recognizer: Recognizer | None, pipeline: Pipeline | None,
                    voice: VoiceEngine | None = None) -> None:
    await cq.message.edit_text(
        _status_text(runtime, recognizer, pipeline, voice), reply_markup=_back_kb()
    )
    await cq.answer()


@router.callback_query(F.data == "panel:health")
async def cb_health(cq: CallbackQuery, pipeline: Pipeline | None, runtime: Runtime,
                    recognizer: Recognizer | None,
                    voice: VoiceEngine | None = None) -> None:
    text = _status_text(runtime, recognizer, pipeline, voice)
    await cq.message.edit_text(text, reply_markup=_back_kb())
    await cq.answer()


@router.callback_query(F.data == "panel:arm")
async def cb_arm(cq: CallbackQuery, runtime: Runtime,
                 voice: VoiceEngine | None = None) -> None:
    runtime.armed = not runtime.armed
    if voice is not None:
        voice.sync_armed(runtime.armed)
    await cq.message.edit_text(
        "Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime, voice)
    )
    await cq.answer("Охрана вкл" if runtime.armed else "Охрана выкл")


@router.callback_query(F.data == "panel:exit")
async def cb_exit_toggle(cq: CallbackQuery, runtime: Runtime) -> None:
    runtime.notify_exit = not runtime.notify_exit
    await cq.message.edit_text(
        "Тонкая настройка:", reply_markup=_tune_keyboard(runtime)
    )
    await cq.answer("Выход: " + ("вкл" if runtime.notify_exit else "выкл"))


@router.callback_query(F.data == "panel:privacy")
async def cb_privacy_toggle(cq: CallbackQuery, runtime: Runtime) -> None:
    await cq.message.edit_text(
        "Приватность кадров в Telegram\n\n"
        "Выбери, что скрывать. В режиме «лицо», если лицо не найдено, "
        "пикселизируется весь человек - кадр не уйдёт с открытым лицом.\n"
        "Меньше блоков - сильнее пикселизация.",
        reply_markup=_privacy_keyboard(runtime),
    )
    await cq.answer()


@router.callback_query(F.data.startswith("privacy:mode:"))
async def cb_privacy_mode(cq: CallbackQuery, runtime: Runtime) -> None:
    mode = cq.data.split(":", 2)[2]
    if mode not in _PRIVACY_MODES:
        await cq.answer("Неизвестный режим", show_alert=True)
        return
    runtime.privacy_mode = mode
    await cq.message.edit_text(
        "Приватность кадров в Telegram\n\n"
        "Выбери, что скрывать. Меньше блоков - сильнее пикселизация.",
        reply_markup=_privacy_keyboard(runtime),
    )
    await cq.answer("Режим: " + _PRIVACY_LABELS[mode])


@router.callback_query(F.data == "privacy:blocks")
async def cb_privacy_blocks(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(NumberForm.value)
    await state.update_data(field="privacy_blocks", lo=2, hi=64)
    await cq.message.answer(
        f"Сейчас {runtime.privacy_blocks} блоков. Введи число 2..64: "
        "2 - очень сильная пикселизация, 64 - слабее."
    )
    await cq.answer()


@router.callback_query(F.data == "panel:mode")
async def cb_mode(
    cq: CallbackQuery, runtime: Runtime, recognizer: Recognizer | None = None,
) -> None:
    runtime.detect_mode = "identity" if runtime.detect_mode == "presence" else "presence"
    await cq.message.edit_text("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))
    await cq.answer("Режим: " + _mode_text(runtime.detect_mode), show_alert=True)


@router.callback_query(F.data == "panel:detector")
async def cb_detector(cq: CallbackQuery, runtime: Runtime) -> None:
    runtime.detector_backend = _next_item(DETECTORS, runtime.detector_backend)
    await cq.message.edit_text(
        "Тонкая настройка:", reply_markup=_tune_keyboard(runtime)
    )
    await cq.answer(f"Детектор: {runtime.detector_backend}")


@router.callback_query(F.data == "panel:device")
async def cb_device(cq: CallbackQuery, runtime: Runtime) -> None:
    runtime.device = _next_item(DEVICES, runtime.device)
    await cq.message.edit_text(
        "Тонкая настройка:", reply_markup=_tune_keyboard(runtime)
    )
    await cq.answer(f"Устройство: {runtime.device}")


@router.callback_query(F.data == "panel:model")
async def cb_model(cq: CallbackQuery, runtime: Runtime) -> None:
    await cq.message.edit_text(
        "Выбери YOLO-модель. Детектор пересоберётся на ходу:",
        reply_markup=_model_keyboard(runtime),
    )
    await cq.answer()


@router.callback_query(F.data == "panel:model_help")
async def cb_model_help(cq: CallbackQuery) -> None:
    await cq.message.answer(models.comparison(), parse_mode="Markdown")
    await cq.answer()


@router.callback_query(F.data.startswith("model:set:"))
async def cb_model_set(cq: CallbackQuery, runtime: Runtime,
                       pipeline: Pipeline | None) -> None:
    name = cq.data.split(":", 2)[2]
    info = models.get_model(name)
    if info is None:
        await cq.answer("Неизвестная модель", show_alert=True)
        return
    runtime.model_path = name
    note = ""
    if runtime.detector_backend != "yolo":
        runtime.detector_backend = "yolo"
        note = "\nДетектор переключён на yolo."
    if pipeline is None:
        note += "\nПайплайн выключен (нет ключа), модель применится при старте."
    else:
        note += "\nДетектор перезапустится на следующем кадре."
    await cq.message.edit_text(
        models.describe(name) + note, parse_mode="Markdown",
        reply_markup=_model_keyboard(runtime),
    )
    await cq.answer(f"Выбрано: {info.title}")


@router.callback_query(F.data == "panel:snapshot")
async def cb_snapshot(cq: CallbackQuery, pipeline: Pipeline | None) -> None:
    if pipeline is None:
        await cq.answer("Пайплайн выключен", show_alert=True)
        return
    jpeg = await asyncio.to_thread(pipeline.snapshot)
    if jpeg is None:
        await cq.answer("Кадр недоступен", show_alert=True)
        return
    await cq.message.answer_photo(BufferedInputFile(jpeg, filename="snapshot.jpg"))
    await cq.answer()


@router.callback_query(F.data == "panel:people")
async def cb_people(cq: CallbackQuery, storage: Storage) -> None:
    total = storage.count_people()
    text = "Люди:" if total else "Список людей пуст."
    await cq.message.edit_text(text, reply_markup=_people_keyboard(storage, 0))
    await cq.answer()


@router.callback_query(F.data.startswith("people:page:"))
async def cb_people_page(cq: CallbackQuery, storage: Storage) -> None:
    page = int(cq.data.split(":")[2])
    await cq.message.edit_text("Люди:", reply_markup=_people_keyboard(storage, page))
    await cq.answer()


@router.callback_query(F.data.startswith("person:view:"))
async def cb_person_view(cq: CallbackQuery, storage: Storage) -> None:
    person_id = int(cq.data.split(":")[2])
    person = storage.get_person_by_id(person_id)
    if person is None:
        await cq.answer("Не найден", show_alert=True)
        return
    n = len(storage.embeddings_for(person_id))
    await cq.message.edit_text(
        f"{person['name']}\nэталонов: {n}\nсоздан: {person['created_at']}",
        reply_markup=_person_view_keyboard(person_id),
    )
    await cq.answer()


@router.callback_query(F.data == "person:add")
async def cb_person_add(cq: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(AddPersonForm.name)
    await cq.message.answer("Имя человека (или «отмена»):")
    await cq.answer()


@router.callback_query(F.data == "person:add_cancel")
async def cb_person_add_cancel(cq: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await cq.message.answer("Добавление человека отменено.")
    await cq.answer()


@router.callback_query(F.data == "person:capture")
async def cb_person_capture(
    cq: CallbackQuery, state: FSMContext, storage: Storage, crypto: Crypto | None,
    recognizer: Recognizer | None, pipeline: Pipeline | None,
) -> None:
    data = await state.get_data()
    name = str(data.get("name") or "").strip()
    if await state.get_state() != AddPersonForm.photo.state or not name:
        await cq.answer("Сначала начни добавление человека.", show_alert=True)
        return
    face = _face(recognizer, pipeline)
    if crypto is None or face is None or not face.available() or pipeline is None:
        await cq.answer("Распознавание или камера недоступны.", show_alert=True)
        return
    embedding = await asyncio.to_thread(pipeline.capture_face_embedding)
    if embedding is None:
        await cq.answer("Лицо не найдено. Посмотри в камеру и повтори.", show_alert=True)
        return
    storage.upsert_person(name, crypto.encrypt(embedding))
    await state.clear()
    await cq.message.answer(f"«{name}» добавлен по кадру камеры. Фото не сохранено.")
    await cq.answer("Лицо сохранено")


@router.callback_query(F.data.startswith("person:del:"))
async def cb_person_del(cq: CallbackQuery, storage: Storage) -> None:
    person_id = int(cq.data.split(":")[2])
    person = storage.get_person_by_id(person_id)
    if person is None:
        await cq.answer("Не найден", show_alert=True)
        return
    n = len(storage.embeddings_for(person_id))
    photo_note = f" и {n} эталонов" if n else ""
    await cq.message.edit_text(
        f"Удалить «{person['name']}»{photo_note}?\nЭто действие необратимо.",
        reply_markup=_person_confirm_keyboard(person_id),
    )
    await cq.answer()


@router.callback_query(F.data.startswith("person:delconfirm:"))
async def cb_person_del_confirm(cq: CallbackQuery, storage: Storage) -> None:
    person_id = int(cq.data.split(":")[2])
    person = storage.get_person_by_id(person_id)
    name = person["name"] if person else "?"
    ok = storage.delete_person_by_id(person_id)
    text = f"Удалён: {name}" if ok else "Не найден."
    await cq.message.edit_text(text, reply_markup=_people_keyboard(storage))
    await cq.answer(text)


@router.callback_query(F.data.startswith("person:photo:"))
async def cb_person_photo(cq: CallbackQuery, state: FSMContext) -> None:
    person_id = int(cq.data.split(":")[2])
    await state.set_state(PhotoWaitForm.photo)
    await state.update_data(person_id=person_id)
    await cq.message.answer("Пришли ещё одно фото лица (или «отмена»).")
    await cq.answer()


@router.callback_query(F.data == "panel:events")
async def cb_events(cq: CallbackQuery, storage: Storage,
                    crypto: Crypto | None, runtime: Runtime) -> None:
    await cq.answer()
    await _send_events_page(cq.message, storage, crypto, runtime, 0)


@router.callback_query(F.data == "panel:audit")
async def cb_audit(cq: CallbackQuery, storage: Storage, crypto: Crypto | None) -> None:
    await cq.answer()
    await _send_audit_log(cq.message, storage, crypto)


@router.callback_query(F.data.startswith("events:page:"))
async def cb_events_page(cq: CallbackQuery, storage: Storage,
                         crypto: Crypto | None, runtime: Runtime) -> None:
    page = int(cq.data.split(":")[2])
    await cq.answer()
    await _send_events_page(cq.message, storage, crypto, runtime, page)


@router.callback_query(F.data == "events:noop")
async def cb_events_noop(cq: CallbackQuery) -> None:
    await cq.answer()


@router.callback_query(F.data.startswith("event:who:"))
async def cb_event_who(cq: CallbackQuery, storage: Storage) -> None:
    event_id = int(cq.data.split(":")[2])
    await cq.message.answer(
        "Кто это?", reply_markup=_who_keyboard(event_id, storage)
    )
    await cq.answer()


@router.callback_query(F.data.startswith("event:note:"))
async def cb_event_note(cq: CallbackQuery, state: FSMContext) -> None:
    event_id = int(cq.data.split(":")[2])
    await state.set_state(TextForm.value)
    await state.update_data(kind="event_note", event_id=event_id)
    await cq.message.answer(
        "Заметка к событию: кто пришёл, цель визита или что произошло. "
        "До 500 символов, «отмена» - выйти."
    )
    await cq.answer()


@router.callback_query(F.data.startswith("event:set:"))
async def cb_event_set(cq: CallbackQuery, storage: Storage,
                       crypto: Crypto | None, recognizer: Recognizer | None,
                       pipeline: Pipeline | None) -> None:
    parts = cq.data.split(":")
    event_id, person_id = int(parts[2]), int(parts[3])
    text = _assign_event(event_id, person_id, storage, crypto, _face(recognizer, pipeline))
    await cq.message.answer(text)
    await cq.answer("Сохранено")


@router.callback_query(F.data.startswith("event:new:"))
async def cb_event_new(cq: CallbackQuery, state: FSMContext) -> None:
    event_id = int(cq.data.split(":")[2])
    await state.set_state(TextForm.value)
    await state.update_data(kind="assign_name", event_id=event_id)
    await cq.message.answer("Имя нового человека (или «отмена»):")
    await cq.answer()


@router.callback_query(F.data == "panel:help")
async def cb_help(cq: CallbackQuery) -> None:
    await cq.message.answer(HELP)
    await cq.answer()


async def _edit_voice_panel(
    cq: CallbackQuery, runtime: Runtime, voice: VoiceEngine | None,
) -> None:
    await cq.message.edit_text(
        _voice_text(runtime, voice),
        reply_markup=_voice_keyboard(runtime, voice),
    )


@router.callback_query(F.data == "voice:open")
async def cb_voice_open(cq: CallbackQuery, runtime: Runtime,
                        voice: VoiceEngine | None = None) -> None:
    await _edit_voice_panel(cq, runtime, voice)
    await cq.answer()


_VOICE_TOGGLES = {
    "voice_enabled",
    "voice_alarm_enabled",
    "voice_siren_enabled",
    "voice_speech_enabled",
    "voice_welcome",
    "voice_goodbye",
    "voice_presence",
    "voice_armed_announce",
    "voice_alarm_on_presence",
    "voice_mute_quiet",
    "mute_known_present",
}

_VOICE_NUM_FIELDS = {
    "voice_repeat_seconds": (2.0, 3600.0, "Повтор тревоги, сек (минимум 2):"),
    "voice_timeout_seconds": (0.0, 86400.0, "Таймаут тревоги, сек (0 = пока не снимут):"),
    "voice_grace_seconds": (0.0, 3600.0, "Пауза после ухода незнакомца, сек:"),
    "voice_cooldown_seconds": (0.0, 86400.0, "Кулдаун приветствия/прощания, сек:"),
    "voice_rate": (0.4, 2.5, "Скорость речи 0.4..2.5:"),
    "voice_volume_pct": (0.0, 100.0, "Громкость 0..100:"),
}


@router.callback_query(F.data.startswith("voice:tog:"))
async def cb_voice_toggle(cq: CallbackQuery, runtime: Runtime,
                          voice: VoiceEngine | None = None) -> None:
    field = cq.data.split(":", 2)[2]
    if field not in _VOICE_TOGGLES:
        await cq.answer("Неизвестная настройка", show_alert=True)
        return
    setattr(runtime, field, not bool(getattr(runtime, field)))
    await _edit_voice_panel(cq, runtime, voice)
    await cq.answer(f"{field}: {_on(bool(getattr(runtime, field)))}")


@router.callback_query(F.data == "voice:clear_on")
async def cb_voice_clear_on(cq: CallbackQuery, runtime: Runtime,
                            voice: VoiceEngine | None = None) -> None:
    runtime.voice_clear_on = _next_item(list(CLEAR_ON_VALUES), runtime.voice_clear_on)
    await _edit_voice_panel(cq, runtime, voice)
    await cq.answer("Снятие: " + _CLEAR_ON_LABELS.get(runtime.voice_clear_on, runtime.voice_clear_on))


@router.callback_query(F.data == "voice:clear_info")
async def cb_voice_clear_info(cq: CallbackQuery) -> None:
    await cq.answer("Снять: кнопкой в Telegram или лицом человека из списка", show_alert=True)


@router.callback_query(F.data.startswith("voice:num:"))
async def cb_voice_number(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    field = cq.data.split(":", 2)[2]
    spec = _VOICE_NUM_FIELDS.get(field)
    if spec is None:
        await cq.answer("Неизвестное поле", show_alert=True)
        return
    lo, hi, prompt = spec
    await state.set_state(NumberForm.value)
    await state.update_data(field=field, lo=lo, hi=hi)
    current = runtime.voice_volume * 100 if field == "voice_volume_pct" else getattr(runtime, field)
    await cq.message.answer(f"{prompt}\nСейчас: {current:g}")
    await cq.answer()


@router.callback_query(F.data == "voice:phrases")
async def cb_voice_phrases(cq: CallbackQuery, runtime: Runtime) -> None:
    await cq.message.edit_text(
        "Фразы. Нажми чтобы заменить. Плейсхолдер {name} подставляет имя.",
        reply_markup=_phrase_keyboard(runtime),
    )
    await cq.answer()


@router.callback_query(F.data.startswith("voice:ph:"))
async def cb_voice_phrase_edit(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    key = cq.data.split(":", 2)[2]
    if key not in DEFAULT_PHRASES:
        await cq.answer("Неизвестная фраза", show_alert=True)
        return
    await state.set_state(TextForm.value)
    await state.update_data(kind=f"phrase:{key}")
    title = PHRASE_TITLES.get(key, key)
    await cq.message.answer(
        f"{title}. Сейчас:\n{runtime.voice_phrase(key)}\n\n"
        "Пришли новый текст (или «отмена»)."
    )
    await cq.answer()


@router.callback_query(F.data == "voice:voices")
async def cb_voice_voices(cq: CallbackQuery, runtime: Runtime,
                          voice: VoiceEngine | None = None) -> None:
    if voice is None:
        await cq.answer("Голосовой движок не создан", show_alert=True)
        return
    voices = await asyncio.to_thread(voice.list_voices)
    if not voices:
        await cq.message.edit_text(
            "TTS-голоса не найдены. На Windows нужен голосовой пакет "
            "(например Ирина), на Linux - espeak-ng.",
            reply_markup=_voice_slots_keyboard(),
        )
        await cq.answer()
        return
    await cq.message.edit_text(
        "Выбери профиль, для которого нужно назначить голос:",
        reply_markup=_voice_slots_keyboard(),
    )
    await cq.answer()


@router.callback_query(F.data.startswith("voice:slot:"))
async def cb_voice_slot(cq: CallbackQuery, runtime: Runtime,
                        voice: VoiceEngine | None = None) -> None:
    slot = cq.data.split(":", 2)[2]
    if slot not in _VOICE_SLOTS or voice is None:
        await cq.answer("Голосовой движок или профиль недоступны.", show_alert=True)
        return
    voices = await asyncio.to_thread(voice.list_voices)
    title, _field = _VOICE_SLOTS[slot]
    await cq.message.edit_text(
        f"Голос для профиля «{title}»:",
        reply_markup=_voices_keyboard(runtime, voices, slot),
    )
    await cq.answer()


@router.callback_query(F.data.startswith("voice:vset:"))
async def cb_voice_voice_set(cq: CallbackQuery, runtime: Runtime,
                             voice: VoiceEngine | None = None) -> None:
    if voice is None:
        await cq.answer("Голосовой движок не создан", show_alert=True)
        return
    parts = cq.data.split(":")
    if len(parts) != 4 or parts[2] not in _VOICE_SLOTS:
        await cq.answer("Неизвестный профиль голоса", show_alert=True)
        return
    slot, idx = parts[2], int(parts[3])
    title, field = _VOICE_SLOTS[slot]
    if idx < 0:
        setattr(runtime, field, "")
        await cq.message.edit_text(
            f"Профиль «{title}» использует обычный голос.",
            reply_markup=_voice_slots_keyboard(),
        )
        await cq.answer("Сохранено")
        return
    voices = await asyncio.to_thread(voice.list_voices)
    if idx >= len(voices):
        await cq.answer("Голос не найден", show_alert=True)
        return
    if str(getattr(runtime, field)) == voices[idx][0]:
        await cq.answer("Этот голос уже выбран", show_alert=True)
        return
    setattr(runtime, field, voices[idx][0])
    await cq.message.edit_text(
        f"Профиль «{title}»: {voices[idx][1]}",
        reply_markup=_voices_keyboard(runtime, voices, slot),
    )
    await cq.answer("Сохранено")


@router.callback_query(F.data.startswith("voice:prev:"))
async def cb_voice_preview(cq: CallbackQuery,
                           voice: VoiceEngine | None = None) -> None:
    if voice is None:
        await cq.answer("Голосовой движок не создан", show_alert=True)
        return
    parts = cq.data.split(":")
    if len(parts) != 4 or parts[2] not in _VOICE_SLOTS:
        await cq.answer("Неизвестный профиль голоса", show_alert=True)
        return
    slot, idx = parts[2], int(parts[3])
    voices = await asyncio.to_thread(voice.list_voices)
    if idx < 0 or idx >= len(voices):
        await cq.answer("Голос не найден", show_alert=True)
        return
    voice_id = voices[idx][0]
    if await asyncio.to_thread(voice.preview_voice, voice_id):
        await cq.answer("Пример голоса воспроизводится")
    else:
        await cq.answer("Нет синтезатора речи или аудиовыхода - открой «Статус»", show_alert=True)


@router.callback_query(F.data.startswith("voice:phrase_preview:"))
async def cb_voice_phrase_preview(cq: CallbackQuery, runtime: Runtime,
                                  voice: VoiceEngine | None = None) -> None:
    key = cq.data.split(":", 2)[2]
    if key not in DEFAULT_PHRASES or voice is None:
        await cq.answer("Пример недоступен", show_alert=True)
        return
    text = runtime.voice_phrase(key).replace("{name}", "гость")
    if voice.speech_available and voice.audio_available and voice.announce(text):
        await cq.answer("Фраза воспроизводится")
    else:
        await cq.answer("Нет синтезатора речи или аудиовыхода - открой «Статус»", show_alert=True)


@router.callback_query(F.data == "voice:say")
async def cb_voice_say(cq: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(TextForm.value)
    await state.update_data(kind="say")
    await cq.message.answer(
        "Текст для озвучки, например «Отойди от двери». «отмена» - выйти."
    )
    await cq.answer()


@router.callback_query(F.data == "voice:test")
async def cb_voice_test(cq: CallbackQuery, voice: VoiceEngine | None = None) -> None:
    if voice is None:
        await cq.answer("Голосовой движок не создан", show_alert=True)
        return
    if voice.test_alarm():
        await cq.answer("Тест отправлен в динамик")
    else:
        await cq.answer("Аудиовыход не найден: проверь устройство и статус", show_alert=True)


@router.callback_query(F.data == "voice:trigger")
async def cb_voice_trigger(cq: CallbackQuery, runtime: Runtime,
                           voice: VoiceEngine | None = None) -> None:
    if voice is None:
        await cq.answer("Голосовой движок не создан", show_alert=True)
        return
    voice.trigger_alarm()
    await _edit_voice_panel(cq, runtime, voice)
    await cq.answer("Тревога запущена")


@router.callback_query(F.data == "voice:dismiss")
async def cb_voice_dismiss(cq: CallbackQuery, runtime: Runtime,
                           voice: VoiceEngine | None = None) -> None:
    if voice is None:
        await cq.answer("Голосовой движок не создан", show_alert=True)
        return
    notice = voice.dismiss("manual")
    if cq.message:
        try:
            await cq.message.delete()
            await cq.message.answer(
                "Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime, voice),
            )
        except Exception:
            await _edit_voice_panel(cq, runtime, voice)
    await cq.answer("Тревога снята" if notice else "Тревоги не было")


@router.callback_query(F.data == "panel:camera")
async def cb_camera(cq: CallbackQuery, runtime: Runtime) -> None:
    await cq.message.edit_text(
        "Камеры\nВыберите камеру, чтобы посмотреть состояние и запросить "
        "одноразовый тестовый снимок.",
        reply_markup=_camera_keyboard(runtime),
    )
    await cq.answer()


@router.callback_query(F.data.startswith("camera:view:"))
async def cb_camera_view(cq: CallbackQuery, runtime: Runtime,
                         pipeline: Pipeline | None) -> None:
    try:
        index = int(cq.data.split(":")[2])
        source = runtime.camera_sources()[index]
    except (IndexError, ValueError):
        await cq.answer("Камера уже изменилась. Обновите список.", show_alert=True)
        return
    await cq.message.edit_text(
        _camera_info_text(source, index, pipeline, runtime),
        reply_markup=_camera_detail_keyboard(index),
    )
    await cq.answer()


@router.callback_query(F.data.startswith("camera:conf:"))
async def cb_camera_conf(cq: CallbackQuery, runtime: Runtime) -> None:
    try:
        index = int(cq.data.split(":")[2])
        source = runtime.camera_sources()[index]
    except (IndexError, ValueError):
        await cq.answer("Камера уже изменилась. Обновите список.", show_alert=True)
        return
    await cq.message.edit_text(
        _camera_conf_text(index, source, runtime),
        reply_markup=_camera_conf_keyboard(index, runtime),
    )
    await cq.answer()


async def _refresh_camera_conf(cq: CallbackQuery, runtime: Runtime, index: int) -> None:
    source = runtime.camera_sources()[index]
    try:
        await cq.message.edit_text(
            _camera_conf_text(index, source, runtime),
            reply_markup=_camera_conf_keyboard(index, runtime),
        )
    except Exception:
        pass


@router.callback_query(F.data.startswith("camera:tog:"))
async def cb_camera_toggle(cq: CallbackQuery, runtime: Runtime) -> None:
    try:
        _, _, raw_index, key = cq.data.split(":")
        index = int(raw_index)
        current = runtime.camera_override(index).get(key)
        if key == "notify_enter":
            new_value = not runtime.camera_notify_enter(index)
        elif key == "notify_exit":
            new_value = not runtime.camera_notify_exit(index)
        else:
            new_value = not runtime.camera_enabled(index)
        runtime.set_camera_override(index, key, new_value)
    except (IndexError, ValueError, ConfigurationError) as exc:
        await cq.answer(str(exc) or "Не удалось изменить", show_alert=True)
        return
    del current
    await _refresh_camera_conf(cq, runtime, index)
    await cq.answer("Сохранено")


@router.callback_query(F.data.startswith("camera:mode:"))
async def cb_camera_mode(cq: CallbackQuery, runtime: Runtime) -> None:
    try:
        index = int(cq.data.split(":")[2])
        mode = runtime.camera_detect_mode(index)
        runtime.set_camera_override(
            index, "detect_mode", "presence" if mode == "identity" else "identity",
        )
    except (IndexError, ValueError, ConfigurationError) as exc:
        await cq.answer(str(exc) or "Не удалось изменить", show_alert=True)
        return
    await _refresh_camera_conf(cq, runtime, index)
    await cq.answer("Сохранено")


@router.callback_query(F.data.startswith("camera:reset:"))
async def cb_camera_reset(cq: CallbackQuery, runtime: Runtime) -> None:
    try:
        index = int(cq.data.split(":")[2])
    except (IndexError, ValueError):
        await cq.answer("Камера уже изменилась", show_alert=True)
        return
    for key in ("cooldown_seconds", "notify_enter", "notify_exit", "enabled", "quiet_hours", "detect_mode"):
        runtime.clear_camera_override(index, key)
    await _refresh_camera_conf(cq, runtime, index)
    await cq.answer("Настройки сброшены")


@router.callback_query(F.data.startswith("camera:num:"))
async def cb_camera_num(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    try:
        _, _, raw_index, key = cq.data.split(":")
        index = int(raw_index)
        runtime.camera_sources()[index]
    except (IndexError, ValueError):
        await cq.answer("Камера уже изменилась", show_alert=True)
        return
    await state.set_state(NumberForm.value)
    await state.update_data(
        field="camera_cooldown", camera_index=index, camera_key=key, lo=0.0, hi=86400.0,
    )
    await cq.message.answer(
        f"Камера {index + 1}: введите кулдаун в секундах (0..86400). «отмена» - выйти."
    )
    await cq.answer()


@router.callback_query(F.data.startswith("camera:rename:"))
async def cb_camera_rename(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    try:
        index = int(cq.data.split(":")[2])
        runtime.camera_sources()[index]
    except (IndexError, ValueError):
        await cq.answer("Камера уже изменилась", show_alert=True)
        return
    await state.set_state(CameraForm.name)
    await state.update_data(camera_index=index)
    await cq.message.answer(
        f"Камера {index + 1}: введите имя (до 40 символов). "
        "Пустое сообщение или «-» сбросит имя. «отмена» - выйти."
    )
    await cq.answer()


@router.callback_query(F.data.startswith("camera:test:"))
async def cb_camera_test(cq: CallbackQuery, runtime: Runtime,
                         pipeline: Pipeline | None) -> None:
    if pipeline is None:
        await cq.answer("Пайплайн выключен - снимок недоступен.", show_alert=True)
        return
    try:
        index = int(cq.data.split(":")[2])
        source = runtime.camera_sources()[index]
    except (IndexError, ValueError):
        await cq.answer("Камера уже изменилась. Обновите список.", show_alert=True)
        return
    jpeg = await asyncio.to_thread(pipeline.snapshot, source)
    if jpeg is None:
        await cq.answer("Кадр ещё не получен. Подождите несколько секунд.", show_alert=True)
        return
    sent = await cq.message.answer_photo(
        BufferedInputFile(jpeg, filename=f"camera-{index + 1}-test.jpg"),
        caption=(
            f"Тестовый снимок: камера {index + 1}. "
            f"Будет удалён через {TEST_SNAPSHOT_TTL_SECONDS} сек."
        ),
    )
    # JPEG создаётся только в памяти. После отправки ссылка освобождается, а
    # сообщение в Telegram удаляется отдельной задачей.
    if sent is not None:
        asyncio.create_task(_delete_test_snapshot(sent))
    await cq.answer("Тестовый снимок отправлен")


@router.callback_query(F.data == "camera:edit")
async def cb_camera_edit(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(CameraForm.source)
    await cq.message.answer(
        "Настройка камер\n\n"
        f"Текущий список: {runtime.camera_source}\n\n"
        "Отправьте один источник или до четырёх через |.\n"
        "• `0` - встроенная или USB-камера\n"
        "• `rtsp://host/stream` - поток IP-камеры\n"
        "• `http://host/video` - MJPEG-поток\n\n"
        "Для сетевой камеры её хост должен быть в DOTEYE_ALLOWED_URL_HOSTS. "
        "Нажмите «Инструкция», если не знаете адрес потока.",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Инструкция", callback_data="camera:help")],
            [InlineKeyboardButton(text="Отмена", callback_data="camera:cancel")],
        ]),
    )
    await cq.answer()


@router.callback_query(F.data == "camera:help")
async def cb_camera_help(cq: CallbackQuery) -> None:
    await cq.message.answer(
        "Подсказки по подключению\n\n"
        "1. Для USB-камеры начните с `0`; если камер несколько, попробуйте `1`.\n"
        "2. Для IP-камеры используйте адрес RTSP или MJPEG из её приложения/инструкции. "
        "Камера и DotEye должны быть в одной локальной сети.\n"
        "3. Добавьте только имя или IP хоста камеры в DOTEYE_ALLOWED_URL_HOSTS, затем "
        "повторите ввод.\n"
        "4. После сохранения откройте камеру в этом меню и нажмите «Тестовый снимок».\n\n"
        "Если кадра нет, проверьте питание камеры, адрес потока и доступность хоста из "
        "устройства с DotEye. Не публикуйте поток камеры в интернет.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Ввести источник", callback_data="camera:edit")],
            [InlineKeyboardButton(text="◀ К камерам", callback_data="panel:camera")],
        ]),
    )
    await cq.answer()


@router.callback_query(F.data == "camera:cancel")
async def cb_camera_cancel(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    await state.clear()
    await cq.message.answer("Настройка камеры отменена.", reply_markup=_camera_keyboard(runtime))
    await cq.answer()


@router.callback_query(F.data == "panel:remote")
async def cb_remote(cq: CallbackQuery, runtime: Runtime, state: FSMContext) -> None:
    if runtime.remote_processing:
        runtime.remote_processing = False
        await cq.message.edit_text("Тонкая настройка:", reply_markup=_tune_keyboard(runtime))
        await cq.answer("Remote выкл")
        return
    if not runtime.remote_url:
        await state.set_state(TextForm.value)
        await state.update_data(kind="remote_url")
        await cq.message.answer("URL remote-сервера, например http://192.168.0.10:8099")
        await cq.answer()
        return
    runtime.remote_processing = True
    await cq.message.edit_text("Тонкая настройка:", reply_markup=_tune_keyboard(runtime))
    await cq.answer("Remote вкл")


@router.callback_query(F.data == "panel:quiet")
async def cb_quiet(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(TextForm.value)
    await state.update_data(kind="quiet")
    await cq.message.answer(
        f"Сейчас: {runtime.quiet_hours or 'выкл'}\n"
        "Интервал HH:MM-HH:MM (например 22:00-07:00) или 0 чтобы выключить."
    )
    await cq.answer()


@router.callback_query(F.data == "panel:confidence")
async def cb_confidence(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(NumberForm.value)
    await state.update_data(field="min_confidence", lo=0.0, hi=1.0)
    await cq.message.answer(f"Порог детекции сейчас {runtime.min_confidence}. Число 0..1:")
    await cq.answer()


@router.callback_query(F.data == "panel:cooldown")
async def cb_cooldown_panel(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(NumberForm.value)
    await state.update_data(field="cooldown_seconds", lo=0.0, hi=86400.0)
    await cq.message.answer(f"Кулдаун сейчас {runtime.cooldown_seconds} сек:")
    await cq.answer()


@router.callback_query(F.data == "panel:interval")
async def cb_interval(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(NumberForm.value)
    await state.update_data(field="detection_interval", lo=0.05, hi=30.0)
    await cq.message.answer(f"Интервал сейчас {runtime.detection_interval} сек:")
    await cq.answer()


@router.callback_query(F.data == "panel:face")
async def cb_face(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(NumberForm.value)
    await state.update_data(field="face_threshold", lo=0.05, hi=1.0)
    await cq.message.answer(f"Порог лица сейчас {runtime.face_threshold}. Меньше - строже:")
    await cq.answer()


@router.callback_query(F.data == "panel:nms_iou")
async def cb_nms_iou(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(NumberForm.value)
    await state.update_data(field="nms_iou", lo=0.0, hi=1.0)
    await cq.message.answer(
        f"NMS IoU сейчас {runtime.nms_iou:g}. Это порог объединения "
        "перекрывающихся боксов (меньше = строже, 0..1):"
    )
    await cq.answer()


@router.callback_query(F.data == "panel:person_min_area")
async def cb_person_min_area(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(NumberForm.value)
    await state.update_data(field="person_min_area", lo=0.0, hi=0.5)
    await cq.message.answer(
        f"Минимальная площадь человека сейчас {runtime.person_min_area:g} "
        "(доля кадра, 0..0.5). Мелкие «квадратики» меньше этой площади отбрасываются:"
    )
    await cq.answer()


@router.callback_query(F.data == "panel:zones")
async def cb_zones(cq: CallbackQuery, runtime: Runtime) -> None:
    zones = parse_zones(runtime.zones_json)
    text = "Зоны (нажми чтобы удалить):" if zones else "Зон нет - весь кадр."
    await cq.message.edit_text(text, reply_markup=_zones_keyboard(runtime))
    await cq.answer()


@router.callback_query(F.data == "zone:add")
async def cb_zone_add(cq: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(TextForm.value)
    await state.update_data(kind="zone_name")
    await cq.message.answer("Имя зоны, например «дверь»:")
    await cq.answer()


@router.callback_query(F.data.startswith("zone:del:"))
async def cb_zone_del(cq: CallbackQuery, runtime: Runtime) -> None:
    idx = int(cq.data.split(":")[2])
    zones = parse_zones(runtime.zones_json)
    if 0 <= idx < len(zones):
        zones.pop(idx)
        runtime.zones_json = dump_zones(zones)
    text = "Зоны (нажми чтобы удалить):" if zones else "Зон нет - весь кадр."
    await cq.message.edit_text(text, reply_markup=_zones_keyboard(runtime))
    await cq.answer("Удалено")


@router.message(Command("mode"))
async def cmd_mode(message: Message, runtime: Runtime) -> None:
    new = "identity" if runtime.detect_mode == "presence" else "presence"
    runtime.detect_mode = new
    await message.answer(f"Режим переключён на: {new}")


@router.message(Command("detector"))
async def cmd_detector(message: Message, runtime: Runtime, command: Command) -> None:
    value = (command.args or "").strip().lower()
    if not value:
        await message.answer(
            f"Текущий бэкенд: {runtime.detector_backend}\n"
            "Сменить: /detector auto | yolo | yunet | motion"
        )
        return
    if value not in DETECTORS:
        await message.answer("Допустимо: auto | yolo | yunet | motion")
        return
    runtime.detector_backend = value
    await message.answer(f"Детектор: {value}")


@router.message(Command("device"))
async def cmd_device(message: Message, runtime: Runtime, command: Command) -> None:
    value = (command.args or "").strip().lower()
    if not value:
        await message.answer(
            f"Текущее устройство: {runtime.device}\n"
            "Сменить: /device cpu | cuda | mps\n"
            "cuda - GPU NVIDIA, mps - Apple Silicon, cpu - всегда доступен."
        )
        return
    if value not in DEVICES:
        await message.answer("Допустимо: cpu | cuda | mps")
        return
    runtime.device = value
    await message.answer(
        f"Устройство: {value}\nДетектор пересоберётся на ходу."
    )


@router.message(Command("model"))
async def cmd_model(message: Message, runtime: Runtime, command: Command) -> None:
    value = (command.args or "").strip()
    if not value:
        await message.answer(
            f"Текущая модель: {runtime.model_path}\n"
            "Выбрать: /model <имя>, например /model yolov8s.pt\n\n"
            + models.comparison(),
            parse_mode="Markdown",
        )
        return
    info = models.get_model(value)
    if info is None:
        await message.answer(
            f"«{value}» нет в каталоге. Доступные: "
            + ", ".join(models.YOLO_MODELS)
        )
        return
    runtime.model_path = value
    if runtime.detector_backend != "yolo":
        runtime.detector_backend = "yolo"
    await message.answer(
        models.describe(value) + "\n\nДетектор пересоберётся на ходу.",
        parse_mode="Markdown",
    )


@router.message(Command("people"))
async def cmd_people(message: Message, storage: Storage) -> None:
    rows = storage.list_people()
    if not rows:
        await message.answer("Список людей пуст.")
        return
    await message.answer("Люди:", reply_markup=_people_keyboard(storage))


@router.message(Command("camera"))
async def cmd_camera(message: Message, runtime: Runtime) -> None:
    await message.answer(
        "Камеры\nВыберите камеру или измените список источников.",
        reply_markup=_camera_keyboard(runtime),
    )


@router.message(CameraForm.source)
async def proc_camera_source(message: Message, state: FSMContext, runtime: Runtime) -> None:
    if _cancel_requested(message):
        await state.clear()
        await message.answer("Отменено.")
        return
    try:
        runtime.camera_source = (message.text or "0").strip()
    except ConfigurationError as exc:
        await message.answer(f"Источник отклонён: {exc}")
        return
    await state.clear()
    await message.answer(
        "Список камер сохранён. Пайплайн применит изменение на следующем кадре. "
        "Откройте камеру через минуту и запросите тестовый снимок.",
        reply_markup=_camera_keyboard(runtime),
    )


@router.message(CameraForm.name)
async def proc_camera_name(message: Message, state: FSMContext, runtime: Runtime) -> None:
    if _cancel_requested(message):
        await state.clear()
        await message.answer("Отменено.")
        return
    data = await state.get_data()
    index = int(data.get("camera_index", -1))
    raw = (message.text or "").strip()
    try:
        runtime.set_camera_name(index, "" if raw in ("-", "сброс") else raw)
    except ConfigurationError as exc:
        await message.answer(f"Имя отклонено: {exc}")
        return
    await state.clear()
    label = runtime.camera_label(index)
    await message.answer(
        f"Имя сохранено: камера {index + 1} - {label}",
        reply_markup=_camera_keyboard(runtime),
    )


@router.message(Command("confidence"))
async def cmd_confidence(message: Message, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(NumberForm.confidence)
    await message.answer(f"Текущий порог: {runtime.min_confidence}. Введи число 0..1.")


@router.message(NumberForm.confidence)
async def proc_confidence(message: Message, state: FSMContext, runtime: Runtime) -> None:
    if _cancel_requested(message):
        await state.clear()
        await message.answer("Отменено.")
        return
    try:
        value = float((message.text or "").replace(",", "."))
    except ValueError:
        await message.answer("Нужно число, например 0.5")
        return
    runtime.min_confidence = max(0.0, min(1.0, value))
    await state.clear()
    await message.answer(f"Порог детекции: {runtime.min_confidence}")


@router.message(Command("cooldown"))
async def cmd_cooldown(message: Message, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(NumberForm.cooldown)
    await message.answer(f"Текущий кулдаун: {runtime.cooldown_seconds} сек. Введи число.")


@router.message(NumberForm.cooldown)
async def proc_cooldown(message: Message, state: FSMContext, runtime: Runtime) -> None:
    if _cancel_requested(message):
        await state.clear()
        await message.answer("Отменено.")
        return
    try:
        value = float((message.text or "").replace(",", "."))
    except ValueError:
        await message.answer("Нужно число, например 30")
        return
    runtime.cooldown_seconds = max(0.0, value)
    await state.clear()
    await message.answer(f"Кулдаун: {runtime.cooldown_seconds} сек")


@router.message(NumberForm.value)
async def proc_number(message: Message, state: FSMContext, runtime: Runtime) -> None:
    if _cancel_requested(message):
        await state.clear()
        await message.answer("Отменено.")
        return
    try:
        value = float((message.text or "").replace(",", "."))
    except ValueError:
        await message.answer("Нужно число. «отмена» - выйти.")
        return
    data = await state.get_data()
    field = data.get("field", "min_confidence")
    lo = float(data.get("lo", 0.0))
    hi = float(data.get("hi", 1.0))
    value = max(lo, min(hi, value))
    if field == "voice_volume_pct":
        runtime.voice_volume = value / 100.0
        shown = f"voice_volume = {runtime.voice_volume:g}"
    elif field == "camera_cooldown":
        index = int(data.get("camera_index", -1))
        key = str(data.get("camera_key", "cooldown_seconds"))
        runtime.set_camera_override(index, key, value)
        shown = f"камера {index + 1}: кулдаун = {value:g} сек"
        await state.clear()
        await message.answer(shown, reply_markup=_camera_conf_keyboard(index, runtime))
        return
    else:
        setattr(runtime, field, value)
        shown = f"{field} = {value}"
    await state.clear()
    await message.answer(shown)


@router.message(TextForm.value)
async def proc_text(
    message: Message, state: FSMContext, runtime: Runtime,
    storage: Storage, crypto: Crypto | None, recognizer: Recognizer | None,
    pipeline: Pipeline | None, voice: VoiceEngine | None = None,
) -> None:
    if _cancel_requested(message):
        await state.clear()
        await message.answer("Отменено.")
        return
    text = (message.text or "").strip()
    data = await state.get_data()
    kind = data.get("kind")
    if kind == "quiet":
        if text.lower() in ("0", "off", "выкл", "-"):
            runtime.quiet_hours = ""
            await message.answer("Тихие часы выключены.")
        else:
            try:
                runtime.quiet_hours = text
            except ConfigurationError as exc:
                await message.answer(f"Тихие часы отклонены: {exc}")
                return
            await message.answer(f"Тихие часы: {text}")
        await state.clear()
        return
    if kind == "remote_url":
        try:
            runtime.remote_url = text
            runtime.remote_processing = True
        except ConfigurationError as exc:
            await message.answer(f"Remote URL отклонён: {exc}")
            return
        await state.clear()
        await message.answer(f"Remote: {text}")
        return
    if kind == "zone_name":
        await state.update_data(zone_name=text, kind="zone_rect")
        await message.answer("Четыре числа 0..1 через пробел: x1 y1 x2 y2\nнапример: 0 0.2 0.5 1")
        return
    if kind == "zone_rect":
        parts = text.replace(",", " ").split()
        if len(parts) != 4:
            await message.answer("Нужно 4 числа: x1 y1 x2 y2")
            return
        try:
            x1, y1, x2, y2 = (float(p) for p in parts)
        except ValueError:
            await message.answer("Числа, например 0 0 0.4 1")
            return
        zones = parse_zones(runtime.zones_json)
        zones.append(Zone(data.get("zone_name") or "зона", x1, y1, x2, y2).clamp())
        runtime.zones_json = dump_zones(zones)
        await state.clear()
        await message.answer(f"Зона «{zones[-1].name}» добавлена.")
        return
    if kind == "assign_name":
        event_id = int(data["event_id"])
        if not text:
            await message.answer("Имя не может быть пустым.")
            return
        pid = storage.upsert_person(text, None)
        msg = _assign_event(event_id, pid, storage, crypto, _face(recognizer, pipeline))
        await state.clear()
        await message.answer(msg)
        return
    if kind == "event_note":
        event_id = int(data["event_id"])
        if len(text) > 500:
            await message.answer("Не более 500 символов.")
            return
        if crypto is None:
            await message.answer("Заметка недоступна: ключ шифрования не задан.")
            return
        if storage.get_event(event_id) is None:
            await state.clear()
            await message.answer("Событие не найдено.")
            return
        storage.update_event_note(event_id, crypto.encrypt(text.encode("utf-8")) if text else None)
        await state.clear()
        await message.answer("Заметка сохранена." if text else "Заметка удалена.")
        return
    if kind == "say":
        if voice is None:
            await state.clear()
            await message.answer("Голосовой движок не создан.")
            return
        if not voice.speech_available or not voice.audio_available:
            await message.answer(
                "Озвучка недоступна: нет синтезатора речи или аудиовыхода. "
                "Открой «Статус» в панели и проверь строку «Голос»."
            )
            return
        if not voice.announce(text):
            await message.answer(
                "Озвучка недоступна: нет синтезатора речи или аудиовыхода. "
                "Открой «Статус» в панели и проверь строку «Голос»."
            )
            return
        await state.clear()
        await message.answer(f"Озвучиваю: {text}")
        return
    if isinstance(kind, str) and kind.startswith("phrase:"):
        key = kind.split(":", 1)[1]
        try:
            runtime.set_voice_phrase(key, text)
        except ConfigurationError as exc:
            await message.answer(f"Фраза отклонена: {exc}")
            return
        await state.clear()
        title = PHRASE_TITLES.get(key, key)
        await message.answer(
            f"{title} сохранена.",
            reply_markup=_phrase_keyboard(runtime),
        )
        return
    await state.clear()
    await message.answer("Не понял ввод.")


@router.message(Command("add"))
async def cmd_add(message: Message, state: FSMContext) -> None:
    await state.set_state(AddPersonForm.name)
    await message.answer("Имя человека (или «отмена»):")


@router.message(AddPersonForm.name)
async def proc_add_name(message: Message, state: FSMContext) -> None:
    if _cancel_requested(message):
        await state.clear()
        await message.answer("Отменено.")
        return
    name = (message.text or "").strip()
    if not name:
        await message.answer("Имя не может быть пустым. Попробуй ещё.")
        return
    await state.update_data(name=name)
    await state.set_state(AddPersonForm.photo)
    await message.answer(
        f"Пришли фото лица для «{name}» или покажи лицо в камеру и нажми кнопку.",
        reply_markup=_capture_face_keyboard(),
    )


@router.message(AddPersonForm.photo, F.photo)
async def proc_add_photo(
    message: Message, state: FSMContext, storage: Storage,
    crypto: Crypto | None, recognizer: Recognizer | None, pipeline: Pipeline | None,
) -> None:
    if crypto is None:
        await message.answer("DOTEYE_CRYPTO_KEY не задан, сохранение фото невозможно.")
        return
    data = await state.get_data()
    name = data.get("name", "")
    face = _face(recognizer, pipeline)
    if face is None or not face.available():
        await message.answer(
            "Распознавание лиц недоступно. Установи зависимости распознавания "
            "и перезапусти бота, затем повтори добавление."
        )
        return

    photo = message.photo[-1]
    file = await message.bot.download(photo)
    jpeg = file.read()
    embedding = _embed_jpeg(jpeg, face, None)
    if embedding is None:
        await message.answer("Лицо не найдено на фото, пришли другое.")
        return

    storage.upsert_person(name, crypto.encrypt(embedding))
    await state.clear()
    await message.answer(f"«{name}» добавлен с фото.")


@router.message(AddPersonForm.photo)
async def proc_add_photo_invalid(message: Message, state: FSMContext) -> None:
    if _cancel_requested(message):
        await state.clear()
        await message.answer("Отменено.")
        return
    await message.answer("Нужно фото (отправь изображение). «отмена» - выйти.")


@router.message(PhotoWaitForm.photo, F.photo)
async def proc_extra_photo(
    message: Message, state: FSMContext, storage: Storage,
    crypto: Crypto | None, recognizer: Recognizer | None, pipeline: Pipeline | None,
) -> None:
    data = await state.get_data()
    person_id = int(data["person_id"])
    person = storage.get_person_by_id(person_id)
    if person is None:
        await state.clear()
        await message.answer("Человек уже удалён.")
        return
    face = _face(recognizer, pipeline)
    if crypto is None or face is None or not face.available():
        await state.clear()
        await message.answer("Распознавание недоступно, фото не сохранено.")
        return
    photo = message.photo[-1]
    file = await message.bot.download(photo)
    embedding = _embed_jpeg(file.read(), face, None)
    if embedding is None:
        await message.answer("Лицо не найдено, пришли другое.")
        return
    storage.add_embedding(person_id, crypto.encrypt(embedding))
    await state.clear()
    n = len(storage.embeddings_for(person_id))
    await message.answer(f"«{person['name']}»: эталонов теперь {n}.")


@router.message(PhotoWaitForm.photo)
async def proc_extra_photo_invalid(message: Message, state: FSMContext) -> None:
    if _cancel_requested(message):
        await state.clear()
        await message.answer("Отменено.")
        return
    await message.answer("Нужно фото. «отмена» - выйти.")


@router.message(Command("remove"))
async def cmd_remove(message: Message, state: FSMContext) -> None:
    await state.set_state(RemovePersonForm.name)
    await message.answer("Имя человека для удаления (или «отмена»):")


@router.message(RemovePersonForm.name)
async def proc_remove(message: Message, state: FSMContext, storage: Storage) -> None:
    if _cancel_requested(message):
        await state.clear()
        await message.answer("Отменено.")
        return
    name = (message.text or "").strip()
    if storage.delete_person(name):
        await message.answer(f"Удалён: {name}")
    else:
        await message.answer(f"Не найден: {name}")
    await state.clear()


@router.message(Command("events"))
async def cmd_events(
    message: Message, storage: Storage, crypto: Crypto | None, runtime: Runtime
) -> None:
    await _send_events_page(message, storage, crypto, runtime, 0)


@router.message(Command("audit"))
async def cmd_audit(message: Message, storage: Storage, crypto: Crypto | None) -> None:
    await _send_audit_log(message, storage, crypto)


@router.message(Command("say"))
async def cmd_say(message: Message, command: Command, state: FSMContext,
                  voice: VoiceEngine | None = None) -> None:
    text = (command.args or "").strip()
    if not text:
        await state.set_state(TextForm.value)
        await state.update_data(kind="say")
        await message.answer("Текст для озвучки (или «отмена»):")
        return
    if voice is None:
        await message.answer("Голосовой движок не создан.")
        return
    if not voice.announce(text):
        await message.answer("Пустой текст.")
        return
    await message.answer(f"Озвучиваю: {text}")


@router.message(Command("alarm"))
async def cmd_alarm(message: Message, command: Command,
                    voice: VoiceEngine | None = None) -> None:
    if voice is None:
        await message.answer("Голосовой движок не создан.")
        return
    arg = (command.args or "").strip().lower()
    if arg in ("off", "stop", "0", "выкл", "снять"):
        notice = voice.dismiss("manual")
        await message.answer("Тревога снята." if notice else "Тревоги не было.")
        return
    phrase = (command.args or "").strip() or None
    voice.trigger_alarm(phrase)
    await message.answer("Тревога запущена.")


@router.message()
async def fallback(message: Message) -> None:
    await message.answer("Не понял. /help - список команд. /cancel - отменить ввод.")


async def run_bot(
    settings: Settings,
    storage: Storage,
    runtime: Runtime,
    crypto: Crypto | None,
    recognizer: Recognizer | None,
    pipeline: Pipeline | None,
    voice: VoiceEngine | None = None,
) -> None:
    if not settings.has_token:
        raise RuntimeError("DOTEYE_BOT_TOKEN не задан")

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher(storage=MemoryStorage())
    di = AccessMiddleware(
        settings, storage, runtime, crypto, recognizer, pipeline, voice
    )
    dp.message.middleware(di)
    dp.callback_query.middleware(di)
    dp.include_router(router)

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


async def notify_startup(
    settings: Settings, runtime: Runtime, recognizer: Recognizer | None,
    pipeline: Pipeline | None,
) -> None:
    """Сообщить админам, что бот запущен, и прислать текущие настройки."""
    if not settings.has_admins:
        return
    bot = Bot(token=settings.bot_token)
    text = "DotEye запущен.\n\n" + _status_text(runtime, recognizer, pipeline)
    try:
        for admin in settings.admin_ids:
            try:
                await bot.send_message(admin, text, reply_markup=_panel_keyboard(runtime))
            except Exception as exc:  # noqa: BLE001
                print(f"[notify] admin {admin}: {exc}")
    finally:
        await bot.session.close()


def queue_notifications(
    settings: Settings, storage: Storage, crypto: Crypto, event: Any,
) -> None:
    """Сохранить доставки каждому админу до обращения к Telegram API.

    Запись в SQLite происходит раньше отправки, поэтому рестарт процесса и
    временная ошибка Telegram не теряют срабатывание камеры.
    """
    if not settings.has_admins:
        return
    caption = event.caption or (
        f"DotEye: {event.person_name or 'неизвестный'}"
        + (f" ({event.confidence:.2f})" if event.confidence else "")
    )
    unknown_enter = bool(
        event.event_id
        and event.person_name is None
        and event.event_type == "enter"
        and event.jpeg
    )
    show_dismiss = unknown_enter or event.event_type == "alarm"
    payload = json.dumps(
        {
            "caption": caption,
            "event_id": event.event_id,
            "unknown_enter": unknown_enter,
            "show_dismiss": show_dismiss,
            "jpeg_encrypted": bool(event.jpeg),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if event.event_id is not None:
        event_key = f"event:{event.event_id}"
    else:
        # Alert не имеет event_id. Хеш создаёт одинаковый ключ только для
        # одного и того же срабатывания в пределах временной метки pipeline.
        digest = hashlib.sha256(
            f"{event.event_type}\0{caption}\0{event.detected_at:.6f}".encode()
        ).hexdigest()
        event_key = f"alert:{digest}"
    for admin_id in settings.admin_ids:
        storage.enqueue_notification(
            f"{event_key}:admin:{admin_id}",
            event.event_id,
            admin_id,
            payload,
            crypto.encrypt(event.jpeg) if event.jpeg else None,
        )


def _retry_at(attempts: int) -> str:
    """Экспоненциальная задержка 1s..1h без зависимости от event loop."""
    seconds = min(OUTBOX_MAX_RETRY_SECONDS, 2 ** min(max(0, attempts), 12))
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


async def send_notifications(
    settings: Settings,
    storage: Storage,
    crypto: Crypto | None,
    *,
    poll_seconds: float = OUTBOX_POLL_SECONDS,
) -> None:
    """Доставлять сохранённые уведомления с повтором после ошибок Telegram."""
    if not settings.has_admins:
        return
    bot = Bot(token=settings.bot_token)
    try:
        while True:
            rows = storage.claim_due_notifications(limit=OUTBOX_BATCH_SIZE)
            if not rows:
                await asyncio.sleep(poll_seconds)
                continue
            for row in rows:
                try:
                    payload = json.loads(str(row["payload"]))
                    caption = str(payload["caption"])
                    event_id = payload.get("event_id")
                    markup = None
                    rows: list[list[InlineKeyboardButton]] = []
                    if payload.get("unknown_enter") and isinstance(event_id, int):
                        rows.append([
                            InlineKeyboardButton(
                                text="Это кто?",
                                callback_data=f"event:who:{event_id}",
                            )
                        ])
                    if isinstance(event_id, int):
                        rows.append([
                            InlineKeyboardButton(
                                text="Добавить заметку",
                                callback_data=f"event:note:{event_id}",
                            )
                        ])
                    if payload.get("show_dismiss"):
                        rows.append([
                            InlineKeyboardButton(
                                text="Снять тревогу",
                                callback_data="voice:dismiss",
                            )
                        ])
                    rows.append([
                        InlineKeyboardButton(
                            text="Вернуться в панель",
                            callback_data="panel:open",
                        )
                    ])
                    if rows:
                        markup = InlineKeyboardMarkup(inline_keyboard=rows)
                    jpeg = row["jpeg"]
                    if jpeg:
                        if crypto is None:
                            raise RuntimeError("crypto недоступен для уведомления с фото")
                        if not payload.get("jpeg_encrypted"):
                            raise RuntimeError("outbox содержит незашифрованное фото")
                        photo = BufferedInputFile(crypto.decrypt(bytes(jpeg)), filename="event.jpg")
                        await bot.send_photo(int(row["admin_id"]), photo, caption=caption, reply_markup=markup)
                    else:
                        await bot.send_message(int(row["admin_id"]), caption, reply_markup=markup)
                except Exception as exc:  # noqa: BLE001
                    storage.retry_notification(
                        int(row["id"]), _retry_at(int(row["attempts"])), str(exc),
                    )
                    print(f"[notify] admin {row['admin_id']}: {exc}")
                else:
                    storage.mark_notification_sent(int(row["id"]))
    finally:
        await bot.session.close()
