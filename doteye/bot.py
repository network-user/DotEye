"""Telegram-бот на aiogram 3.

Настройка целиком внутри чата: охрана, тихие часы, камеры, зоны,
списки людей и событий, «это кто?» на неизвестном входе.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

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
    Message,
    TelegramObject,
)

from doteye import models
from doteye.annotate import crop_box
from doteye.config import Settings
from doteye.crypto import Crypto
from doteye.pipeline import Pipeline
from doteye.recognizer import Recognizer
from doteye.runtime import Runtime
from doteye.storage import Storage
from doteye.zones import Zone, dump_zones, parse_zones

HELP = (
    "DotEye - кто зашёл в комнату.\n\n"
    "/panel - админ-панель (кнопки)\n"
    "/status - текущие настройки\n"
    "/mode - presence <-> identity\n"
    "/camera - источник кадров (несколько через | )\n"
    "/detector - auto | yolo | yunet | motion\n"
    "/device - cpu | cuda | mps\n"
    "/model - выбрать YOLO-модель\n"
    "/confidence - порог детекции (0..1)\n"
    "/cooldown - пауза повторного входа, сек\n"
    "/people - известные люди\n"
    "/add - добавить человека (имя + фото)\n"
    "/remove - удалить человека\n"
    "/events - последние события\n"
    "/cancel - отменить текущий ввод\n"
    "/help - эта справка"
)

DETECTORS = ["auto", "yolo", "yunet", "motion"]
DEVICES = ["cpu", "cuda", "mps"]
EVENTS_PAGE = 3


class CameraForm(StatesGroup):
    source = State()


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


class AccessMiddleware(BaseMiddleware):
    """Пускает дальше только админов и прокидывает зависимости в хендлеры."""

    def __init__(self, settings: Settings, storage: Storage, runtime: Runtime,
                 crypto: Crypto | None, recognizer: Recognizer | None,
                 pipeline: Pipeline | None) -> None:
        self._settings = settings
        self._storage = storage
        self._runtime = runtime
        self._crypto = crypto
        self._recognizer = recognizer
        self._pipeline = pipeline

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

        if isinstance(event, Message) and not _is_admin(event, self._settings):
            await event.answer("Нет доступа.")
            return None
        if isinstance(event, CallbackQuery) and not _is_admin_user(
            event.from_user.id if event.from_user else None, self._settings
        ):
            await event.answer("Нет доступа.", show_alert=True)
            return None
        return await handler(event, data)


router = Router()


def _panel_keyboard(runtime: Runtime) -> InlineKeyboardMarkup:
    mode_next = "identity" if runtime.detect_mode == "presence" else "presence"
    det_next = _next_item(DETECTORS, runtime.detector_backend)
    arm = "вкл" if runtime.armed else "выкл"
    quiet = runtime.quiet_hours or "выкл"
    exit_s = "вкл" if runtime.notify_exit else "выкл"
    remote = "вкл" if runtime.remote_processing else "выкл"
    rows = [
        [
            InlineKeyboardButton(text=f"Охрана: {arm}", callback_data="panel:arm"),
            InlineKeyboardButton(
                text=f"Режим: {runtime.detect_mode} -> {mode_next}",
                callback_data="panel:mode",
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"Детектор: {runtime.detector_backend} (след. {det_next})",
                callback_data="panel:detector",
            ),
            InlineKeyboardButton(
                text=f"Устройство: {runtime.device}", callback_data="panel:device"
            ),
        ],
        [
            InlineKeyboardButton(text="YOLO-модель", callback_data="panel:model"),
            InlineKeyboardButton(text="Различия моделей", callback_data="panel:model_help"),
        ],
        [
            InlineKeyboardButton(text="Превью камеры", callback_data="panel:snapshot"),
            InlineKeyboardButton(text="Здоровье", callback_data="panel:health"),
        ],
        [
            InlineKeyboardButton(text="Люди", callback_data="panel:people"),
            InlineKeyboardButton(text="События", callback_data="panel:events"),
        ],
        [
            InlineKeyboardButton(text="Камера", callback_data="panel:camera"),
            InlineKeyboardButton(text=f"Remote: {remote}", callback_data="panel:remote"),
        ],
        [
            InlineKeyboardButton(
                text=f"Порог {runtime.min_confidence:g}", callback_data="panel:confidence"
            ),
            InlineKeyboardButton(
                text=f"Кулдаун {runtime.cooldown_seconds:g}с", callback_data="panel:cooldown"
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"Интервал {runtime.detection_interval:g}с",
                callback_data="panel:interval",
            ),
            InlineKeyboardButton(
                text=f"Лицо {runtime.face_threshold:g}", callback_data="panel:face"
            ),
        ],
        [
            InlineKeyboardButton(text=f"Тихие: {quiet}", callback_data="panel:quiet"),
            InlineKeyboardButton(text=f"Выход: {exit_s}", callback_data="panel:exit"),
        ],
        [
            InlineKeyboardButton(text="Зоны", callback_data="panel:zones"),
            InlineKeyboardButton(text="Статус", callback_data="panel:status"),
        ],
        [InlineKeyboardButton(text="Справка", callback_data="panel:help")],
    ]
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


def _people_keyboard(storage: Storage) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for row in storage.list_people():
        n = int(row["face_count"] or 0)
        rows.append([
            InlineKeyboardButton(
                text=f"{row['name']} ({n} фото)",
                callback_data=f"person:view:{row['id']}",
            )
        ])
    rows.append([InlineKeyboardButton(text="Добавить", callback_data="person:add")])
    rows.append([InlineKeyboardButton(text="Назад", callback_data="panel:open")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _person_view_keyboard(person_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Ещё фото", callback_data=f"person:photo:{person_id}")],
        [InlineKeyboardButton(text="Удалить", callback_data=f"person:del:{person_id}")],
        [InlineKeyboardButton(text="Назад", callback_data="panel:people")],
    ])


def _events_nav(page: int, total: int) -> InlineKeyboardMarkup:
    buttons: list[InlineKeyboardButton] = []
    if page > 0:
        buttons.append(InlineKeyboardButton(text="←", callback_data=f"events:page:{page - 1}"))
    if (page + 1) * EVENTS_PAGE < total:
        buttons.append(InlineKeyboardButton(text="→", callback_data=f"events:page:{page + 1}"))
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


def _status_text(runtime: Runtime, recognizer: Recognizer | None,
                 pipeline: Pipeline | None) -> str:
    face = _face(recognizer, pipeline)
    face_s = "on" if face is not None and face.available() else "off/dummy"
    remote = "on" if runtime.remote_processing else "off"
    running = "on" if pipeline is not None and pipeline.running else "off"
    info = models.get_model(runtime.model_path)
    model_title = info.title if info else runtime.model_path
    quiet = runtime.quiet_hours or "выкл"
    lines = [
        "Настройки DotEye:",
        f"Охрана: {'вкл' if runtime.armed else 'выкл'}",
        f"Тихие часы: {quiet}",
        f"Пайплайн: {running}",
        f"Режим: {runtime.detect_mode}",
        f"Камера: {runtime.camera_source}",
        f"Детектор задан: {runtime.detector_backend} (device={runtime.device})",
        f"Модель: {model_title}",
        f"imgsz: {runtime.imgsz}",
        f"Мин. уверенность: {runtime.min_confidence}",
        f"Кулдаун: {runtime.cooldown_seconds} сек",
        f"Интервал: {runtime.detection_interval} сек",
        f"Порог лица: {runtime.face_threshold}",
        f"Распознавание: {face_s}",
        f"Remote: {remote} ({runtime.remote_url or 'нет URL'})",
        f"Уведомлять выход: {'да' if runtime.notify_exit else 'нет'}",
    ]
    if pipeline is not None:
        lines.append("")
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
        who = row["person_name"] or "неизвестный"
        kind = row["event_type"] or "enter"
        verb = "вошёл" if kind == "enter" else ("вышел" if kind == "exit" else kind)
        conf = f", conf={row['confidence']:.2f}" if row["confidence"] else ""
        cam = f"\nкамера {row['camera_source']}" if row["camera_source"] else ""
        zone = f", зона {row['zone']}" if row["zone"] else ""
        caption = f"{row['detected_at']}\n{verb} {who}{conf}{cam}{zone}"
        markup = None
        if row["person_name"] is None and kind == "enter":
            markup = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(
                    text="Это кто?", callback_data=f"event:who:{row['id']}"
                )
            ]])
        if not row["frame"]:
            await message.answer(caption, reply_markup=markup)
            continue
        try:
            jpeg = crypto.decrypt(row["frame"])
        except Exception:
            await message.answer(caption + "\n(кадр не расшифрован)")
            continue
        await message.answer_photo(
            BufferedInputFile(jpeg, filename=f"event_{row['id']}.jpg"),
            caption=caption, reply_markup=markup,
        )


# -- команды ------------------------------------------------------------

@router.message(CommandStart())
async def cmd_start(message: Message, settings: Settings) -> None:
    admin = "админ" if _is_admin(message, settings) else "гость"
    await message.answer(
        f"DotEye на связи ({admin}).\n"
        "Управление - /panel. Подключи камеру, выбери режим.\n"
        "/help - все команды. /cancel - отменить ввод."
    )


@router.message(Command("panel"))
async def cmd_panel(message: Message, runtime: Runtime) -> None:
    await message.answer("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))


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
                     recognizer: Recognizer | None, pipeline: Pipeline | None) -> None:
    await message.answer(_status_text(runtime, recognizer, pipeline))


@router.callback_query(F.data == "panel:open")
async def cb_open(cq: CallbackQuery, runtime: Runtime) -> None:
    await cq.message.edit_text("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))
    await cq.answer()


@router.callback_query(F.data == "panel:status")
async def cb_status(cq: CallbackQuery, runtime: Runtime,
                    recognizer: Recognizer | None, pipeline: Pipeline | None) -> None:
    await cq.message.edit_text(
        _status_text(runtime, recognizer, pipeline), reply_markup=_back_kb()
    )
    await cq.answer()


@router.callback_query(F.data == "panel:health")
async def cb_health(cq: CallbackQuery, pipeline: Pipeline | None, runtime: Runtime,
                    recognizer: Recognizer | None) -> None:
    text = _status_text(runtime, recognizer, pipeline)
    await cq.message.edit_text(text, reply_markup=_back_kb())
    await cq.answer()


@router.callback_query(F.data == "panel:arm")
async def cb_arm(cq: CallbackQuery, runtime: Runtime) -> None:
    runtime.armed = not runtime.armed
    await cq.message.edit_text("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))
    await cq.answer("Охрана вкл" if runtime.armed else "Охрана выкл")


@router.callback_query(F.data == "panel:exit")
async def cb_exit_toggle(cq: CallbackQuery, runtime: Runtime) -> None:
    runtime.notify_exit = not runtime.notify_exit
    await cq.message.edit_text("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))
    await cq.answer("Выход: " + ("вкл" if runtime.notify_exit else "выкл"))


@router.callback_query(F.data == "panel:mode")
async def cb_mode(
    cq: CallbackQuery, runtime: Runtime, recognizer: Recognizer | None = None,
) -> None:
    runtime.detect_mode = "identity" if runtime.detect_mode == "presence" else "presence"
    await cq.message.edit_text("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))
    await cq.answer(f"Режим: {runtime.detect_mode}")


@router.callback_query(F.data == "panel:detector")
async def cb_detector(cq: CallbackQuery, runtime: Runtime) -> None:
    runtime.detector_backend = _next_item(DETECTORS, runtime.detector_backend)
    await cq.message.edit_text("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))
    await cq.answer(f"Детектор: {runtime.detector_backend}")


@router.callback_query(F.data == "panel:device")
async def cb_device(cq: CallbackQuery, runtime: Runtime) -> None:
    runtime.device = _next_item(DEVICES, runtime.device)
    await cq.message.edit_text("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))
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
    rows = storage.list_people()
    text = "Люди:" if rows else "Список людей пуст."
    await cq.message.edit_text(text, reply_markup=_people_keyboard(storage))
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


@router.callback_query(F.data.startswith("person:del:"))
async def cb_person_del(cq: CallbackQuery, storage: Storage) -> None:
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


@router.callback_query(F.data.startswith("events:page:"))
async def cb_events_page(cq: CallbackQuery, storage: Storage,
                         crypto: Crypto | None, runtime: Runtime) -> None:
    page = int(cq.data.split(":")[2])
    await cq.answer()
    await _send_events_page(cq.message, storage, crypto, runtime, page)


@router.callback_query(F.data.startswith("event:who:"))
async def cb_event_who(cq: CallbackQuery, storage: Storage) -> None:
    event_id = int(cq.data.split(":")[2])
    await cq.message.answer(
        "Кто это?", reply_markup=_who_keyboard(event_id, storage)
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


@router.callback_query(F.data == "panel:camera")
async def cb_camera(cq: CallbackQuery, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(CameraForm.source)
    await cq.message.answer(
        f"Сейчас: {runtime.camera_source}\n"
        "Отправь источник: 0, rtsp://..., http://...\n"
        "Несколько камер через |  например  0|rtsp://host/stream\n"
        "«отмена» - выйти."
    )
    await cq.answer()


@router.callback_query(F.data == "panel:remote")
async def cb_remote(cq: CallbackQuery, runtime: Runtime, state: FSMContext) -> None:
    if runtime.remote_processing:
        runtime.remote_processing = False
        await cq.message.edit_text("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))
        await cq.answer("Remote выкл")
        return
    if not runtime.remote_url:
        await state.set_state(TextForm.value)
        await state.update_data(kind="remote_url")
        await cq.message.answer("URL remote-сервера, например http://192.168.0.10:8099")
        await cq.answer()
        return
    runtime.remote_processing = True
    await cq.message.edit_text("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))
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
async def cmd_camera(message: Message, state: FSMContext) -> None:
    await state.set_state(CameraForm.source)
    await message.answer(
        "Отправь источник: 0 (вебка), rtsp://... или http://...\n"
        "Несколько через |  Например: 0|rtsp://192.168.0.8/stream\n"
        "«отмена» - выйти."
    )


@router.message(CameraForm.source)
async def proc_camera_source(message: Message, state: FSMContext, runtime: Runtime) -> None:
    if _cancel_requested(message):
        await state.clear()
        await message.answer("Отменено.")
        return
    runtime.camera_source = (message.text or "0").strip()
    await state.clear()
    await message.answer(f"Источник задан: {runtime.camera_source}")


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
    setattr(runtime, field, value)
    await state.clear()
    await message.answer(f"{field} = {value}")


@router.message(TextForm.value)
async def proc_text(
    message: Message, state: FSMContext, runtime: Runtime,
    storage: Storage, crypto: Crypto | None, recognizer: Recognizer | None,
    pipeline: Pipeline | None,
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
            runtime.quiet_hours = text
            await message.answer(f"Тихие часы: {text}")
        await state.clear()
        return
    if kind == "remote_url":
        runtime.remote_url = text
        runtime.remote_processing = True
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
    await message.answer(f"Теперь пришли фото лица для «{name}».")


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
        storage.upsert_person(name, None)
        await state.clear()
        await message.answer(
            f"«{name}» добавлен без фото: распознавание лиц недоступно "
            "(нет insightface)."
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
) -> None:
    if not settings.has_token:
        raise RuntimeError("DOTEYE_BOT_TOKEN не задан")

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher(storage=MemoryStorage())
    di = AccessMiddleware(settings, storage, runtime, crypto, recognizer, pipeline)
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


async def send_notifications(
    settings: Settings, events: "asyncio.Queue[Any]"
) -> None:
    """Разослать события админам: фото + подпись, у неизвестных кнопка «Это кто?»."""
    if not settings.has_admins:
        return
    bot = Bot(token=settings.bot_token)
    try:
        while True:
            event = await events.get()
            caption = event.caption or (
                f"DotEye: {event.person_name or 'неизвестный'}"
                + (f" ({event.confidence:.2f})" if event.confidence else "")
            )
            markup = None
            if (
                event.event_id
                and event.person_name is None
                and event.event_type == "enter"
                and event.jpeg
            ):
                markup = InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(
                        text="Это кто?",
                        callback_data=f"event:who:{event.event_id}",
                    )
                ]])
            for admin in settings.admin_ids:
                try:
                    if event.jpeg:
                        photo = BufferedInputFile(event.jpeg, filename="event.jpg")
                        await bot.send_photo(admin, photo, caption=caption, reply_markup=markup)
                    else:
                        await bot.send_message(admin, caption)
                except Exception as exc:  # noqa: BLE001
                    print(f"[notify] admin {admin}: {exc}")
    finally:
        await bot.session.close()
