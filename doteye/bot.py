"""Telegram-бот на aiogram 3.

Команды настройки целиком внутри чата:
  /panel      - админ-панель на inline-кнопках
  /start      - приветствие
  /status     - текущие настройки
  /mode       - переключить presence | identity (детекция | распознавание лиц)
  /camera     - задать источник (0 = вебка, rtsp/http адрес)
  /detector   - бэкенд детектора: auto | yolo | yunet | motion
  /device     - устройство инференса: cpu | cuda | mps
  /model      - выбрать YOLO-модель (с описанием мощности)
  /confidence - минимальная уверенность детекции
  /cooldown   - пауза между уведомлениями, сек
  /people     - список известных людей
  /add        - добавить человека (имя), затем прислать фото
  /remove     - удалить человека по имени
  /events     - последние события с кадрами
  /help       - справка

Все настройки хранятся в Storage; пайплайн читает их через runtime.py
и пересобирает камеру/детектор на ходу. Доступ - только для админов
из DOTEYE_ADMIN_IDS.
"""

from __future__ import annotations

import asyncio
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
from doteye.config import Settings
from doteye.crypto import Crypto
from doteye.pipeline import Pipeline
from doteye.recognizer import Recognizer
from doteye.runtime import Runtime
from doteye.storage import Storage

HELP = (
    "DotEye - кто зашёл в комнату.\n\n"
    "/panel - админ-панель (кнопки)\n"
    "/status - текущие настройки\n"
    "/mode - presence <-> identity\n"
    "/camera - источник кадров (0, rtsp://, http://)\n"
    "/detector - auto | yolo | yunet | motion\n"
    "/device - cpu | cuda | mps\n"
    "/model - выбрать YOLO-модель\n"
    "/confidence - порог детекции (0..1)\n"
    "/cooldown - пауза между уведомлениями, сек\n"
    "/people - известные люди\n"
    "/add - добавить человека (имя + фото)\n"
    "/remove - удалить человека\n"
    "/events - последние события\n"
    "/help - эта справка"
)


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


def _is_admin(message: Message, settings: Settings) -> bool:
    if not settings.has_admins:
        return True  # список админов не задан - не блокируем (dev-режим)
    return message.from_user is not None and message.from_user.id in settings.admin_ids


def _is_admin_user(user_id: int | None, settings: Settings) -> bool:
    if not settings.has_admins:
        return True
    return user_id is not None and user_id in settings.admin_ids


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


DEVICES = ["cpu", "cuda", "mps"]


def _panel_keyboard(runtime: Runtime) -> InlineKeyboardMarkup:
    mode_next = "identity" if runtime.detect_mode == "presence" else "presence"
    backend_next = "yolo" if runtime.detector_backend != "yolo" else "auto"
    rows = [
        [
            InlineKeyboardButton(text="Статус", callback_data="panel:status"),
            InlineKeyboardButton(
                text=f"Режим: {runtime.detect_mode} -> {mode_next}",
                callback_data="panel:mode",
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"Детектор: {runtime.detector_backend} -> {backend_next}",
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
            InlineKeyboardButton(text="Люди", callback_data="panel:people"),
        ],
        [
            InlineKeyboardButton(text="События", callback_data="panel:events"),
            InlineKeyboardButton(text="Справка", callback_data="panel:help"),
        ],
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


def _status_text(runtime: Runtime, recognizer: Recognizer | None,
                 pipeline: Pipeline | None) -> str:
    face = "on" if recognizer is not None and recognizer.available() else "off/dummy"
    remote = "on" if runtime.remote_processing else "off"
    running = "on" if pipeline is not None and pipeline.running else "off"
    info = models.get_model(runtime.model_path)
    model_title = info.title if info else runtime.model_path
    return (
        "Настройки DotEye:\n"
        f"Пайплайн: {running}\n"
        f"Режим: {runtime.detect_mode}\n"
        f"Камера: {runtime.camera_source}\n"
        f"Детектор: {runtime.detector_backend} (device={runtime.device})\n"
        f"Модель: {model_title}\n"
        f"Мин. уверенность: {runtime.min_confidence}\n"
        f"Кулдаун: {runtime.cooldown_seconds} сек\n"
        f"Порог лица: {runtime.face_threshold}\n"
        f"Распознавание: {face}\n"
        f"Remote: {remote}"
    )


@router.message(CommandStart())
async def cmd_start(message: Message, settings: Settings) -> None:
    admin = "админ" if _is_admin(message, settings) else "гость"
    await message.answer(
        f"DotEye на связи ({admin}).\n"
        "Управление - /panel. Подключи камеру, выбери режим.\n"
        "/help - все команды."
    )


@router.message(Command("panel"))
async def cmd_panel(message: Message, runtime: Runtime) -> None:
    await message.answer("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP)


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
    await cq.message.answer(_status_text(runtime, recognizer, pipeline))
    await cq.answer()


@router.callback_query(F.data == "panel:mode")
async def cb_mode(cq: CallbackQuery, runtime: Runtime) -> None:
    runtime.detect_mode = "identity" if runtime.detect_mode == "presence" else "presence"
    await cq.message.edit_text("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))
    await cq.answer(f"Режим: {runtime.detect_mode}")


@router.callback_query(F.data == "panel:detector")
async def cb_detector(cq: CallbackQuery, runtime: Runtime) -> None:
    order = ["auto", "yolo", "yunet", "motion"]
    idx = order.index(runtime.detector_backend) if runtime.detector_backend in order else 0
    runtime.detector_backend = order[(idx + 1) % len(order)]
    await cq.message.edit_text("Админ-панель DotEye:", reply_markup=_panel_keyboard(runtime))
    await cq.answer(f"Детектор: {runtime.detector_backend}")


@router.callback_query(F.data == "panel:device")
async def cb_device(cq: CallbackQuery, runtime: Runtime) -> None:
    idx = DEVICES.index(runtime.device) if runtime.device in DEVICES else 0
    runtime.device = DEVICES[(idx + 1) % len(DEVICES)]
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
    if not rows:
        await cq.message.answer("Список людей пуст.")
    else:
        lines = [f"[{'+' if r['has_face'] else '-'}] {r['name']}" for r in rows]
        await cq.message.answer("Люди (+, если есть фото):\n" + "\n".join(lines))
    await cq.answer()


@router.callback_query(F.data == "panel:events")
async def cb_events(cq: CallbackQuery, storage: Storage,
                    crypto: Crypto | None, runtime: Runtime) -> None:
    await cq.answer()
    await _send_events(cq.message, storage, crypto, runtime)


@router.callback_query(F.data == "panel:help")
async def cb_help(cq: CallbackQuery) -> None:
    await cq.message.answer(HELP)
    await cq.answer()


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
    if value not in ("auto", "yolo", "yunet", "motion"):
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
    lines = []
    for r in rows:
        mark = "+" if r["has_face"] else "-"
        lines.append(f"[{mark}] {r['name']}")
    await message.answer("Люди (+, если есть фото):\n" + "\n".join(lines))


@router.message(Command("camera"))
async def cmd_camera(message: Message, state: FSMContext) -> None:
    await state.set_state(CameraForm.source)
    await message.answer("Отправь источник: 0 (вебка), rtsp://... или http://...")


@router.message(CameraForm.source)
async def proc_camera_source(message: Message, state: FSMContext, runtime: Runtime) -> None:
    runtime.camera_source = (message.text or "0").strip()
    await state.clear()
    await message.answer(f"Источник задан: {runtime.camera_source}")


@router.message(Command("confidence"))
async def cmd_confidence(message: Message, state: FSMContext, runtime: Runtime) -> None:
    await state.set_state(NumberForm.confidence)
    await message.answer(f"Текущий порог: {runtime.min_confidence}. Введи число 0..1.")


@router.message(NumberForm.confidence)
async def proc_confidence(message: Message, state: FSMContext, runtime: Runtime) -> None:
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
    try:
        value = float((message.text or "").replace(",", "."))
    except ValueError:
        await message.answer("Нужно число, например 30")
        return
    runtime.cooldown_seconds = max(1.0, value)
    await state.clear()
    await message.answer(f"Кулдаун: {runtime.cooldown_seconds} сек")


@router.message(Command("add"))
async def cmd_add(message: Message, state: FSMContext) -> None:
    await state.set_state(AddPersonForm.name)
    await message.answer("Имя человека:")


@router.message(AddPersonForm.name)
async def proc_add_name(message: Message, state: FSMContext) -> None:
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
    crypto: Crypto | None, recognizer: Recognizer | None,
) -> None:
    if crypto is None:
        await message.answer("DOTEYE_CRYPTO_KEY не задан, сохранение фото невозможно.")
        return
    data = await state.get_data()
    name = data.get("name", "")
    if recognizer is None or not recognizer.available():
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

    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        await message.answer("Не удалось прочитать фото, пришли ещё раз.")
        return
    embedding = recognizer.embed(frame)
    if embedding is None:
        await message.answer("Лицо не найдено на фото, пришли другое.")
        return

    storage.upsert_person(name, crypto.encrypt(embedding))
    await state.clear()
    await message.answer(f"«{name}» добавлен с фото.")


@router.message(AddPersonForm.photo)
async def proc_add_photo_invalid(message: Message) -> None:
    await message.answer("Нужно фото (отправь изображение).")


@router.message(Command("remove"))
async def cmd_remove(message: Message, state: FSMContext) -> None:
    await state.set_state(RemovePersonForm.name)
    await message.answer("Имя человека для удаления:")


@router.message(RemovePersonForm.name)
async def proc_remove(message: Message, state: FSMContext, storage: Storage) -> None:
    name = (message.text or "").strip()
    if storage.delete_person(name):
        await message.answer(f"Удалён: {name}")
    else:
        await message.answer(f"Не найден: {name}")
    await state.clear()


async def _send_events(
    message: Message, storage: Storage, crypto: Crypto | None, runtime: Runtime
) -> None:
    if crypto is None:
        await message.answer("DOTEYE_CRYPTO_KEY не задан, кадры недоступны.")
        return
    rows = storage.recent_events(runtime.events_limit)
    if not rows:
        await message.answer("Событий пока нет.")
        return
    for row in rows:
        who = row["person_name"] or "неизвестный"
        conf = f", conf={row['confidence']:.2f}" if row["confidence"] else ""
        caption = f"{row['detected_at']}\n{who}{conf}"
        if not row["frame"]:
            await message.answer(caption)
            continue
        try:
            jpeg = crypto.decrypt(row["frame"])
        except Exception:
            await message.answer(caption + "\n(кадр не расшифрован)")
            continue
        await message.answer_photo(
            BufferedInputFile(jpeg, filename=f"event_{row['id']}.jpg"), caption=caption
        )


@router.message(Command("events"))
async def cmd_events(
    message: Message, storage: Storage, crypto: Crypto | None, runtime: Runtime
) -> None:
    await _send_events(message, storage, crypto, runtime)


@router.message()
async def fallback(message: Message) -> None:
    await message.answer("Не понял. /help - список команд.")


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
    """Разослать события админам: фото + подпись."""
    if not settings.has_admins:
        return
    bot = Bot(token=settings.bot_token)
    try:
        while True:
            event = await events.get()
            who = event.person_name or "неизвестный"
            conf = f" ({event.confidence:.2f})" if event.confidence else ""
            caption = f"DotEye: {who}{conf}"
            photo = BufferedInputFile(event.jpeg, filename="event.jpg")
            for admin in settings.admin_ids:
                try:
                    await bot.send_photo(admin, photo, caption=caption)
                except Exception as exc:  # noqa: BLE001
                    print(f"[notify] admin {admin}: {exc}")
    finally:
        await bot.session.close()
