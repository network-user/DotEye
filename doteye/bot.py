"""Telegram-бот на aiogram 3.

Команды настройки целиком внутри чата:
  /start      - приветствие
  /status     - текущие настройки
  /mode       - переключить presence | identity (детекция | распознавание лиц)
  /camera     - задать источник (0 = вебка, rtsp/http адрес)
  /detector   - бэкенд детектора: auto | yolo | hog
  /confidence - минимальная уверенность детекции
  /cooldown   - пауза между уведомлениями, сек
  /people     - список известных людей
  /add        - добавить человека (имя), затем прислать фото
  /remove     - удалить человека по имени
  /events     - последние события с кадрами
  /help       - справка

Все настройки хранятся в Storage; пайплайн читает их через runtime.py.
Доступ - только для id из DOTEYE_ADMIN_IDS.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BufferedInputFile, Message, TelegramObject

from doteye.config import Settings
from doteye.crypto import Crypto
from doteye.recognizer import Recognizer
from doteye.runtime import Runtime
from doteye.storage import Storage

HELP = (
    "DotEye - кто зашёл в комнату.\n\n"
    "/status - текущие настройки\n"
    "/mode - presence <-> identity\n"
    "/camera - источник кадров (0, rtsp://, http://)\n"
    "/detector - auto | yolo | yunet | motion\n"
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


class AccessMiddleware(BaseMiddleware):
    """Пускает дальше только админов и прокидывает зависимости в хендлеры."""

    def __init__(self, settings: Settings, storage: Storage, runtime: Runtime,
                 crypto: Crypto | None, recognizer: Recognizer | None) -> None:
        self._settings = settings
        self._storage = storage
        self._runtime = runtime
        self._crypto = crypto
        self._recognizer = recognizer

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
        if isinstance(event, Message) and not _is_admin(event, self._settings):
            await event.answer("Нет доступа.")
            return None
        return await handler(event, data)


router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message, settings: Settings) -> None:
    admin = "админ" if _is_admin(message, settings) else "гость"
    await message.answer(
        f"DotEye на связи ({admin}).\n"
        "Подключи камеру /camera, выбери режим /mode.\n"
        "/help - все команды."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP)


@router.message(Command("status"))
async def cmd_status(message: Message, runtime: Runtime, recognizer: Recognizer | None) -> None:
    face = "on" if recognizer is not None and recognizer.available() else "off/dummy"
    remote = "on" if runtime.remote_processing else "off"
    await message.answer(
        "Настройки:\n"
        f"Режим: {runtime.detect_mode}\n"
        f"Камера: {runtime.camera_source}\n"
        f"Детектор: {runtime.detector_backend} (device={runtime.device})\n"
        f"Мин. уверенность: {runtime.min_confidence}\n"
        f"Кулдаун: {runtime.cooldown_seconds} сек\n"
        f"Порог лица: {runtime.face_threshold}\n"
        f"Распознавание: {face}\n"
        f"Remote: {remote}"
    )


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


@router.message(Command("events"))
async def cmd_events(
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


@router.message()
async def fallback(message: Message) -> None:
    await message.answer("Не понял. /help - список команд.")


async def run_bot(
    settings: Settings,
    storage: Storage,
    runtime: Runtime,
    crypto: Crypto | None,
    recognizer: Recognizer | None,
) -> None:
    if not settings.has_token:
        raise RuntimeError("DOTEYE_BOT_TOKEN не задан")

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher(storage=MemoryStorage())
    di = AccessMiddleware(settings, storage, runtime, crypto, recognizer)
    dp.message.middleware(di)
    dp.include_router(router)

    try:
        await dp.start_polling(bot)
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
