"""Тесты бота: клавиатуры, статус, callback-хендлеры панели, доступ.

Используются простые стабы Message/CallbackQuery - без Telegram API.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from doteye import bot, models
from doteye.config import Settings
from doteye.crypto import Crypto, generate_key_b64
from doteye.runtime import Runtime
from doteye.audio import DummyPlayer
from doteye.storage import Storage
from doteye.tts import DummyTTS
from doteye.voice import build_voice


class FakeMessage:
    def __init__(self, text: str = "", user_id: int = 1) -> None:
        self.text = text
        self.from_user = SimpleNamespace(id=user_id)
        self.sent: list[tuple] = []
        self.photos: list[tuple] = []
        self.edits: list[tuple] = []

    async def answer(self, text: str, **kwargs) -> None:
        self.sent.append((text, kwargs))

    async def answer_photo(self, photo, **kwargs) -> None:
        self.photos.append((photo, kwargs))

    async def edit_text(self, text: str, **kwargs) -> None:
        self.edits.append((text, kwargs))


class FakeCallback:
    def __init__(self, data: str, user_id: int = 1) -> None:
        self.data = data
        self.from_user = SimpleNamespace(id=user_id)
        self.message = FakeMessage(user_id=user_id)
        self.answers: list[tuple] = []

    async def answer(self, text: str = "", **kwargs) -> None:
        self.answers.append((text, kwargs))


@pytest.fixture()
def env(tmp_path: Path):
    settings = Settings(model_path="yolov8n.pt", device="cpu", detector_backend="auto")
    storage = Storage(tmp_path / "bot.db")
    runtime = Runtime(settings, storage)
    crypto = Crypto(generate_key_b64())
    yield SimpleNamespace(
        settings=settings, storage=storage, runtime=runtime, crypto=crypto
    )
    storage.close()


def test_panel_keyboard_has_device_button(env) -> None:
    kb = bot._panel_keyboard(env.runtime)
    labels = [b.text for row in kb.inline_keyboard for b in row]
    assert any("Охрана" in label for label in labels)
    assert any("Режим" in label for label in labels)
    assert any("События" in label for label in labels)
    assert any("Люди" in label for label in labels)
    assert any("Превью" in label for label in labels)
    assert any("Камера" in label for label in labels)
    assert any("Здоровье" in label for label in labels)
    assert any("Тонкая настройка" in label for label in labels)


def test_tune_keyboard_contains_advanced_controls(env) -> None:
    kb = bot._tune_keyboard(env.runtime)
    labels = [b.text for row in kb.inline_keyboard for b in row]
    assert any("Устройство" in label for label in labels)
    assert any("Детектор" in label for label in labels)
    assert any("Модель" in label for label in labels)
    assert any("Порог" in label for label in labels)
    assert any("Кулдаун" in label for label in labels)
    assert any("Интервал" in label for label in labels)
    assert any("Remote" in label for label in labels)


def test_camera_keyboard_lists_sources_and_hides_credentials(env) -> None:
    env.runtime.camera_source = "0|1"
    kb = bot._camera_keyboard(env.runtime)
    labels = [b.text for row in kb.inline_keyboard for b in row]
    assert any("Камера 1" in label for label in labels)
    assert any("Камера 2" in label for label in labels)
    assert "user:password" not in bot._camera_source_text("rtsp://user:password@cam.local/live")


def test_camera_keyboard_uses_custom_names(env) -> None:
    env.runtime.camera_source = "0|1"
    env.runtime.set_camera_name(0, "Вход")
    kb = bot._camera_keyboard(env.runtime)
    labels = [b.text for row in kb.inline_keyboard for b in row]
    assert any("Вход" in label for label in labels)


@pytest.mark.asyncio
async def test_cb_camera_toggle_and_reset(env) -> None:
    env.runtime.camera_source = "0|1"
    cq = FakeCallback("camera:tog:0:notify_enter")
    await bot.cb_camera_toggle(cq, env.runtime)
    assert env.runtime.camera_notify_enter(0) is False
    cq = FakeCallback("camera:reset:0")
    await bot.cb_camera_reset(cq, env.runtime)
    assert env.runtime.camera_override(0) == {}


@pytest.mark.asyncio
async def test_cb_camera_mode_toggles(env) -> None:
    env.runtime.camera_source = "0"
    cq = FakeCallback("camera:mode:0")
    await bot.cb_camera_mode(cq, env.runtime)
    assert env.runtime.camera_detect_mode(0) == "identity"
    await bot.cb_camera_mode(cq, env.runtime)
    assert env.runtime.camera_detect_mode(0) == "presence"


def test_camera_conf_text_shows_settings(env) -> None:
    env.runtime.camera_source = "0"
    rhs = bot._camera_conf_keyboard(0, env.runtime)
    labels = [b.text for row in rhs.inline_keyboard for b in row]
    assert any("Обработка" in label for label in labels)
    assert any("Вход" in label for label in labels)
    assert any("Режим" in label for label in labels)


@pytest.mark.asyncio
async def test_camera_rename_flow(env) -> None:
    env.runtime.camera_source = "0"
    from aiogram.fsm.context import FSMContext
    from aiogram.fsm.storage.memory import MemoryStorage

    state = FSMContext(storage=MemoryStorage(), key=bot.CameraForm.name)
    cq = FakeCallback("camera:rename:0")
    await bot.cb_camera_rename(cq, state, env.runtime)
    await bot.proc_camera_name(FakeMessage("Крыльцо"), state, env.runtime)
    assert env.runtime.camera_label(0) == "Крыльцо"


@pytest.mark.asyncio
async def test_camera_view_and_test_snapshot(env) -> None:
    class Pipeline:
        def camera_info(self, source: str):
            assert source == "0"
            return {"healthy": True, "has_frame": True, "reconnects": 0, "last_error": None}

        def snapshot(self, source: str):
            assert source == "0"
            return b"jpeg"

    pipeline = Pipeline()
    cq = FakeCallback("camera:view:0")
    await bot.cb_camera_view(cq, env.runtime, pipeline)  # type: ignore[arg-type]
    assert "Камера 1" in cq.message.edits[-1][0]

    cq = FakeCallback("camera:test:0")
    await bot.cb_camera_test(cq, env.runtime, pipeline)  # type: ignore[arg-type]
    assert cq.message.photos


def test_model_keyboard_marks_current(env) -> None:
    kb = bot._model_keyboard(env.runtime)
    labels = [b.text for row in kb.inline_keyboard for b in row]
    current = [label for label in labels if "[текущая]" in label]
    assert len(current) == 1
    assert len(labels) == len(models.YOLO_MODELS) + 1  # + Назад


def test_status_text_contains_key_fields(env) -> None:
    text = bot._status_text(env.runtime, None, None)
    assert "presence" in text
    assert "YOLOv8n" in text
    assert "cpu" in text
    assert "Охрана" in text
    assert "Голос" in text


@pytest.mark.asyncio
async def test_cb_mode_toggles(env) -> None:
    recognizer = type("Recognizer", (), {"available": lambda self: True})()
    cq = FakeCallback("panel:mode")
    await bot.cb_mode(cq, env.runtime, recognizer)
    assert env.runtime.detect_mode == "identity"
    await bot.cb_mode(cq, env.runtime, recognizer)
    assert env.runtime.detect_mode == "presence"


@pytest.mark.asyncio
async def test_cb_detector_cycles(env) -> None:
    cq = FakeCallback("panel:detector")
    await bot.cb_detector(cq, env.runtime)
    assert env.runtime.detector_backend == "yolo"
    await bot.cb_detector(cq, env.runtime)
    assert env.runtime.detector_backend == "yunet"


@pytest.mark.asyncio
async def test_cb_device_cycles(env) -> None:
    cq = FakeCallback("panel:device")
    await bot.cb_device(cq, env.runtime)
    assert env.runtime.device == "cuda"
    await bot.cb_device(cq, env.runtime)
    assert env.runtime.device == "mps"
    await bot.cb_device(cq, env.runtime)
    assert env.runtime.device == "cpu"


@pytest.mark.asyncio
async def test_cb_model_set_switches_backend_and_saves(env) -> None:
    cq = FakeCallback("model:set:yolov8s.pt")
    await bot.cb_model_set(cq, env.runtime, None)
    assert env.runtime.model_path == "yolov8s.pt"
    assert env.runtime.detector_backend == "yolo"
    assert cq.message.edits


@pytest.mark.asyncio
async def test_cb_model_set_unknown(env) -> None:
    cq = FakeCallback("model:set:nope.pt")
    await bot.cb_model_set(cq, env.runtime, None)
    assert env.runtime.model_path == "yolov8n.pt"
    assert any("Неизвестная" in a[0] for a in cq.answers)


@pytest.mark.asyncio
async def test_cb_snapshot_without_pipeline(env) -> None:
    cq = FakeCallback("panel:snapshot")
    await bot.cb_snapshot(cq, None)
    assert any("выключен" in a[0] for a in cq.answers)


@pytest.mark.asyncio
async def test_cb_people_empty_and_filled(env) -> None:
    cq = FakeCallback("panel:people")
    await bot.cb_people(cq, env.storage)
    assert any("пуст" in text for text, _ in cq.message.edits)

    env.storage.upsert_person("alice", b"emb")
    await bot.cb_people(cq, env.storage)
    kwargs = cq.message.edits[-1][1]
    labels = [b.text for row in kwargs["reply_markup"].inline_keyboard for b in row]
    assert any("alice" in label for label in labels)


@pytest.mark.asyncio
async def test_cb_arm_toggles(env) -> None:
    cq = FakeCallback("panel:arm")
    assert env.runtime.armed is True
    await bot.cb_arm(cq, env.runtime)
    assert env.runtime.armed is False


@pytest.mark.asyncio
async def test_assign_event_helper(env) -> None:
    pid = env.storage.upsert_person("alice", None)
    eid = env.storage.add_event(None, env.crypto.encrypt(b"not-a-jpeg"), 0.1)
    text = bot._assign_event(eid, pid, env.storage, env.crypto, None)
    assert "alice" in text
    row = env.storage.get_event(eid)
    assert row["person_id"] == pid


def test_is_admin_rules() -> None:
    open_settings = Settings(admin_ids=[], allow_open_access=False)
    assert bot._is_admin(FakeMessage(user_id=99), open_settings) is False
    open_dev = Settings(environment="development", admin_ids=[], allow_open_access=True)
    assert bot._is_admin(FakeMessage(user_id=99), open_dev) is True

    closed = Settings(admin_ids=[1, 2])
    assert bot._is_admin(FakeMessage(user_id=1), closed) is True
    assert bot._is_admin(FakeMessage(user_id=3), closed) is False
    assert bot._is_admin_user(None, closed) is False
    assert bot._is_admin_user(2, closed) is True


def test_queue_notifications_persists_one_delivery_per_admin(env) -> None:
    event_id = env.storage.add_event(None, b"encrypted", 0.6)
    event = SimpleNamespace(
        event_id=event_id,
        event_type="enter",
        person_name=None,
        confidence=0.6,
        jpeg=b"preview",
        detected_at=123.0,
        caption="DotEye: unknown",
    )
    settings = Settings(admin_ids=[11, 22])
    bot.queue_notifications(settings, env.storage, env.crypto, event)
    bot.queue_notifications(settings, env.storage, env.crypto, event)

    rows = env.storage.claim_due_notifications(limit=10)
    assert [int(row["admin_id"]) for row in rows] == [11, 22]
    assert all(row["jpeg"] != b"preview" for row in rows)
    assert all(env.crypto.decrypt(bytes(row["jpeg"])) == b"preview" for row in rows)
    assert all(json.loads(str(row["payload"]))["unknown_enter"] is True for row in rows)
    assert all(json.loads(str(row["payload"]))["show_dismiss"] is True for row in rows)


@pytest.mark.asyncio
async def test_middleware_blocks_stranger(env) -> None:
    from aiogram.types import Message, User

    closed = Settings(admin_ids=[1])
    middleware = bot.AccessMiddleware(closed, env.storage, env.runtime, env.crypto, None, None)
    called = False

    async def handler(event, data):
        nonlocal called
        called = True

    sent: list = []

    class AnswerableMessage(Message):
        async def answer(self, text, **kwargs):  # type: ignore[override]
            sent.append(text)

    message = AnswerableMessage.model_construct(
        message_id=1,
        date=0,
        chat=SimpleNamespace(id=999),
        from_user=User.model_construct(id=999, is_bot=False, first_name="x"),
        text="hi",
    )
    await middleware(handler, message, {})
    assert called is False
    assert sent


@pytest.mark.asyncio
async def test_middleware_injects_and_passes(env) -> None:
    middleware = bot.AccessMiddleware(
        env.settings, env.storage, env.runtime, env.crypto, None, None
    )
    seen: dict = {}

    async def handler(event, data):
        seen.update(data)

    await middleware(handler, FakeMessage(user_id=1), {})
    assert seen["runtime"] is env.runtime
    assert seen["storage"] is env.storage
    assert seen["crypto"] is env.crypto
    assert seen["voice"] is None


@pytest.mark.asyncio
async def test_cb_voice_toggle_and_panel(env) -> None:
    cq = FakeCallback("voice:open")
    await bot.cb_voice_open(cq, env.runtime, None)
    assert cq.message.edits
    assert "Голос и тревога" in cq.message.edits[-1][0]

    cq = FakeCallback("voice:tog:voice_welcome")
    assert env.runtime.voice_welcome is True
    await bot.cb_voice_toggle(cq, env.runtime, None)
    assert env.runtime.voice_welcome is False


@pytest.mark.asyncio
async def test_voice_say_and_alarm(env) -> None:
    tts = DummyTTS()
    voice = build_voice(env.runtime, tts, DummyPlayer(), start_worker=False)
    assert voice.announce("Отойди!")
    voice.drain()
    assert tts.texts[-1] == "Отойди!"

    cq = FakeCallback("voice:trigger")
    await bot.cb_voice_trigger(cq, env.runtime, voice)
    assert voice.alarming
    cq = FakeCallback("voice:dismiss")
    await bot.cb_voice_dismiss(cq, env.runtime, voice)
    assert voice.alarming is False
