"""Точка входа DotEye.

Собирает компоненты (камера, детектор, распознавание, хранилище) и запускает
Telegram-бота и пайплайн обработки в одном процессе. aiogram работает в
event loop, а OpenCV/YOLO - CPU-bound, поэтому шаг пайплайна выполняется
в отдельном потоке через asyncio.to_thread.

События пайплайна кладутся в asyncio.Queue, откуда их разбирает
send_notifications() и рассылает админам фото + подпись.

Запуск:
    python -m doteye.main
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

from doteye.bot import notify_startup, run_bot, send_notifications
from doteye.camera import build_cameras
from doteye.config import get_settings
from doteye.crypto import Crypto
from doteye.detector import build_detector
from doteye.pipeline import DetectionEvent, Pipeline
from doteye.recognizer import build_recognizer
from doteye.runtime import Runtime
from doteye.storage import Storage


def _enqueue(events: asyncio.Queue[DetectionEvent], event: DetectionEvent) -> None:
    """Положить событие; при переполнении вытеснить самое старое."""
    try:
        events.put_nowait(event)
        return
    except asyncio.QueueFull:
        pass
    try:
        events.get_nowait()
    except asyncio.QueueEmpty:
        pass
    try:
        events.put_nowait(event)
    except asyncio.QueueFull:
        print("[queue] drop")


async def _pipeline_loop(
    pipeline: Pipeline, events: asyncio.Queue[DetectionEvent]
) -> None:
    """Фоновый цикл: гоняет step() и складывает события в очередь."""
    while True:
        try:
            batch = await asyncio.to_thread(pipeline.step)
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] ошибка шага: {exc}")
            await asyncio.sleep(1.0)
            continue
        for event in batch:
            who = event.person_name or event.event_type
            print(f"[event] {event.event_type} {who} ({event.confidence:.2f})")
            _enqueue(events, event)
        await asyncio.sleep(pipeline.poll_interval)


async def main() -> None:
    settings = get_settings()
    if not settings.has_token:
        print("[config] DOTEYE_BOT_TOKEN не задан. Заполни .env (см. .env.example).")
        return

    storage = Storage(settings.db_path)
    runtime = Runtime(settings, storage)

    try:
        crypto = Crypto(settings.crypto_key_env)
    except ValueError as exc:
        print(f"[config] {exc}. Сгенерируй ключ: python -m doteye.crypto")
        print("[config] Пайплайн и кадры выключены до заполнения ключа.")
        crypto = None

    recognizer = build_recognizer(runtime.device)

    events: asyncio.Queue[DetectionEvent] = asyncio.Queue(maxsize=32)

    pipeline = None
    if crypto is not None:
        cameras = build_cameras(runtime.camera_source)
        detector = build_detector(
            runtime.detector_backend, runtime.model_path, runtime.device,
            runtime.min_confidence, runtime.remote_processing, runtime.remote_url,
            settings.face_model, crypto,
            imgsz=runtime.imgsz,
            remote_fallback=runtime.remote_fallback,
            remote_insecure=runtime.remote_insecure,
        )
        pipeline = Pipeline(cameras, detector, recognizer, storage, crypto, runtime)
        pipeline.start()

    tasks = [
        asyncio.create_task(
            run_bot(settings, storage, runtime, crypto, recognizer, pipeline)
        )
    ]
    if settings.has_admins:
        tasks.append(asyncio.create_task(send_notifications(settings, events)))
    if pipeline is not None:
        tasks.append(asyncio.create_task(_pipeline_loop(pipeline, events)))

    await notify_startup(settings, runtime, recognizer, pipeline)

    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if pipeline is not None:
            pipeline.stop()
        storage.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[main] остановлено")
