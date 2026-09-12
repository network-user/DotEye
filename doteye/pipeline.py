"""Пайплайн обработки: камера -> детекция -> (опц. распознавание) -> событие.

Собирает CameraSource + Detector + Recognizer + Storage в цикл.
Компоненты пересобираются на ходу, если из чата поменяли источник камеры
или бэкенд детектора (см. runtime.py). Отвечает за троттлинг
(detection_interval) и кулдаун событий (cooldown_seconds).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from doteye.camera import CameraSource, build_camera
from doteye.crypto import Crypto
from doteye.detector import Detector, build_detector
from doteye.recognizer import Recognizer
from doteye.runtime import Runtime
from doteye.storage import Storage


class DetectionEvent:
    """Результат одного срабатывания.

    jpeg - готовое изображение (JPEG) для отправки в Telegram,
    сохраняется в Storage зашифрованным.
    """

    def __init__(
        self,
        person_name: str | None,
        confidence: float,
        jpeg: bytes,
        detected_at: float,
    ) -> None:
        self.person_name = person_name
        self.confidence = confidence
        self.jpeg = jpeg
        self.detected_at = detected_at


class Pipeline:
    def __init__(
        self,
        camera: CameraSource,
        detector: Detector,
        recognizer: Recognizer | None,
        storage: Storage,
        crypto: Crypto,
        runtime: Runtime,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._recognizer = recognizer
        self._storage = storage
        self._crypto = crypto
        self._runtime = runtime
        self._last_event_ts = 0.0
        # кулдаун отдельно на каждого (имя | "unknown"), а не на всю сцену
        self._last_seen: dict[str, float] = {}
        self._running = False
        # пауза между итерациями фонового цикла (main задаёт из Settings)
        self.poll_interval: float = 0.1
        # ключи компонентов, чтобы понимать, когда нужна пересборка
        self._camera_source = str(runtime.camera_source)
        self._detector_backend = runtime.detector_backend
        self._model_path = runtime.model_path
        self._min_confidence = runtime.min_confidence
        self._device = runtime.device

    def start(self) -> None:
        self._running = True
        print(
            f"[pipeline] started (camera={self._camera_source}, "
            f"detector={self._detector.backend}, model={self._model_path}, "
            f"device={self._device}, mode={self._runtime.detect_mode})"
        )

    def stop(self) -> None:
        self._running = False
        self._camera.close()
        self._detector.close()
        print("[pipeline] stopped")

    def _maybe_rebuild_camera(self) -> None:
        source = str(self._runtime.camera_source)
        if source == self._camera_source:
            return
        print(f"[pipeline] camera -> {source}")
        old = self._camera
        self._camera = build_camera(source)
        self._camera_source = source
        old.close()

    def _maybe_rebuild_detector(self) -> None:
        backend = self._runtime.detector_backend
        model_path = self._runtime.model_path
        min_conf = self._runtime.min_confidence
        device = self._runtime.device
        if (backend == self._detector_backend
                and model_path == self._model_path
                and min_conf == self._min_confidence
                and device == self._device):
            return
        print(
            f"[pipeline] detector -> {backend}, model -> {model_path}, "
            f"device -> {device}"
        )
        old = self._detector
        self._detector = build_detector(
            backend, model_path, device,
            min_conf, self._runtime.remote_processing,
            self._runtime.remote_url, self._runtime.face_model, self._crypto,
        )
        self._detector_backend = backend
        self._model_path = model_path
        self._min_confidence = min_conf
        self._device = device
        old.close()

    def _identify(self, frame: np.ndarray) -> tuple[str | None, float]:
        """Сравнить лицо в кадре с зарегистрированными embeddings."""
        if self._recognizer is None or not self._recognizer.available():
            return None, 0.0
        probe = self._recognizer.embed(frame)
        if probe is None:
            return None, 0.0

        best_name: str | None = None
        best_distance = float("inf")
        threshold = self._runtime.face_threshold
        for row in self._storage.list_people_with_embeddings():
            stored = row["embedding"]
            if not stored:
                continue
            try:
                plain = self._crypto.decrypt(stored)
            except Exception:
                continue
            distance = self._recognizer.distance(probe, plain)
            if distance < best_distance:
                best_distance = distance
                best_name = row["name"]

        if best_name is not None and best_distance <= threshold:
            confidence = max(0.0, 1.0 - best_distance)
            return best_name, confidence
        return None, 0.0

    def step(self) -> DetectionEvent | None:
        """Один прогон. Возвращает событие, если есть человек и пройден кулдаун.

        Кулдаун считается отдельно для каждого человека (имя в identity mode,
        иначе "unknown"), поэтому известный может не спамить, пока другой
        заходит - его событие всё равно пройдёт.
        """
        self._maybe_rebuild_camera()
        self._maybe_rebuild_detector()

        frame = self._camera.read()
        if frame is None:
            return None

        boxes = self._detector.detect(frame)
        if not boxes:
            return None

        person_name: str | None = None
        confidence = 0.0
        if self._runtime.detect_mode == "identity":
            person_name, confidence = self._identify(frame)

        now = time.time()
        key = person_name or "unknown"
        if now - self._last_seen.get(key, 0.0) < self._runtime.cooldown_seconds:
            return None

        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._runtime.jpeg_quality]
        )
        if not ok:
            return None
        jpeg = buf.tobytes()
        encrypted = self._crypto.encrypt(jpeg)

        person_id = None
        if person_name is not None:
            row = self._storage.get_person(person_name)
            person_id = row["id"] if row else None
        self._storage.add_event(person_id, encrypted, confidence)

        self._last_seen[key] = now
        self._last_event_ts = now
        return DetectionEvent(person_name, confidence, jpeg, now)

    @property
    def running(self) -> bool:
        return self._running

    def snapshot(self) -> bytes | None:
        """Текущий кадр камеры как JPEG (для превью в админ-панели).

        Вызывать из потока пайплайна, чтобы не читать камеру параллельно.
        """
        frame = self._camera.read()
        if frame is None:
            return None
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._runtime.jpeg_quality]
        )
        return buf.tobytes() if ok else None
