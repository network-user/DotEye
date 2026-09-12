"""Тесты пайплайна: детекция -> событие, кулдаун, распознавание."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from doteye.crypto import Crypto, generate_key_b64
from doteye.detector import Detector
from doteye.pipeline import Pipeline
from doteye.recognizer import Recognizer
from doteye.runtime import Runtime
from doteye.storage import Storage
from doteye.config import Settings


class FakeCamera:
    def __init__(self) -> None:
        self.closed = False

    def read(self) -> np.ndarray:
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakeDetector(Detector):
    def __init__(self, boxes: bool = True) -> None:
        self._boxes = boxes

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        return [(0, 0, 10, 10)] if self._boxes else []

    def close(self) -> None:
        pass


class FakeRecognizer(Recognizer):
    def __init__(self, name: str | None, distance: float) -> None:
        self._name = name
        self._distance = distance

    def available(self) -> bool:
        return True

    def embed(self, frame: np.ndarray) -> bytes:
        return b"probe"

    def distance(self, a: bytes, b: bytes) -> float:
        return self._distance


def make_settings(**over: object) -> Settings:
    base = {
        "detect_mode": "presence",
        "min_confidence": 0.5,
        "detection_interval": 1.0,
        "cooldown_seconds": 30.0,
        "face_threshold": 0.4,
        "camera_source": "0",
        "detector_backend": "auto",
        "model_path": "yolov8n.pt",
    }
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


def build(tmp_path: Path, recognizer: Recognizer | None = None, **over: object) -> Pipeline:
    settings = make_settings(**over)
    storage = Storage(tmp_path / "p.db")
    runtime = Runtime(settings, storage)
    crypto = Crypto(generate_key_b64())
    return Pipeline(FakeCamera(), FakeDetector(), recognizer, storage, crypto, runtime)  # type: ignore[arg-type]


def test_event_created(tmp_path: Path) -> None:
    pipe = build(tmp_path)
    event = pipe.step()
    assert event is not None
    assert event.person_name is None
    assert event.jpeg[:2] == b"\xff\xd8"  # JPEG magic


def test_no_event_without_person(tmp_path: Path) -> None:
    pipe = build(tmp_path)
    pipe._detector = FakeDetector(boxes=False)  # type: ignore[assignment]
    assert pipe.step() is None


def test_cooldown_blocks_second_event(tmp_path: Path) -> None:
    pipe = build(tmp_path, cooldown_seconds=100.0)
    assert pipe.step() is not None
    assert pipe.step() is None


def test_cooldown_is_per_person(tmp_path: Path) -> None:
    """Кулдаун "unknown" не должен блокировать другое имя."""
    pipe = build(tmp_path, cooldown_seconds=100.0)
    assert pipe.step() is not None  # unknown
    assert pipe.step() is None       # тот же unknown в кулдауне

    pipe._runtime.detect_mode = "identity"
    pipe._recognizer = FakeRecognizer("alice", 0.1)
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    event = pipe.step()
    assert event is not None
    assert event.person_name == "alice"


def test_identity_recognizes_person(tmp_path: Path) -> None:
    pipe = build(
        tmp_path,
        recognizer=FakeRecognizer("alice", 0.1),
        detect_mode="identity",
        face_threshold=0.4,
    )
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    event = pipe.step()
    assert event is not None
    assert event.person_name == "alice"
    assert event.confidence > 0

    row = pipe._storage.recent_events()[0]
    assert row["person_name"] == "alice"


def test_identity_unknown_when_far(tmp_path: Path) -> None:
    pipe = build(
        tmp_path,
        recognizer=FakeRecognizer("alice", 0.9),
        detect_mode="identity",
        face_threshold=0.4,
    )
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    event = pipe.step()
    assert event is not None
    assert event.person_name is None


def test_model_change_triggers_rebuild(tmp_path: Path, monkeypatch) -> None:
    pipe = build(tmp_path)
    calls: list[tuple] = []

    def fake_build(backend, model_path, *args, **kwargs):
        calls.append((backend, model_path))
        return FakeDetector()

    monkeypatch.setattr("doteye.pipeline.build_detector", fake_build)
    pipe._runtime.model_path = "yolov8s.pt"
    pipe.step()
    assert calls and calls[-1][1] == "yolov8s.pt"

    calls.clear()
    pipe.step()  # модель не менялась - пересборки нет
    assert calls == []


def test_runtime_model_override(tmp_path: Path) -> None:
    pipe = build(tmp_path)
    assert pipe._runtime.model_path == "yolov8n.pt"
    pipe._runtime.model_path = "yolov8m.pt"
    assert pipe._runtime.model_path == "yolov8m.pt"


def test_device_change_triggers_rebuild(tmp_path: Path, monkeypatch) -> None:
    pipe = build(tmp_path)
    calls: list[tuple] = []

    def fake_build(backend, model_path, device, *args, **kwargs):
        calls.append((backend, model_path, device))
        return FakeDetector()

    monkeypatch.setattr("doteye.pipeline.build_detector", fake_build)
    pipe._runtime.device = "cuda"
    pipe.step()
    assert calls and calls[-1][2] == "cuda"

    calls.clear()
    pipe.step()
    assert calls == []
