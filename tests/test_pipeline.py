"""Тесты пайплайна: вход/выход, кулдаун, распознавание, кэш кадра."""

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
        self.reads = 0

    def read(self) -> np.ndarray:
        self.reads += 1
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakeDetector(Detector):
    def __init__(self, boxes: bool = True) -> None:
        self._boxes = boxes

    @property
    def backend(self) -> str:
        return "yolo"

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        return [(0, 0, 10, 10)] if self._boxes else []

    def close(self) -> None:
        pass


class TwoBoxDetector(Detector):
    @property
    def backend(self) -> str:
        return "yolo"

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        return [(0, 0, 10, 10), (20, 0, 30, 10)]

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
        "track_max_misses": 1,
        "notify_exit": True,
        "armed": True,
        "quiet_hours": "",
    }
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


def build(tmp_path: Path, recognizer: Recognizer | None = None, **over: object) -> Pipeline:
    settings = make_settings(**over)
    storage = Storage(tmp_path / "p.db")
    runtime = Runtime(settings, storage)
    crypto = Crypto(generate_key_b64())
    pipe = Pipeline(FakeCamera(), FakeDetector(), recognizer, storage, crypto, runtime)  # type: ignore[arg-type]
    pipe.use_motion_gate = False
    return pipe


def test_event_created(tmp_path: Path) -> None:
    pipe = build(tmp_path)
    events = pipe.step()
    assert len(events) == 1
    event = events[0]
    assert event.person_name is None
    assert event.event_type == "enter"
    assert event.jpeg[:2] == b"\xff\xd8"
    assert event.event_id is not None


def test_no_event_without_person(tmp_path: Path) -> None:
    pipe = build(tmp_path)
    pipe._detector = FakeDetector(boxes=False)  # type: ignore[assignment]
    assert pipe.step() == []


def test_standing_person_does_not_spam(tmp_path: Path) -> None:
    pipe = build(tmp_path, cooldown_seconds=0.0)
    assert pipe.step()
    assert pipe.step() == []
    assert pipe.step() == []


def test_two_boxes_two_enters(tmp_path: Path) -> None:
    pipe = build(tmp_path, cooldown_seconds=0.0)
    pipe._detector = TwoBoxDetector()  # type: ignore[assignment]
    events = pipe.step()
    assert len(events) == 2


def test_identity_recognizes_person(tmp_path: Path) -> None:
    pipe = build(
        tmp_path,
        recognizer=FakeRecognizer("alice", 0.1),
        detect_mode="identity",
        face_threshold=0.4,
    )
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    events = pipe.step()
    assert events
    assert events[0].person_name == "alice"
    assert events[0].confidence > 0

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
    events = pipe.step()
    assert events
    assert events[0].person_name is None


def test_identity_min_distance_across_embeddings(tmp_path: Path) -> None:
    pipe = build(
        tmp_path,
        recognizer=FakeRecognizer("alice", 0.1),
        detect_mode="identity",
        face_threshold=0.4,
    )
    pid = pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"far"))
    pipe._storage.add_embedding(pid, pipe._crypto.encrypt(b"near"))
    events = pipe.step()
    assert events[0].person_name == "alice"


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
    pipe.step()
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


def test_recognizer_rebuilds_on_identity_switch(tmp_path: Path, monkeypatch) -> None:
    built: list[str] = []

    def fake_build(device: str = "cpu"):
        built.append(device)
        return FakeRecognizer("bob", 0.1)

    monkeypatch.setattr("doteye.pipeline.build_recognizer", fake_build)
    pipe = build(tmp_path, recognizer=None, detect_mode="presence")
    pipe._runtime.detect_mode = "identity"
    pipe._storage.upsert_person("bob", pipe._crypto.encrypt(b"ref"))
    events = pipe.step()
    assert built
    assert events
    assert events[0].person_name == "bob"


def test_snapshot_uses_cache_not_camera(tmp_path: Path) -> None:
    pipe = build(tmp_path)
    cam = pipe._cameras[0][1]
    assert pipe.snapshot() is None
    pipe.step()
    reads = cam.reads
    jpeg = pipe.snapshot()
    assert jpeg is not None
    assert cam.reads == reads


def test_disarmed_no_events(tmp_path: Path) -> None:
    pipe = build(tmp_path, armed=False)
    assert pipe.step() == []
    assert pipe._storage.count_events() == 0


def test_quiet_hours_store_without_notify(tmp_path: Path) -> None:
    pipe = build(tmp_path, quiet_hours="00:00-00:00")
    events = pipe.step()
    assert events == []
    assert pipe._storage.count_events() == 1


def test_exit_after_misses(tmp_path: Path) -> None:
    pipe = build(tmp_path, cooldown_seconds=0.0, track_max_misses=1, notify_exit=True)
    assert pipe.step()[0].event_type == "enter"
    pipe._detector = FakeDetector(boxes=False)  # type: ignore[assignment]
    assert pipe.step() == []
    exited = pipe.step()
    assert exited and exited[0].event_type == "exit"


def test_zone_filters_box(tmp_path: Path) -> None:
    pipe = build(tmp_path)
    pipe._runtime.zones_json = (
        '[{"name":"право","x1":0.6,"y1":0.0,"x2":1.0,"y2":1.0}]'
    )
    assert pipe.step() == []


def test_enqueue_drops_oldest() -> None:
    import asyncio

    from doteye.main import _enqueue
    from doteye.pipeline import DetectionEvent

    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    a = DetectionEvent("a", 0, b"x", 1.0)
    b = DetectionEvent("b", 0, b"x", 2.0)
    c = DetectionEvent("c", 0, b"x", 3.0)
    _enqueue(q, a)
    _enqueue(q, b)
    _enqueue(q, c)
    assert q.qsize() == 2
    first = q.get_nowait()
    assert first.person_name == "b"
