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


class SequenceDetector(Detector):
    """Выдаёт разные наборы боксов на каждом вызове - имитирует сбой трекинга."""

    def __init__(self, steps: list[list[tuple[int, int, int, int]]]) -> None:
        self.steps = steps
        self.calls = 0

    @property
    def backend(self) -> str:
        return "yolo"

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        boxes = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        return boxes

    def close(self) -> None:
        pass


class CountingRecognizer(FakeRecognizer):
    def __init__(self, distance: float) -> None:
        super().__init__(None, distance)
        self.embed_calls = 0

    def embed(self, frame: np.ndarray) -> bytes:
        self.embed_calls += 1
        return b"probe"


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


def test_capture_face_embedding_uses_cached_camera_frame(tmp_path: Path) -> None:
    recognizer = CountingRecognizer(distance=1.0)
    pipe = build(tmp_path, recognizer)
    assert pipe.capture_face_embedding() is None
    pipe.step()
    assert pipe.capture_face_embedding() == b"probe"
    assert recognizer.embed_calls == 1


def test_camera_snapshot_and_info_are_bound_to_source(tmp_path: Path) -> None:
    pipe = build(tmp_path)
    assert pipe.snapshot("0") is None
    pipe.step()
    jpeg = pipe.snapshot("0")
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"
    info = pipe.camera_info("0")
    assert info is not None
    assert info["has_frame"] is True
    assert pipe.camera_info("missing") is None


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


def test_known_person_reentry_is_not_spammed(tmp_path: Path) -> None:
    # Узнанный «alice» уже в кадре (первый бокс). Затем трекер после краткого
    # сбоя «видит» того же человека как новый трек: повторный вход того же
    # имени не должен рассылаться. Один человек, а не поток сообщений.
    pipe = build(
        tmp_path,
        recognizer=FakeRecognizer("alice", 0.1),
        detect_mode="identity",
        face_threshold=0.4,
        cooldown_seconds=0.0,
    )
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    pipe._detector = SequenceDetector([
        [(0, 0, 10, 10)],               # первый вход alice
        [(0, 0, 10,10), (20, 0, 30, 10)],  # второй бокс = «новый» трек того же лица
    ])
    first = pipe.step()
    assert len(first) == 1
    assert first[0].person_name == "alice"

    second = pipe.step()
    assert second == []


def test_known_person_reentry_notifies_when_mute_off(tmp_path: Path) -> None:
    # С выключенным флагом то же повторное появление снова даёт событие.
    pipe = build(
        tmp_path,
        recognizer=FakeRecognizer("alice", 0.1),
        detect_mode="identity",
        face_threshold=0.4,
        cooldown_seconds=0.0,
    )
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    pipe._runtime.mute_known_present = False
    pipe._detector = SequenceDetector([
        [(0, 0, 10, 10)],
        [(20, 0, 30, 10)],
    ])
    pipe.step()
    second = pipe.step()
    assert second


def test_identity_unknown_track_is_rate_limited(tmp_path: Path) -> None:
    recognizer = CountingRecognizer(0.9)
    pipe = build(tmp_path, recognizer=recognizer, detect_mode="identity")
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    pipe.step()
    pipe.step()
    assert recognizer.embed_calls == 1


def test_identity_reference_cache_invalidates_after_embedding_change(tmp_path: Path) -> None:
    recognizer = CountingRecognizer(0.9)
    pipe = build(tmp_path, recognizer=recognizer, detect_mode="identity")
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    calls = 0
    original = pipe._crypto.decrypt

    def count_decrypt(value: bytes) -> bytes:
        nonlocal calls
        calls += 1
        return original(value)

    pipe._crypto.decrypt = count_decrypt  # type: ignore[method-assign]
    pipe.step()
    pipe._storage.upsert_person("bob", pipe._crypto.encrypt(b"ref2"))
    pipe._detector = TwoBoxDetector()  # type: ignore[assignment]
    pipe.step()
    assert calls == 3


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


def test_identity_unknown_raises_voice_alarm(tmp_path: Path) -> None:
    from doteye.audio import DummyPlayer
    from doteye.tts import DummyTTS
    from doteye.voice import build_voice

    pipe = build(
        tmp_path,
        recognizer=FakeRecognizer("alice", 0.9),
        detect_mode="identity",
        face_threshold=0.4,
    )
    tts = DummyTTS()
    voice = build_voice(pipe._runtime, tts, DummyPlayer(), start_worker=False)
    pipe.set_voice(voice)
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    events = pipe.step()
    assert voice.alarming
    assert any(event.event_type == "alarm" for event in events)
    voice.drain()
    assert any("посторонний" in text.casefold() for text in tts.texts)


def test_late_identify_clears_voice_alarm(tmp_path: Path) -> None:
    from doteye.audio import DummyPlayer
    from doteye.tts import DummyTTS
    from doteye.voice import build_voice

    rec = FakeRecognizer("alice", 0.9)
    pipe = build(
        tmp_path,
        recognizer=rec,
        detect_mode="identity",
        face_threshold=0.4,
        cooldown_seconds=0.0,
    )
    voice = build_voice(pipe._runtime, DummyTTS(), DummyPlayer(), start_worker=False)
    pipe.set_voice(voice)
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    pipe.step()
    assert voice.alarming
    rec._distance = 0.1
    events = pipe.step()
    assert voice.alarming is False
    assert any(event.event_type == "alarm_cleared" for event in events)


class MultiCameraPipeline(Pipeline):
    pass


def _multi_build(tmp_path: Path, cameras, recognizer: Recognizer | None = None, **over: object) -> Pipeline:
    settings = make_settings(**over)
    storage = Storage(tmp_path / "multi.db")
    runtime = Runtime(settings, storage)
    crypto = Crypto(generate_key_b64())
    pipe = Pipeline(cameras, FakeDetector(), recognizer, storage, crypto, runtime)  # type: ignore[arg-type]
    pipe.use_motion_gate = False
    return pipe


def test_multi_camera_events_are_tagged_per_camera(tmp_path: Path) -> None:
    cams = [("0", FakeCamera()), ("1", FakeCamera())]
    pipe = _multi_build(tmp_path, cams, cooldown_seconds=0.0, camera_parallel=False)
    events = pipe.step()
    sources = {event.camera_source for event in events}
    assert sources == {"0", "1"}
    assert len(pipe._trackers) == 2


def test_camera_custom_name_used_in_event(tmp_path: Path) -> None:
    from doteye.config import Settings

    settings = Settings(camera_source="0|1", camera_parallel=False)
    storage = Storage(tmp_path / "named.db")
    runtime = Runtime(settings, storage)
    runtime.set_camera_name(0, "Вход")
    crypto = Crypto(generate_key_b64())
    cams = [("0", FakeCamera()), ("1", FakeCamera())]
    pipe = Pipeline(cams, FakeDetector(), None, storage, crypto, runtime)  # type: ignore[arg-type]
    pipe.use_motion_gate = False
    events = pipe.step()
    labels = {event.camera_source for event in events}
    assert "Вход" in labels


def test_camera_override_cooldown_blocks_second_enter(tmp_path: Path) -> None:
    from doteye.config import Settings

    settings = Settings(camera_source="0|1", camera_parallel=False)
    storage = Storage(tmp_path / "ov.db")
    runtime = Runtime(settings, storage)
    runtime.set_camera_override(0, "cooldown_seconds", 999.0)
    crypto = Crypto(generate_key_b64())
    cams = [("0", FakeCamera()), ("1", FakeCamera())]
    pipe = Pipeline(cams, FakeDetector(), None, storage, crypto, runtime)  # type: ignore[arg-type]
    pipe.use_motion_gate = False
    first = pipe.step()
    assert {event.camera_source for event in first} == {"0", "1"}
    second = pipe.step()
    assert second == []


def test_camera_disabled_skips_processing(tmp_path: Path) -> None:
    from doteye.config import Settings

    settings = Settings(camera_source="0|1", camera_parallel=False)
    storage = Storage(tmp_path / "dis.db")
    runtime = Runtime(settings, storage)
    runtime.set_camera_override(0, "enabled", False)
    crypto = Crypto(generate_key_b64())
    cams = [("0", FakeCamera()), ("1", FakeCamera())]
    pipe = Pipeline(cams, FakeDetector(), None, storage, crypto, runtime)  # type: ignore[arg-type]
    pipe.use_motion_gate = False
    events = pipe.step()
    assert {event.camera_source for event in events} == {"1"}


def test_parallel_detection_uses_worker_detectors(tmp_path: Path, monkeypatch) -> None:
    built: list[str] = []

    def fake_build(*args, **kwargs):
        built.append("new")
        return FakeDetector()

    monkeypatch.setattr("doteye.pipeline.build_detector", fake_build)
    cams = [("0", FakeCamera()), ("1", FakeCamera())]
    pipe = _multi_build(tmp_path, cams, cooldown_seconds=0.0, camera_parallel=True)
    assert pipe._parallel_enabled() is True
    pipe.step()
    assert len(pipe._parallel) == 2
    # Второй шаг переиспользует воркеров, не создаёт новые.
    pipe.step()
    assert built.count("new") == 2


def test_parallel_identity_uses_worker_recognizer(tmp_path: Path, monkeypatch) -> None:
    built: list[str] = []

    def fake_build_recognizer(device: str = "cpu"):
        built.append(device)
        return FakeRecognizer("alice", 0.1)

    monkeypatch.setattr("doteye.pipeline.build_recognizer", fake_build_recognizer)
    monkeypatch.setattr("doteye.pipeline.build_detector", lambda *a, **k: FakeDetector())
    cams = [("0", FakeCamera()), ("1", FakeCamera())]
    pipe = _multi_build(
        tmp_path, cams, cooldown_seconds=0.0,
        camera_parallel=True, detect_mode="identity",
    )
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    events = pipe.step()
    assert {event.person_name for event in events} == {"alice"}
    # Каждой камере выдаётся свой recognizer.
    assert set(pipe._worker_recognizers) == {"0", "1"}


def test_parallel_disabled_uses_shared_recognizer(tmp_path: Path) -> None:
    cams = [("0", FakeCamera()), ("1", FakeCamera())]
    pipe = _multi_build(
        tmp_path, cams, cooldown_seconds=0.0,
        camera_parallel=False, detect_mode="identity",
        recognizer=FakeRecognizer("alice", 0.1),
    )
    pipe._storage.upsert_person("alice", pipe._crypto.encrypt(b"ref"))
    pipe.step()
    assert pipe._worker_recognizers == {}
