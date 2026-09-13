"""Пайплайн: камера -> детекция -> трек вход/выход -> (опц. лицо) -> событие.

Компоненты пересобираются на ходу. Событие только когда человек появился
или исчез, а не на каждый кадр с боксом. Кулдаун гасит дребезг повторного
входа. Кадр для превью берётся из кэша, без второго read() камеры.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from doteye.annotate import annotate, crop_box, encode_jpeg, privacy_frame
from doteye.camera import CameraSource, build_cameras, parse_sources
from doteye.crypto import Crypto
from doteye.detector import Detector, MotionGate, build_detector
from doteye.recognizer import Recognizer, build_recognizer
from doteye.runtime import Runtime
from doteye.storage import Storage
from doteye.tracker import Box, IoUTracker, Track
from doteye.voice import VoiceEngine, VoiceScene, VoiceTrack
from doteye.zones import filter_boxes, parse_zones


class DetectionEvent:
    """Одно срабатывание: вход, выход или служебный алерт."""

    def __init__(
        self,
        person_name: str | None,
        confidence: float,
        jpeg: bytes | None,
        detected_at: float,
        event_id: int | None = None,
        event_type: str = "enter",
        camera_source: str = "",
        zone: str | None = None,
        caption: str | None = None,
    ) -> None:
        self.person_name = person_name
        self.confidence = confidence
        self.jpeg = jpeg
        self.detected_at = detected_at
        self.event_id = event_id
        self.event_type = event_type
        self.camera_source = camera_source
        self.zone = zone
        self.caption = caption


class Pipeline:
    def __init__(
        self,
        camera: CameraSource | list[tuple[str, CameraSource]],
        detector: Detector,
        recognizer: Recognizer | None,
        storage: Storage,
        crypto: Crypto,
        runtime: Runtime,
    ) -> None:
        if isinstance(camera, list):
            self._cameras: list[tuple[str, CameraSource]] = camera
        else:
            label = parse_sources(runtime.camera_source)[0]
            self._cameras = [(label, camera)]
        self._detector = detector
        self._recognizer = recognizer
        self._storage = storage
        self._crypto = crypto
        self._runtime = runtime
        self._running = False
        self._camera_source = str(runtime.camera_source)
        self._detector_backend = runtime.detector_backend
        self._model_path = runtime.model_path
        self._min_confidence = runtime.min_confidence
        self._device = runtime.device
        self._imgsz = runtime.imgsz
        self._nms_iou = runtime.nms_iou
        self._person_min_area = runtime.person_min_area
        self._remote = runtime.remote_processing
        self._remote_url = runtime.remote_url
        self._mode_seen = runtime.detect_mode
        self._recognizer_device = runtime.device
        self._recognizer_tried = recognizer is not None
        self._trackers: dict[str, IoUTracker] = {}
        self._gates: dict[str, MotionGate] = {}
        self._last_seen: dict[str, float] = {}
        self._parallel: dict[str, Detector] = {}
        self._camera_cfgs: list[tuple[str, float, bool, bool, bool, str]] = []
        self._lock = threading.Lock()
        self._preview_frame: np.ndarray | None = None
        self._preview_items: list[tuple[Box, str, tuple[int, int, int]]] = []
        self._camera_previews: dict[str, tuple[np.ndarray, list[tuple[Box, str, tuple[int, int, int]]]]] = {}
        self._steps = 0
        self._remote_down = False
        self._poll_override: float | None = None
        self._identity_cache_revision = -1
        self._identity_references: list[tuple[str, bytes]] = []
        self._identity_attempts: dict[tuple[str, int], tuple[float, int]] = {}
        # Имена узнанных людей, присутствующих в кадре по итогу прошлого шага.
        # Пока человек физически не ушёл, повторный enter того же имени (после
        # краткого сбоя трекинга или на другой камере) не рассылается заново.
        self._present_known: set[str] = set()
        self._identity_lock = threading.Lock()
        # Первый шаг сразу применяет retention к архиву после перезапуска.
        self._last_prune_at = 0.0
        self._voice: VoiceEngine | None = None
        self.use_motion_gate = True
        self._refresh_camera_cfgs()

    def _refresh_camera_cfgs(self) -> None:
        """Пересобрать снимок per-camera настроек перед шагом."""
        cfgs: list[tuple[str, float, bool, bool, bool, str]] = []
        now = time.time()
        for index, (source, _camera) in enumerate(self._cameras):
            cfgs.append((
                source,
                self._runtime.camera_cooldown(index),
                self._runtime.camera_quiet(index),
                self._runtime.camera_notify_enter(index),
                self._runtime.camera_notify_exit(index),
                self._runtime.camera_detect_mode(index),
            ))
        self._camera_cfgs = cfgs

    def _camera_cfg(self, source: str) -> tuple[str, float, bool, bool, bool, str]:
        for cfg in self._camera_cfgs:
            if cfg[0] == source:
                return cfg
        return (source, self._runtime.cooldown_seconds, self._runtime.is_quiet(),
                self._runtime.should_notify(), self._runtime.notify_exit,
                self._runtime.detect_mode)


    def set_voice(self, voice: VoiceEngine | None) -> None:
        self._voice = voice

    @property
    def voice(self) -> VoiceEngine | None:
        return self._voice

    def _voice_alarming(self) -> bool:
        return self._voice is not None and self._voice.alarming

    def start(self) -> None:
        self._running = True
        cams = ",".join(name for name, _ in self._cameras)
        print(
            f"[pipeline] started (camera={cams}, "
            f"detector={self._detector.backend}, model={self._model_path}, "
            f"device={self._device}, mode={self._runtime.detect_mode})"
        )

    def stop(self) -> None:
        self._running = False
        for _, cam in self._cameras:
            cam.close()
        self._close_parallel()
        self._detector.close()
        print("[pipeline] stopped")

    def _close_parallel(self) -> None:
        for det in self._parallel.values():
            try:
                det.close()
            except Exception:
                pass
        self._parallel.clear()

    def _parallel_enabled(self) -> bool:
        return bool(self._runtime.camera_parallel) and len(self._cameras) > 1

    def _worker_detector(self, source: str) -> Detector:
        """Свой экземпляр детектора на камеру для параллельного инференса."""
        det = self._parallel.get(source)
        if det is not None:
            return det
        det = build_detector(
            self._detector_backend, self._model_path, self._device,
            self._min_confidence, self._remote, self._remote_url,
            self._runtime.face_model, self._crypto,
            imgsz=self._imgsz,
            remote_fallback=self._runtime.remote_fallback,
            remote_insecure=self._runtime.remote_insecure,
            nms_iou=self._nms_iou,
            person_min_area=self._person_min_area,
        )
        self._parallel[source] = det
        return det

    @property
    def poll_interval(self) -> float:
        if self._poll_override is not None:
            return self._poll_override
        return max(0.05, self._runtime.detection_interval)

    @poll_interval.setter
    def poll_interval(self, value: float) -> None:
        self._poll_override = float(value)

    def _maybe_rebuild_camera(self) -> None:
        source = str(self._runtime.camera_source)
        if source == self._camera_source:
            return
        print(f"[pipeline] camera -> {source}")
        old = self._cameras
        self._cameras = build_cameras(source)
        with self._lock:
            self._camera_previews.clear()
        self._camera_source = source
        self._trackers.clear()
        self._gates.clear()
        self._last_seen.clear()
        self._close_parallel()
        for _, cam in old:
            cam.close()
        self._refresh_camera_cfgs()

    def _maybe_rebuild_detector(self) -> None:
        backend = self._runtime.detector_backend
        model_path = self._runtime.model_path
        min_conf = self._runtime.min_confidence
        device = self._runtime.device
        imgsz = self._runtime.imgsz
        nms_iou = self._runtime.nms_iou
        person_min_area = self._runtime.person_min_area
        remote = self._runtime.remote_processing
        remote_url = self._runtime.remote_url
        if (
            backend == self._detector_backend
            and model_path == self._model_path
            and min_conf == self._min_confidence
            and device == self._device
            and imgsz == self._imgsz
            and nms_iou == self._nms_iou
            and person_min_area == self._person_min_area
            and remote == self._remote
            and remote_url == self._remote_url
        ):
            return
        print(
            f"[pipeline] detector -> {backend}, model -> {model_path}, "
            f"device -> {device}, remote -> {remote}"
        )
        old = self._detector
        self._detector = build_detector(
            backend, model_path, device, min_conf, remote, remote_url,
            self._runtime.face_model, self._crypto,
            imgsz=imgsz,
            remote_fallback=self._runtime.remote_fallback,
            remote_insecure=self._runtime.remote_insecure,
            nms_iou=nms_iou,
            person_min_area=person_min_area,
        )
        self._detector_backend = backend
        self._model_path = model_path
        self._min_confidence = min_conf
        self._device = device
        self._imgsz = imgsz
        self._nms_iou = nms_iou
        self._person_min_area = person_min_area
        self._remote = remote
        self._remote_url = remote_url
        old.close()
        self._close_parallel()

    def _maybe_rebuild_recognizer(self) -> None:
        mode = self._runtime.detect_mode
        device = self._runtime.device
        if mode != "identity":
            self._mode_seen = mode
            return
        missing = self._recognizer is None or not self._recognizer.available()
        switched = self._mode_seen != "identity" and any(
            mode == "identity" for *_rest, mode in self._camera_cfgs
        )
        device_changed = device != self._recognizer_device
        if missing and (switched or not self._recognizer_tried):
            self._recognizer = build_recognizer(device)
            self._recognizer_tried = True
            self._recognizer_device = device
        elif device_changed and not missing:
            rebuilt = build_recognizer(device)
            if rebuilt.available():
                self._recognizer = rebuilt
            self._recognizer_device = device
        self._mode_seen = mode

    def _identify_crop(self, frame: np.ndarray, box: Box) -> tuple[str | None, float]:
        if self._recognizer is None or not self._recognizer.available():
            return None, 0.0
        crop = crop_box(frame, box)
        if crop is None:
            return None, 0.0

        with self._identity_lock:
            probe = self._recognizer.embed(crop)
            if probe is None:
                return None, 0.0

            revision = self._storage.people_revision
            if revision != self._identity_cache_revision:
                references: list[tuple[str, bytes]] = []
                for row in self._storage.list_people_with_embeddings():
                    for stored in row["embeddings"]:
                        if not stored:
                            continue
                        try:
                            references.append((str(row["name"]), self._crypto.decrypt(stored)))
                        except Exception:
                            # A corrupt legacy row must not break recognition.
                            continue
                self._identity_references = references
                self._identity_cache_revision = revision

            best_name: str | None = None
            best_distance = float("inf")
            threshold = self._runtime.face_threshold
            for name, plain in self._identity_references:
                distance = self._recognizer.distance(probe, plain)
                if distance < best_distance:
                    best_distance = distance
                    best_name = name

        if best_name is not None and best_distance <= threshold:
            return best_name, max(0.0, 1.0 - best_distance)
        return None, 0.0

    def _should_identify(self, source: str, track: Track, now: float) -> bool:
        """Не повторять дорогой face embedding на каждом кадре одного трека."""
        key = (source, track.id)
        with self._identity_lock:
            revision = self._storage.people_revision
            previous = self._identity_attempts.get(key)
            if previous is not None and previous[1] == revision and now - previous[0] < 2.0:
                return False
            self._identity_attempts[key] = (now, revision)
        return True

    def _tracker(self, source: str) -> IoUTracker:
        tr = self._trackers.get(source)
        if tr is None:
            tr = IoUTracker(max_misses=self._runtime.track_max_misses)
            self._trackers[source] = tr
        else:
            tr.max_misses = self._runtime.track_max_misses
        return tr

    def _gate(self, source: str) -> MotionGate:
        gate = self._gates.get(source)
        if gate is None:
            gate = MotionGate()
            self._gates[source] = gate
        return gate

    def _should_store(self) -> bool:
        return self._runtime.armed

    def _cooldown_ok(self, source: str, key: str, now: float) -> bool:
        cooldown = self._camera_cfg(source)[1]
        return now - self._last_seen.get(key, 0.0) >= cooldown

    def _track_key(self, source: str, track: Track, event_type: str) -> str:
        if track.person_name:
            who = track.person_name
        else:
            # Координаты меняются от шага к шагу и превращали одного гостя в
            # десятки «новых» неизвестных. Для нераспознанного человека
            # источник - граница инцидента; трекер хранит его внутри кадра,
            # а кулдаун защищает от нового сообщения после краткого сбоя.
            who = "unknown"
        return f"{source}:{who}:{event_type}"

    def _emit(
        self,
        frame: np.ndarray,
        track: Track,
        event_type: str,
        source: str,
        now: float,
    ) -> DetectionEvent | None:
        conf = f" ({track.confidence:.2f})" if track.confidence else ""
        zone = f", зона {track.zone}" if track.zone else ""
        label = self._runtime.camera_label(
            next((i for i, (s, _c) in enumerate(self._cameras) if s == source), -1), source,
        )
        color = (40, 200, 80) if track.person_name else (40, 40, 220)
        if event_type == "exit":
            color = (160, 160, 160)
        vis = privacy_frame(
            frame, [track.box], self._runtime.privacy_mode, self._runtime.privacy_blocks,
        ) if self._runtime.privacy_mode != "off" else annotate(
            frame, [(track.box, f"{track.person_name or '?'}{conf}".strip(),
                     (40, 200, 80) if track.person_name else (40, 40, 220))],
        )
        jpeg = encode_jpeg(vis, self._runtime.jpeg_quality)
        original = encode_jpeg(frame, self._runtime.jpeg_quality)
        if jpeg is None or original is None:
            return None
        boxes_json = json.dumps([{
            "x1": track.box[0], "y1": track.box[1],
            "x2": track.box[2], "y2": track.box[3],
            "name": track.person_name, "confidence": track.confidence,
        }])
        person_id = None
        if track.person_name is not None:
            row = self._storage.get_person(track.person_name)
            person_id = int(row["id"]) if row else None
        encrypted = self._crypto.encrypt(original)
        event_id = self._storage.add_event(
            person_id, encrypted, track.confidence,
            event_type=event_type, boxes=boxes_json,
            camera_source=source, zone=track.zone,
        )
        if event_type == "enter" and track.person_name:
            caption = f"DotEye: Обнаружен: {track.person_name}{conf}"
        elif event_type == "enter":
            caption = "DotEye: Обнаружен незнакомый человек"
        elif track.person_name:
            caption = f"DotEye: Вышел: {track.person_name}{conf}"
        else:
            caption = "DotEye: Незнакомый человек вышел"
        caption += f"\nкамера {label}{zone}"
        return DetectionEvent(
            track.person_name, track.confidence, jpeg, now,
            event_id=event_id, event_type=event_type,
            camera_source=label, zone=track.zone, caption=caption,
        )

    def _detector_error(self) -> tuple[str | None, bool]:
        """Первый last_error среди основного детектора и параллельных."""
        detectors = [self._detector, *self._parallel.values()]
        using = bool(getattr(self._detector, "using_fallback", False))
        for det in detectors:
            err = getattr(det, "last_error", None)
            if err:
                using = using or bool(getattr(det, "using_fallback", False))
                return err, using
            using = using or bool(getattr(det, "using_fallback", False))
        return None, using

    def _collect_alerts(self) -> list[DetectionEvent]:
        err, using = self._detector_error()
        out: list[DetectionEvent] = []
        now = time.time()
        if err:
            if not self._remote_down:
                self._remote_down = True
                extra = " Локальный fallback." if using else ""
                out.append(DetectionEvent(
                    None, 0.0, None, now, event_type="alert",
                    caption=f"DotEye: remote недоступен ({err}).{extra}",
                ))
        elif self._remote_down:
            self._remote_down = False
            out.append(DetectionEvent(
                None, 0.0, None, now, event_type="alert",
                caption="DotEye: remote снова доступен.",
            ))
        return out

    def step(self) -> list[DetectionEvent]:
        """Один прогон по всем камерам. События входа/выхода."""
        self._maybe_rebuild_camera()
        self._maybe_rebuild_detector()
        self._maybe_rebuild_recognizer()
        self._refresh_camera_cfgs()

        events: list[DetectionEvent] = []
        events.extend(self._collect_alerts())
        now = time.time()
        zones = parse_zones(self._runtime.zones_json)
        entered_v: list[VoiceTrack] = []
        active_v: list[VoiceTrack] = []
        exited_v: list[VoiceTrack] = []
        newly_v: list[VoiceTrack] = []

        frames: list[tuple[str, np.ndarray, str]] = []
        for index, (source, camera) in enumerate(self._cameras):
            if not self._runtime.camera_enabled(index):
                continue
            frame = camera.read()
            if frame is not None:
                frames.append((source, frame, self._camera_cfg(source)[5]))

        boxes_by_source = self._detect_frames(frames, zones)

        for source, frame, identity in frames:
            cfg = self._camera_cfg(source)
            _, _cooldown, quiet, notify_enter, notify_exit, _mode = cfg
            tracker = self._tracker(source)
            tagged = boxes_by_source.get(source)
            if tagged is None:
                tagged = filter_boxes(self._detector.detect(frame), zones, frame.shape)
            boxes = [box for box, _ in tagged]
            zone_by_box = {box: zone for box, zone in tagged}

            update = tracker.update(boxes)
            preview_items: list[tuple[Box, str, tuple[int, int, int]]] = []

            for track in update.entered:
                track.zone = zone_by_box.get(track.box)
                if identity == "identity" and (self._should_identify(source, track, now) or self._voice_alarming()):
                    name, conf = self._identify_crop(frame, track.box)
                    track.person_name = name
                    track.confidence = conf
                    track.identified = name is not None
                if (
                    track.person_name
                    and self._runtime.mute_known_present
                    and track.person_name in self._present_known
                ):
                    # Этот человек уже был в кадре на прошлом шаге: трек просто
                    # пересоздался после краткого сбоя. Не считать входом, не
                    # рассылать уведомлений и не повторять приветствие - иначе
                    # один человек генерирует поток сообщений, пока стоит в кадре.
                    preview_items.append((
                        track.box,
                        f"{track.person_name} {track.confidence:.2f}".strip(),
                        (40, 200, 80),
                    ))
                    continue
                entered_v.append(_voice_track(source, track))
                key = self._track_key(source, track, "enter")
                preview_items.append((
                    track.box,
                    f"{track.person_name or '?'} {track.confidence:.2f}".strip(),
                    (40, 200, 80) if track.person_name else (40, 40, 220),
                ))
                if not self._should_store():
                    continue
                if not self._cooldown_ok(source, key, now):
                    continue
                self._last_seen[key] = now
                event = self._emit(frame, track, "enter", source, now)
                if event is not None and notify_enter and not quiet:
                    events.append(event)

            for track in update.active:
                # Незнакомый трек догоняется распознаванием на следующих кадрах,
                # пока лицо не попадёт в кадр чётко. Сработало - фиксируем имя,
                # чтобы тревога снялась и прозвучало приветствие.
                if identity == "identity" and not track.identified and (
                    self._should_identify(source, track, now) or self._voice_alarming()
                ):
                    name, conf = self._identify_crop(frame, track.box)
                    track.person_name = name
                    track.confidence = conf
                    track.identified = name is not None
                    if name is not None:
                        newly_v.append(_voice_track(source, track))
                track.zone = zone_by_box.get(track.box, track.zone)
                active_v.append(_voice_track(source, track))
                preview_items.append((
                    track.box,
                    f"{track.person_name or '?'} {track.confidence:.2f}".strip(),
                    (40, 200, 80) if track.person_name else (40, 40, 220),
                ))

            for track in update.exited:
                with self._identity_lock:
                    self._identity_attempts.pop((source, track.id), None)
                exited_v.append(_voice_track(source, track))
                key = self._track_key(source, track, "exit")
                if not self._should_store() or not notify_exit:
                    continue
                if not self._cooldown_ok(source, key, now):
                    continue
                self._last_seen[key] = now
                event = self._emit(frame, track, "exit", source, now)
                if event is not None and notify_enter and not quiet:
                    events.append(event)

            with self._lock:
                self._preview_frame = frame
                self._preview_items = preview_items
                self._camera_previews[source] = (frame, preview_items)

        self._steps += 1
        self._present_known = {t.person_name for t in active_v if t.person_name}
        # Retention is a potentially expensive write. A periodic wall-clock
        # task keeps it out of the hot frame path without letting it disappear.
        if time.monotonic() - self._last_prune_at >= 60.0:
            self._storage.prune_events(
                self._runtime.events_max, self._runtime.events_ttl_days,
            )
            self._last_prune_at = time.monotonic()
        if self._voice is not None:
            notices = self._voice.observe(VoiceScene(
                armed=self._runtime.armed,
                identity=self._runtime.detect_mode == "identity",
                quiet=self._runtime.is_quiet(),
                entered=entered_v,
                active=active_v,
                exited=exited_v,
                newly_identified=newly_v,
            ))
            for notice in notices:
                events.append(DetectionEvent(
                    None, 0.0, None, now,
                    event_type=notice.event_type,
                    caption=notice.caption,
                ))
        return events

    def _detect_frames(
        self,
        frames: list[tuple[str, np.ndarray, str]],
        zones: dict,
    ) -> dict[str, list[tuple[Box, str | None]]]:
        """Прогнать инференс по кадрам, параллельно при нескольких камерах.

        Каждый воркер использует свой экземпляр детектора. Это убирает общий
        неблокирующий lock вокруг модели и не даёт тяжёлой камере задерживать
        остальные. Сбой одной камеры не роняет шаг.
        """
        if not frames:
            return {}
        if not self._parallel_enabled():
            return {
                source: filter_boxes(
                    self._detector.detect(frame), zones, frame.shape,
                )
                for source, frame, _identity in frames
            }

        skip_yolo = self._detector.backend == "motion"
        results: dict[str, list[tuple[Box, str | None]]] = {}
        jobs: list[tuple[str, np.ndarray, Detector]] = []
        for source, frame, _identity in frames:
            tracker = self._tracker(source)
            run_det = True
            if self.use_motion_gate and not skip_yolo and tracker.active_count == 0:
                if not self._gate(source).moved(frame):
                    run_det = False
            jobs.append((source, frame, self._worker_detector(source) if run_det else None))

        def run(job: tuple[str, np.ndarray, Detector | None]) -> tuple[str, list | None]:
            source, frame, detector = job
            if detector is None:
                return source, []
            try:
                return source, filter_boxes(detector.detect(frame), zones, frame.shape)
            except Exception as exc:  # noqa: BLE001
                print(f"[pipeline] детекция камеры {source} упала: {exc}")
                return source, []

        workers = min(len(jobs), max(1, len(self._cameras)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for source, tagged in pool.map(run, jobs):
                results[source] = tagged
        return results

    @property
    def running(self) -> bool:
        return self._running

    def snapshot(self, source: str | None = None) -> bytes | None:
        """Кэш последнего кадра. Не читает камеру (нет гонки с циклом)."""
        with self._lock:
            if source is None:
                frame = self._preview_frame
                items = list(self._preview_items)
            else:
                preview = self._camera_previews.get(source)
                frame = preview[0] if preview is not None else None
                items = list(preview[1]) if preview is not None else []
        if frame is None:
            return None
        if self._runtime.privacy_mode != "off":
            vis = privacy_frame(
                frame, [box for box, _label, _color in items],
                self._runtime.privacy_mode, self._runtime.privacy_blocks,
            )
        else:
            vis = annotate(frame, items) if items else frame
        return encode_jpeg(vis, self._runtime.jpeg_quality)

    def camera_info(self, source: str) -> dict[str, object] | None:
        """Состояние камеры для панели, без открытия или чтения источника."""
        for name, camera in self._cameras:
            if name == source:
                with self._lock:
                    has_frame = source in self._camera_previews
                return {
                    "healthy": bool(getattr(camera, "healthy", True)),
                    "last_error": getattr(camera, "last_error", None),
                    "reconnects": int(getattr(camera, "reconnects", 0)),
                    "has_frame": has_frame,
                }
        return None

    def capture_face_embedding(self) -> bytes | None:
        """Снять эталон лица с последнего кадра камеры без сохранения фото."""
        recognizer = self._recognizer
        if recognizer is None or not recognizer.available():
            return None
        with self._lock:
            frame = self._preview_frame.copy() if self._preview_frame is not None else None
            boxes = [box for box, _label, _color in self._preview_items]
        if frame is None:
            return None
        # Детектор уже выбрал человека. Берём его кроп, чтобы лицо случайного
        # прохожего на дальнем плане не стало эталоном.
        target = crop_box(frame, boxes[0], pad=0.1) if boxes else frame
        return recognizer.embed(target) if target is not None else None

    def health_text(self) -> str:
        lines = [
            f"Пайплайн: {'on' if self._running else 'off'}",
            f"Детектор факт: {self._detector.backend}",
        ]
        err, using = self._detector_error()
        if err:
            lines.append(f"Remote ошибка: {err}")
        if using:
            lines.append("Remote: fallback")
        parallel = self._parallel_enabled()
        if parallel:
            lines.append("Камеры: параллельно (свой детектор на камеру)")
        for index, (name, cam) in enumerate(self._cameras):
            label = self._runtime.camera_label(index, name)
            ok = getattr(cam, "healthy", True)
            cerr = getattr(cam, "last_error", None)
            rec = getattr(cam, "reconnects", 0)
            mark = "ok" if ok else "fail"
            extra = f", {cerr}" if cerr else ""
            rec_s = f", reconnects={rec}" if rec else ""
            lines.append(f"Камера {label}: {mark}{extra}{rec_s}")
        tracks = sum(t.active_count for t in self._trackers.values())
        lines.append(f"Треков: {tracks}")
        lines.append(f"Событий в БД: {self._storage.count_events()}")
        if self._voice is not None:
            lines.append(self._voice.status_line())
        return "\n".join(lines)


def _voice_track(source: str, track: Track) -> VoiceTrack:
    return VoiceTrack(
        track_id=f"{source}:{track.id}",
        person_name=track.person_name,
        zone=track.zone,
        source=source,
        identified=track.identified,
    )
