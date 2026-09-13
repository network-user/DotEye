"""Голос и тревога: сирена + TTS, снятие по известному лицу, ручные фразы.

Синтез и воспроизведение идут в отдельном потоке, observe() только обновляет
состояние и кладёт задания в очередь - пайплайн не ждёт динамики.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from doteye.audio import (
    Player,
    build_player,
    generate_siren,
    scale_wav_volume,
)
from doteye.tts import TTS, build_tts

if TYPE_CHECKING:
    from doteye.runtime import Runtime

DEFAULT_PHRASES: dict[str, str] = {
    "stranger": "Внимание. Обнаружен незнакомец.",
    "repeat": "Покиньте помещение немедленно.",
    "cleared": "Тревога снята.",
    "welcome": "Добро пожаловать, {name}.",
    "goodbye": "До свидания, {name}.",
    "presence": "Обнаружен человек.",
    "armed": "Охрана включена.",
    "disarmed": "Охрана выключена.",
}

PHRASE_TITLES: dict[str, str] = {
    "stranger": "Незнакомец",
    "repeat": "Повтор тревоги",
    "cleared": "Тревога снята",
    "welcome": "Приветствие",
    "goodbye": "Прощание",
    "presence": "Присутствие",
    "armed": "Охрана вкл",
    "disarmed": "Охрана выкл",
}

CLEAR_ON_VALUES = ("both", "known", "exit")

VOICE_PREVIEW_PHRASE = "Съешь же ещё этих мягких французских булок, да выпей чаю."


@dataclass(frozen=True)
class VoiceTrack:
    track_id: str
    person_name: str | None
    zone: str | None
    source: str
    identified: bool = False


@dataclass(frozen=True)
class VoiceScene:
    armed: bool
    identity: bool
    quiet: bool
    entered: list[VoiceTrack] = field(default_factory=list)
    active: list[VoiceTrack] = field(default_factory=list)
    exited: list[VoiceTrack] = field(default_factory=list)
    newly_identified: list[VoiceTrack] = field(default_factory=list)


@dataclass(frozen=True)
class VoiceNotice:
    """Короткое сообщение в Telegram (без кадра)."""

    event_type: str
    caption: str


@dataclass
class VoiceJob:
    kind: str
    text: str = ""
    siren: bool = False
    interrupt: bool = False
    voice_slot: str = "default"
    voice_id: str = ""
    force: bool = False
    future: Future[list[tuple[str, str]]] | None = None


class _SafeMap(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""


def format_phrase(template: str, **kwargs: str | None) -> str:
    mapping = _SafeMap(
        {key: (value if value is not None else "") for key, value in kwargs.items()}
    )
    try:
        return template.format_map(mapping).strip()
    except (ValueError, IndexError):
        return template.strip()


class VoiceEngine:
    def __init__(
        self,
        runtime: Runtime,
        tts: TTS,
        player: Player,
    ) -> None:
        self._runtime = runtime
        self._tts = tts
        self._player = player
        self._queue: queue.Queue[VoiceJob] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._alarming = False
        self._alarm_since = 0.0
        self._last_loop_at = 0.0
        self._empty_since: float | None = None
        self._last_armed: bool | None = None
        self._quiet = False
        self._welcome_at: dict[str, float] = {}
        self._voices: list[tuple[str, str]] | None = None
        self.last_phrase = ""
        self.last_error = ""

    @property
    def alarming(self) -> bool:
        return self._alarming

    @property
    def backend(self) -> str:
        tts = self._tts.name
        player = self._player.name
        return f"{tts}+{player}"

    def status_line(self) -> str:
        alarm = "ТРЕВОГА" if self._alarming else "idle"
        tts = "tts" if self._tts.available() else "нет tts"
        audio = "audio" if self._player.available() else "нет динамика"
        extra = f", «{self.last_phrase}»" if self.last_phrase else ""
        err = f", ошибка {self.last_error}" if self.last_error else ""
        return f"Голос: {self.backend} ({alarm}, {tts}, {audio}{extra}{err})"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker, name="doteye-voice", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._player.stop()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def list_voices(self, timeout: float = 8.0) -> list[tuple[str, str]]:
        if self._voices is not None:
            return self._voices
        if self._thread is None:
            self._tts.ensure()
            self._voices = self._tts.list_voices()
            return self._voices
        fut: Future[list[tuple[str, str]]] = Future()
        self._queue.put(VoiceJob(kind="list_voices", future=fut))
        try:
            return fut.result(timeout=timeout)
        except Exception:
            return []

    def preview_voice(self, voice_id: str) -> None:
        """Озвучить пример голоса через динамики, вне очереди авто-фраз."""
        if not voice_id:
            return
        self._enqueue(VoiceJob(
            kind="manual", text=VOICE_PREVIEW_PHRASE,
            interrupt=True, voice_id=voice_id, force=True,
        ))

    def announce(self, text: str) -> bool:
        """Ручная озвучка из чата. Работает даже при выключенных авто-фразах."""
        cleaned = " ".join((text or "").split())[:500]
        if not cleaned:
            return False
        self._enqueue(VoiceJob(kind="manual", text=cleaned, interrupt=True))
        return True

    def test_alarm(self) -> bool:
        """Поставить тест сирены в очередь и сообщить, есть ли аудиовыход."""
        phrase = self._phrase("stranger")
        self._enqueue(VoiceJob(
            kind="test", text=phrase, siren=True, interrupt=True, voice_slot="alarm",
        ))
        return self._player.available()

    def trigger_alarm(self, text: str | None = None) -> bool:
        phrase = (text or "").strip() or self._phrase("stranger")
        with self._lock:
            return self._raise_alarm(
                source="ручная", zone=None, phrase=phrase, require_armed=False,
            ) is not None

    def dismiss(self, reason: str = "manual") -> VoiceNotice | None:
        with self._lock:
            return self._clear_alarm(who=None, reason=reason, announce=True)

    def sync_armed(self, armed: bool) -> None:
        with self._lock:
            self._apply_armed(armed, force_announce=True)

    def observe(self, scene: VoiceScene) -> list[VoiceNotice]:
        now = time.time()
        notices: list[VoiceNotice] = []
        with self._lock:
            self._quiet = scene.quiet
            self._apply_armed(scene.armed)

            unknown_active = [t for t in scene.active if not t.person_name]
            known_new = [
                t for t in list(scene.newly_identified) + list(scene.entered)
                if t.person_name
            ]
            known_names: list[str] = []
            for track in known_new:
                name = track.person_name or ""
                if name and name not in known_names:
                    known_names.append(name)

            if scene.armed and self._should_raise(scene):
                first = next(
                    (t for t in scene.entered if not t.person_name),
                    scene.entered[0] if scene.entered else None,
                )
                source = first.source if first else ""
                zone = first.zone if first else None
                notice = self._raise_alarm(source=source, zone=zone)
                if notice is not None:
                    notices.append(notice)

            clear_on = self._clear_mode()
            if self._alarming and known_names and clear_on in ("known", "both"):
                notice = self._clear_alarm(
                    who=known_names[0], reason="known", announce=True,
                )
                if notice is not None:
                    notices.append(notice)
                    self._maybe_welcome(known_names[0], now, skip=True)
                    known_names = known_names[1:]

            if unknown_active:
                self._empty_since = None
            elif self._alarming and self._empty_since is None:
                self._empty_since = now

            if (
                self._alarming
                and clear_on in ("exit", "both")
                and self._empty_since is not None
                and now - self._empty_since >= self._runtime.voice_grace_seconds
            ):
                notice = self._clear_alarm(who=None, reason="exit", announce=True)
                if notice is not None:
                    notices.append(notice)

            timeout = self._maybe_timeout(now)
            if timeout is not None:
                notices.append(timeout)

            if self._runtime.voice_welcome and not self._alarming:
                for name in known_names:
                    self._maybe_welcome(name, now, skip=False)

            if self._runtime.voice_goodbye:
                for track in scene.exited:
                    if track.person_name:
                        self._maybe_goodbye(track.person_name, now)

            if (
                self._runtime.voice_presence
                and not scene.identity
                and not self._runtime.voice_alarm_on_presence
                and not self._alarming
            ):
                if scene.entered and self._can_speech(alarm=False):
                    self._enqueue(VoiceJob(
                        kind="speech", text=self._phrase("presence"),
                    ))

            self._maybe_repeat(now)
        return notices

    def drain(self) -> None:
        """Синхронно проиграть очередь (тесты, без воркера)."""
        self._tts.ensure()
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job.kind == "list_voices" and job.future is not None:
                job.future.set_result(self._tts.list_voices())
                continue
            self._play_job(job)

    # -- внутренности ---------------------------------------------------

    def _clear_mode(self) -> str:
        value = (self._runtime.voice_clear_on or "both").strip().lower()
        return value if value in CLEAR_ON_VALUES else "both"

    def _phrase(self, key: str, **kwargs: str | None) -> str:
        template = self._runtime.voice_phrase(key)
        return format_phrase(template, **kwargs)

    def _can_speech(self, *, alarm: bool) -> bool:
        if alarm:
            return self._runtime.voice_speech_enabled
        if not self._runtime.voice_enabled:
            return False
        if self._quiet and self._runtime.voice_mute_quiet:
            return False
        return self._runtime.voice_speech_enabled

    def _can_siren(self) -> bool:
        return self._runtime.voice_siren_enabled

    def _should_raise(self, scene: VoiceScene) -> bool:
        if self._alarming:
            return False
        if not self._runtime.voice_enabled or not self._runtime.voice_alarm_enabled:
            return False
        if not scene.armed:
            return False
        unknown_enter = [t for t in scene.entered if not t.person_name]
        if scene.identity:
            return bool(unknown_enter)
        return bool(scene.entered) and self._runtime.voice_alarm_on_presence

    def _apply_armed(self, armed: bool, force_announce: bool = False) -> None:
        first = self._last_armed is None
        changed = self._last_armed is not None and armed != self._last_armed
        self._last_armed = armed
        if not armed and changed:
            self._clear_alarm(who=None, reason="disarm", announce=False)
        if first and not force_announce:
            return
        if not (changed or force_announce):
            return
        if not self._runtime.voice_armed_announce:
            return
        if not self._can_speech(alarm=False):
            return
        key = "armed" if armed else "disarmed"
        self._enqueue(VoiceJob(kind="speech", text=self._phrase(key), interrupt=True))

    def _raise_alarm(
        self,
        source: str,
        zone: str | None,
        phrase: str | None = None,
        require_armed: bool = True,
    ) -> VoiceNotice | None:
        if require_armed and not self._last_armed:
            return None
        already = self._alarming
        now = time.time()
        self._alarming = True
        if not already:
            self._alarm_since = now
            self._last_loop_at = now
            self._empty_since = None
            text = phrase or self._phrase("stranger")
            self._enqueue(VoiceJob(
                kind="alarm", text=text, siren=True, interrupt=True, voice_slot="alarm",
            ))
            where = f" камера {source}" if source else ""
            zone_s = f", зона {zone}" if zone else ""
            return VoiceNotice(
                "alarm",
                f"DotEye: ТРЕВОГА.{where}{zone_s}\n{text}",
            )
        return None

    def _clear_alarm(
        self,
        who: str | None,
        reason: str,
        announce: bool = True,
    ) -> VoiceNotice | None:
        if not self._alarming:
            return None
        self._alarming = False
        self._empty_since = None
        try:
            self._player.stop()
        except Exception:
            pass
        who_s = who or ""
        if reason == "known" and who_s:
            caption = f"DotEye: тревога снята ({who_s})."
        elif reason == "exit":
            caption = "DotEye: тревога снята (незнакомец ушёл)."
        elif reason == "timeout":
            caption = "DotEye: тревога снята (таймаут)."
        elif reason == "disarm":
            caption = "DotEye: тревога снята (охрана выкл)."
        else:
            caption = "DotEye: тревога снята."
        if announce:
            parts = [self._phrase("cleared")]
            if who_s and self._runtime.voice_welcome:
                welcome = self._phrase("welcome", name=who_s)
                if welcome:
                    parts.append(welcome)
            text = " ".join(p for p in parts if p)
            if text:
                self._enqueue(VoiceJob(
                    kind="speech", text=text, siren=False, interrupt=True, voice_slot="alarm",
                ))
        return VoiceNotice("alarm_cleared", caption)

    def _maybe_timeout(self, now: float) -> VoiceNotice | None:
        limit = self._runtime.voice_timeout_seconds
        if self._alarming and limit > 0 and now - self._alarm_since >= limit:
            return self._clear_alarm(who=None, reason="timeout", announce=True)
        return None

    def _maybe_repeat(self, now: float) -> None:
        if not self._alarming:
            return
        if not self._runtime.voice_enabled:
            return
        if not self._queue.empty():
            return
        repeat = max(2.0, self._runtime.voice_repeat_seconds)
        if now - self._last_loop_at < repeat:
            return
        self._last_loop_at = now
        self._enqueue(VoiceJob(
            kind="alarm", text=self._phrase("repeat"), siren=True, voice_slot="alarm",
        ))

    def _cooldown_ok(self, key: str, now: float) -> bool:
        gap = max(0.0, self._runtime.voice_cooldown_seconds)
        return now - self._welcome_at.get(key, 0.0) >= gap

    def _maybe_welcome(self, name: str, now: float, skip: bool) -> None:
        if skip:
            self._welcome_at[f"in:{name}"] = now
            return
        key = f"in:{name}"
        if not self._cooldown_ok(key, now):
            return
        if not self._can_speech(alarm=False):
            return
        text = self._phrase("welcome", name=name)
        if not text:
            return
        self._welcome_at[key] = now
        self._enqueue(VoiceJob(kind="speech", text=text, voice_slot="welcome"))

    def _maybe_goodbye(self, name: str, now: float) -> None:
        key = f"out:{name}"
        if not self._cooldown_ok(key, now):
            return
        if not self._can_speech(alarm=False):
            return
        text = self._phrase("goodbye", name=name)
        if not text:
            return
        self._welcome_at[key] = now
        self._enqueue(VoiceJob(kind="speech", text=text, voice_slot="welcome"))

    def _enqueue(self, job: VoiceJob) -> None:
        if job.text:
            self.last_phrase = job.text[:80]
        self._queue.put(job)

    def _worker(self) -> None:
        try:
            self._tts.ensure()
            self._voices = self._tts.list_voices()
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            print(f"[voice] tts init: {exc}")
        print(f"[voice] started ({self.backend})")
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                with self._lock:
                    self._maybe_repeat(time.time())
                continue
            if job.kind == "list_voices":
                try:
                    voices = self._tts.list_voices()
                    self._voices = voices
                    if job.future is not None and not job.future.done():
                        job.future.set_result(voices)
                except Exception as exc:  # noqa: BLE001
                    if job.future is not None and not job.future.done():
                        job.future.set_exception(exc)
                continue
            try:
                self._play_job(job)
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                print(f"[voice] play: {exc}")
        print("[voice] stopped")

    def _play_job(self, job: VoiceJob) -> None:
        volume = max(0.0, min(1.0, self._runtime.voice_volume))
        rate = max(0.4, min(2.5, self._runtime.voice_rate))
        voice_id = job.voice_id or self._voice_id(job.voice_slot)
        want_siren = job.siren and self._can_siren()
        preview = job.kind == "manual" and job.force
        want_speech = bool(job.text) and (
            preview or job.kind == "manual" or self._runtime.voice_speech_enabled
        )
        if volume <= 0.001:
            return
        if want_siren:
            wav = scale_wav_volume(generate_siren(volume=0.5), volume)
            print("[voice] siren")
            self._player.play_wav(wav)
        if want_speech:
            wav = self._tts.synthesize(job.text, rate=rate, voice_id=voice_id)
            if wav:
                wav = scale_wav_volume(wav, volume)
                print(f"[voice] say: {job.text[:80]}")
                self._player.play_wav(wav)
            else:
                detail = self._tts.last_error or "синтезатор не вернул аудио"
                self.last_error = f"Синтез речи недоступен: {detail}"
                print(f"[voice] tts пуст: {job.text[:80]}")

    def _voice_id(self, slot: str) -> str:
        if slot == "alarm":
            return self._runtime.voice_alarm_tts_voice or self._runtime.voice_tts_voice
        if slot == "welcome":
            return self._runtime.voice_welcome_tts_voice or self._runtime.voice_tts_voice
        return self._runtime.voice_tts_voice


def build_voice(
    runtime: Runtime,
    tts: TTS | None = None,
    player: Player | None = None,
    start_worker: bool = True,
) -> VoiceEngine:
    engine = VoiceEngine(runtime, tts or build_tts(), player or build_player())
    if start_worker:
        engine.start()
    return engine
