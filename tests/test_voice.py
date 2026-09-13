"""Стейт-машина тревоги и озвучки без реального динамика."""

from __future__ import annotations

from pathlib import Path

from doteye.audio import DummyPlayer
from doteye.config import Settings
from doteye.runtime import Runtime
from doteye.storage import Storage
from doteye.tts import DummyTTS, build_tts
from doteye.voice import (
    VoiceEngine,
    VoiceScene,
    VoiceTrack,
    build_voice,
    format_phrase,
)


def _engine(tmp_path: Path, **over: object) -> tuple[VoiceEngine, DummyTTS, DummyPlayer, Runtime]:
    settings = Settings(**over)  # type: ignore[arg-type]
    storage = Storage(tmp_path / "voice.db")
    runtime = Runtime(settings, storage)
    tts = DummyTTS()
    player = DummyPlayer()
    engine = build_voice(runtime, tts, player, start_worker=False)
    return engine, tts, player, runtime


def _unknown(tid: str = "cam:1") -> VoiceTrack:
    return VoiceTrack(tid, None, None, "cam", False)


def _known(name: str = "Вася", tid: str = "cam:1") -> VoiceTrack:
    return VoiceTrack(tid, name, None, "cam", True)


def test_format_phrase_missing_key() -> None:
    assert format_phrase("Привет, {name}.", name="Анна") == "Привет, Анна."
    assert format_phrase("Привет, {name}.") == "Привет, ."
    assert format_phrase("plain") == "plain"


def test_unknown_enter_raises_alarm(tmp_path: Path) -> None:
    engine, tts, player, _rt = _engine(tmp_path, detect_mode="identity")
    track = _unknown()
    notices = engine.observe(VoiceScene(
        armed=True, identity=True, quiet=False,
        entered=[track], active=[track],
    ))
    assert engine.alarming
    assert notices and notices[0].event_type == "alarm"
    engine.drain()
    assert any("посторонний" in text.casefold() for text in tts.texts)
    # Непрерывная сирена собирается в зацикленный клип, а не в разовые play.
    assert engine._siren_active
    assert engine._siren_clip


def test_known_face_clears_alarm(tmp_path: Path) -> None:
    engine, tts, player, _rt = _engine(tmp_path)
    unknown = _unknown()
    engine.observe(VoiceScene(
        armed=True, identity=True, quiet=False,
        entered=[unknown], active=[unknown],
    ))
    assert engine.alarming
    known = _known("Анна")
    notices = engine.observe(VoiceScene(
        armed=True, identity=True, quiet=False,
        active=[known], newly_identified=[known],
    ))
    assert engine.alarming is False
    assert any(notice.event_type == "alarm_cleared" for notice in notices)
    engine.drain()
    blob = " ".join(tts.texts).casefold()
    assert "снята" in blob
    assert "анна" in blob
    assert player.stop_calls >= 1


def test_known_enter_welcomes_without_alarm(tmp_path: Path) -> None:
    engine, tts, _player, _rt = _engine(tmp_path)
    known = _known("Боря")
    notices = engine.observe(VoiceScene(
        armed=True, identity=True, quiet=False,
        entered=[known], active=[known],
    ))
    assert engine.alarming is False
    assert notices == []
    engine.drain()
    assert any("боря" in text.casefold() for text in tts.texts)


def test_disarmed_skips_alarm(tmp_path: Path) -> None:
    engine, tts, _player, _rt = _engine(tmp_path, armed=False)
    track = _unknown()
    engine.observe(VoiceScene(
        armed=False, identity=True, quiet=False,
        entered=[track], active=[track],
    ))
    assert engine.alarming is False
    engine.drain()
    assert tts.texts == []


def test_quiet_mutes_welcome_but_not_alarm(tmp_path: Path) -> None:
    engine, tts, _player, runtime = _engine(tmp_path)
    runtime.voice_mute_quiet = True
    known = _known("Катя")
    engine.observe(VoiceScene(
        armed=True, identity=True, quiet=True,
        entered=[known], active=[known],
    ))
    engine.drain()
    assert tts.texts == []

    unknown = _unknown("cam:2")
    engine.observe(VoiceScene(
        armed=True, identity=True, quiet=True,
        entered=[unknown], active=[unknown],
    ))
    assert engine.alarming
    engine.drain()
    assert any("посторонний" in text.casefold() for text in tts.texts)


def test_exit_does_not_clear_alarm(tmp_path: Path) -> None:
    engine, _tts, _player, runtime = _engine(tmp_path)
    runtime.voice_grace_seconds = 0
    runtime.voice_clear_on = "exit"
    unknown = _unknown()
    engine.observe(VoiceScene(
        armed=True, identity=True, quiet=False,
        entered=[unknown], active=[unknown],
    ))
    assert engine.alarming
    notices = engine.observe(VoiceScene(
        armed=True, identity=True, quiet=False,
        exited=[unknown],
    ))
    assert engine.alarming is True
    assert not any(notice.event_type == "alarm_cleared" for notice in notices)


def test_clear_on_known_keeps_alarm_after_exit(tmp_path: Path) -> None:
    engine, _tts, _player, runtime = _engine(tmp_path)
    runtime.voice_clear_on = "known"
    runtime.voice_grace_seconds = 0
    unknown = _unknown()
    engine.observe(VoiceScene(
        armed=True, identity=True, quiet=False,
        entered=[unknown], active=[unknown],
    ))
    engine.observe(VoiceScene(
        armed=True, identity=True, quiet=False,
        exited=[unknown],
    ))
    assert engine.alarming is True


def test_manual_alarm_while_disarmed_stays(tmp_path: Path) -> None:
    engine, _tts, _player, _rt = _engine(tmp_path, armed=False)
    engine.trigger_alarm("Отойди!")
    assert engine.alarming
    engine.observe(VoiceScene(armed=False, identity=True, quiet=False))
    assert engine.alarming is True


def test_manual_announce_and_dismiss(tmp_path: Path) -> None:
    engine, tts, _player, _rt = _engine(tmp_path)
    assert engine.announce("Отойди от двери!")
    engine.drain()
    assert tts.texts[-1] == "Отойди от двери!"
    engine.trigger_alarm("Внимание, тревога.")
    assert engine.alarming
    notice = engine.dismiss("manual")
    assert notice is not None
    assert engine.alarming is False


def test_timeout_clears(tmp_path: Path) -> None:
    engine, _tts, _player, runtime = _engine(tmp_path)
    runtime.voice_timeout_seconds = 0.01
    unknown = _unknown()
    engine.observe(VoiceScene(
        armed=True, identity=True, quiet=False,
        entered=[unknown], active=[unknown],
    ))
    engine._alarm_since = 0.0
    notices = engine.observe(VoiceScene(
        armed=True, identity=True, quiet=False,
        active=[unknown],
    ))
    assert engine.alarming is True
    assert not any(n.event_type == "alarm_cleared" for n in notices)


def test_presence_alarm_optional(tmp_path: Path) -> None:
    engine, _tts, _player, runtime = _engine(tmp_path)
    track = _unknown()
    engine.observe(VoiceScene(
        armed=True, identity=False, quiet=False,
        entered=[track], active=[track],
    ))
    assert engine.alarming is False
    runtime.voice_alarm_on_presence = True
    engine.observe(VoiceScene(
        armed=True, identity=False, quiet=False,
        entered=[track], active=[track],
    ))
    assert engine.alarming is True


def test_phrase_override(tmp_path: Path) -> None:
    engine, tts, _player, runtime = _engine(tmp_path)
    runtime.set_voice_phrase("stranger", "Чужой на объекте.")
    track = _unknown()
    engine.observe(VoiceScene(
        armed=True, identity=True, quiet=False,
        entered=[track], active=[track],
    ))
    engine.drain()
    assert "Чужой на объекте." in tts.texts


def test_build_tts_is_lazy() -> None:
    tts = build_tts()
    assert tts.name


def test_alarm_builds_mixed_clip(tmp_path: Path) -> None:
    """Клип тревоги смешивает сирену и речь в один WAV потока."""
    engine, _tts, player, _rt = _engine(tmp_path)
    from doteye.audio import is_wav

    speech = b""
    import io as _io, wave  # noqa: E402

    buf = _io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00" * 6000)
    speech = buf.getvalue()
    engine._build_alarm_clip(speech, 0.8)
    clip = engine._siren_clip
    assert clip is not None and is_wav(clip)


def test_stop_siren_enqueues_fade(tmp_path: Path) -> None:
    engine, _tts, player, runtime = _engine(tmp_path)
    engine._activate_siren()
    engine._stop_siren()
    assert engine._siren_active is False
    # Fade-джоб попал в очередь.
    jobs = list(engine._queue.queue)
    assert any(job.kind == "fade" for job in jobs)
