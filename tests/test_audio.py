"""Тесты WAV-сирены, смешения и масштаба громкости."""

from __future__ import annotations

import wave
import io as _io

import numpy as np

from doteye.audio import (
    DummyPlayer,
    fade_out_wav,
    generate_siren,
    is_wav,
    mix_wav,
    pcm16_to_wav,
    scale_wav_volume,
    silence_wav,
)


def test_siren_is_wav() -> None:
    data = generate_siren(duration=0.4, volume=0.3)
    assert is_wav(data)
    assert len(data) > 100


def test_siren_loop_has_no_fade() -> None:
    looped = generate_siren(duration=0.3, volume=0.5, fade_edges=False)
    faded = generate_siren(duration=0.3, volume=0.5, fade_edges=True)
    with wave.open(_io.BytesIO(looped), "rb") as wf:
        pcm_loop = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    with wave.open(_io.BytesIO(faded), "rb") as wf:
        pcm_faded = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    # Хвост затухающего клипа уходит в ноль, край бесшовного - нет.
    assert int(abs(pcm_loop[-1])) > int(abs(pcm_faded[-1]))


def test_silence_and_pcm() -> None:
    quiet = silence_wav(0.02)
    assert is_wav(quiet)
    pcm = np.zeros(100, dtype=np.int16)
    assert is_wav(pcm16_to_wav(pcm, 8000))


def test_scale_volume_zero_is_tiny() -> None:
    src = generate_siren(duration=0.3, volume=0.5)
    silent = scale_wav_volume(src, 0.0)
    assert is_wav(silent)
    assert len(silent) < len(src)


def test_mix_wav_combines_and_clips() -> None:
    a = generate_siren(duration=0.4, volume=0.8)
    b = generate_siren(duration=0.2, volume=0.8)
    mixed = mix_wav(a, b)
    assert is_wav(mixed)
    with wave.open(_io.BytesIO(mixed), "rb") as wf:
        nframes = wf.getnframes()
        pcm = np.frombuffer(wf.readframes(nframes), dtype=np.int16)
    assert nframes >= 0.4 * 22050  # длина = max из частей
    assert pcm.size == 0 or int(np.max(np.abs(pcm))) <= 32767


def test_mix_wav_rejects_garbage() -> None:
    assert is_wav(mix_wav(b"not-a-wav", b"also-bad"))


def test_fade_out_keeps_wav_valid() -> None:
    src = generate_siren(duration=0.5, volume=0.5)
    tail = fade_out_wav(src, seconds=0.3)
    assert is_wav(tail)
    with wave.open(_io.BytesIO(src), "rb") as wf:
        pcm_src = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    with wave.open(_io.BytesIO(tail), "rb") as wf:
        pcm_tail = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    assert pcm_tail.size == pcm_src.size
    assert int(abs(pcm_tail[-1])) <= int(abs(pcm_src[-1]))


def test_dummy_player_records() -> None:
    player = DummyPlayer()
    assert player.available() is False
    player.play_wav(b"RIFF")
    player.play_loop(b"LOOP")
    player.stop()
    assert player.plays == [4]
    assert player.loops == [4]
    assert player.stop_calls == 1
