"""Тесты WAV-сирены и масштаба громкости."""

from __future__ import annotations

from doteye.audio import (
    DummyPlayer,
    generate_siren,
    is_wav,
    pcm16_to_wav,
    scale_wav_volume,
    silence_wav,
)
import numpy as np


def test_siren_is_wav() -> None:
    data = generate_siren(duration=0.4, volume=0.3)
    assert is_wav(data)
    assert len(data) > 100


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


def test_dummy_player_records() -> None:
    player = DummyPlayer()
    assert player.available() is False
    player.play_wav(b"RIFF")
    player.stop()
    assert player.plays == [4]
    assert player.stop_calls == 1
