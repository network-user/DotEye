"""Воспроизведение WAV и генерация сирены тревоги.

Плеер выбирается по платформе: winsound (Windows), afplay (macOS),
aplay/paplay/ffplay (Linux). В тестах и без устройства - DummyPlayer.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from abc import ABC, abstractmethod

import numpy as np

SAMPLE_RATE = 22050


def pcm16_to_wav(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Собрать WAV (mono PCM16) из массива сэмплов."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buf.getvalue()


def silence_wav(seconds: float = 0.05, sample_rate: int = SAMPLE_RATE) -> bytes:
    n = max(1, int(sample_rate * seconds))
    return pcm16_to_wav(np.zeros(n, dtype=np.int16), sample_rate)


def is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def scale_wav_volume(data: bytes, volume: float) -> bytes:
    """Масштабировать амплитуду PCM16 WAV. volume 0..1."""
    volume = max(0.0, min(1.0, float(volume)))
    if not is_wav(data) or volume >= 0.999:
        return data
    if volume <= 0.001:
        return silence_wav(0.01)
    src = io.BytesIO(data)
    try:
        with wave.open(src, "rb") as wf:
            params = wf.getparams()
            raw = wf.readframes(wf.getnframes())
            sampwidth = wf.getsampwidth()
            nchannels = wf.getnchannels()
    except wave.Error:
        return data
    if sampwidth != 2:
        return data
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    pcm = np.clip(pcm * volume, -32768, 32767).astype(np.int16)
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(2)
        wf.setframerate(params.framerate)
        wf.writeframes(pcm.tobytes())
    return out.getvalue()


def fade_out_wav(data: bytes, seconds: float = 0.4) -> bytes:
    """Плавно затушить хвост WAV, чтобы останов звука не был резким."""
    seconds = max(0.0, float(seconds))
    if not is_wav(data) or seconds <= 0.0:
        return data
    src = io.BytesIO(data)
    try:
        with wave.open(src, "rb") as wf:
            params = wf.getparams()
            raw = wf.readframes(wf.getnframes())
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
    except wave.Error:
        return data
    if sampwidth != 2:
        return data
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    n = pcm.size
    fade = int(framerate * seconds)
    if fade <= 0 or n == 0:
        return data
    fade = min(fade, n)
    env = np.ones(n, dtype=np.float32)
    env[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(params.nchannels)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes((pcm * env).astype(np.int16).tobytes())
    return out.getvalue()


def generate_siren(
    duration: float = 2.2,
    sample_rate: int = SAMPLE_RATE,
    volume: float = 0.45,
    *,
    fade_edges: bool = True,
) -> bytes:
    """Двухтональная сирена (880/1180 Гц), без внешних файлов.

    При fade_edges=False края не ослабляются - клип закольцовывается без щелчка
    в SND_LOOP.
    """
    duration = max(0.3, float(duration))
    volume = max(0.0, min(1.0, float(volume)))
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    period = 0.28
    high = ((t / period).astype(np.int32) % 2) == 1
    freq = np.where(high, 1180.0, 880.0).astype(np.float32)
    wave_f = np.sin(2.0 * np.pi * freq * t)
    if fade_edges:
        fade = max(1, int(0.02 * sample_rate))
        envelope = np.ones(n, dtype=np.float32)
        envelope[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        wave_f = wave_f * envelope
    pcm = (wave_f * (32767.0 * volume)).astype(np.int16)
    return pcm16_to_wav(pcm, sample_rate)


def mix_wav(*parts: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Смешать несколько WAV (PCM16) в один ролик по наибольшей длине.

    Сумма сэмплов с клиппингом - сирена и речь звучат одновременно одним
    потоком winsound.
    """
    pcm16_parts: list[np.ndarray] = []
    length = 0
    for data in parts:
        if not is_wav(data):
            continue
        src = io.BytesIO(data)
        try:
            with wave.open(src, "rb") as wf:
                if wf.getsampwidth() != 2:
                    raise wave.Error
                raw = wf.readframes(wf.getnframes())
        except wave.Error:
            continue
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.int32)
        length = max(length, pcm.size)
        pcm16_parts.append(pcm)
    if not pcm16_parts:
        return silence_wav(0.05, sample_rate)
    mix = np.zeros(length, dtype=np.int32)
    for pcm in pcm16_parts:
        mix[: pcm.size] += pcm
    mix = np.clip(mix, -32768, 32767).astype(np.int16)
    return pcm16_to_wav(mix, sample_rate)


class Player(ABC):
    """Блокирующее воспроизведение WAV и останов из другого потока."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def available(self) -> bool:
        return True

    @abstractmethod
    def play_wav(self, data: bytes) -> None:
        ...

    def play_loop(self, data: bytes) -> None:
        """Зациклить WAV, пока не вызовут stop(). По умолчанию - разовый play."""
        self.play_wav(data)

    @abstractmethod
    def stop(self) -> None:
        ...


class DummyPlayer(Player):
    def __init__(self) -> None:
        self.plays: list[int] = []
        self.loops: list[int] = []
        self.stop_calls = 0

    @property
    def name(self) -> str:
        return "dummy"

    def available(self) -> bool:
        return False

    def play_wav(self, data: bytes) -> None:
        self.plays.append(len(data))

    def play_loop(self, data: bytes) -> None:
        self.loops.append(len(data))

    def stop(self) -> None:
        self.stop_calls += 1


class WinsoundPlayer(Player):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path: str | None = None

    @property
    def name(self) -> str:
        return "winsound"

    def play_wav(self, data: bytes) -> None:
        import winsound

        path = _write_temp_wav(data)
        with self._lock:
            self._path = path
        try:
            winsound.PlaySound(path, winsound.SND_FILENAME)
        finally:
            _unlink_quiet(path)
            with self._lock:
                if self._path == path:
                    self._path = None

    def play_loop(self, data: bytes) -> None:
        import winsound

        with self._lock:
            if self._path is not None:
                _unlink_quiet(self._path)
            path = _write_temp_wav(data)
            self._path = path
        # SND_ASYNC | SND_LOOP зацикливает без блокировки; SND_PURGE в stop()
        # останавливает и освобождает файл.
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)

    def stop(self) -> None:
        import winsound

        with self._lock:
            path = self._path
            self._path = None
        winsound.PlaySound(None, winsound.SND_PURGE)
        if path:
            _unlink_quiet(path)


class ProcessPlayer(Player):
    """Внешний проигрыватель: aplay, paplay, afplay, ffplay."""

    def __init__(self, binary: str, args: list[str] | None = None) -> None:
        self._binary = binary
        self._args = args or []
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def name(self) -> str:
        return self._binary

    def play_wav(self, data: bytes) -> None:
        path = _write_temp_wav(data)
        cmd = [self._binary, *self._args, path]
        try:
            with self._lock:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc = self._proc
            proc.wait()
        finally:
            with self._lock:
                self._proc = None
            _unlink_quiet(path)

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def _write_temp_wav(data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="doteye_")
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return path


def _unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def build_player() -> Player:
    if sys.platform == "win32":
        return WinsoundPlayer()
    if sys.platform == "darwin" and shutil.which("afplay"):
        return ProcessPlayer("afplay")
    if shutil.which("aplay"):
        return ProcessPlayer("aplay", ["-q"])
    if shutil.which("paplay"):
        return ProcessPlayer("paplay")
    if shutil.which("ffplay"):
        return ProcessPlayer(
            "ffplay",
            ["-nodisp", "-autoexit", "-loglevel", "quiet"],
        )
    return DummyPlayer()
