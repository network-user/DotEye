"""Синтез речи: pyttsx3 (SAPI/espeak), затем espeak-ng, иначе заглушка.

Инициализация COM/движка только в потоке, который вызывает ensure()
и synthesize() - pyttsx3 не потокобезопасен.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

from doteye.audio import is_wav, silence_wav


class TTS(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def available(self) -> bool:
        return False

    @property
    def last_error(self) -> str:
        return ""

    def ensure(self) -> None:
        return None

    def list_voices(self) -> list[tuple[str, str]]:
        return []

    @abstractmethod
    def synthesize(
        self, text: str, rate: float = 1.0, voice_id: str = ""
    ) -> bytes | None:
        ...


# Маркеры, по которым голос считается русским. Проверяются в id, имени и
# списке языков голоса. Реальные id Windows (SAPI5/SAPI) часто содержат
# "MSSpeech", "TTS_MS_...RU..." или имена "Ирина", "Pavel", "Elena".
RUSSIAN_VOICE_MARKERS = (
    "ru-ru",
    "russian",
    "россия",
    "русск",
    "ирина",
    "irina",
    "pavel",
    "павел",
    "elena",
    "елена",
    "zira",
    "katya",
    "катя",
    "ms-ru",
    "ru-",
    "_ru",
)


def is_russian_voice(voice: object) -> bool:
    """Определить русский голос по id, имени и списку языков."""
    blob = " ".join(
        str(getattr(voice, attr, "") or "") for attr in ("id", "name")
    ).casefold()
    langs = getattr(voice, "languages", None) or []
    for lang in langs:
        if isinstance(lang, (bytes, bytearray)):
            try:
                lang = lang.decode("utf-8", "ignore")
            except Exception:
                lang = ""
        blob += " " + str(lang).casefold()
    return any(marker in blob for marker in RUSSIAN_VOICE_MARKERS)


def _voice_title(voice: object) -> str:
    vid = str(getattr(voice, "id", "") or "")
    name = str(getattr(voice, "name", "") or "")
    return name or vid


def _is_russian_token(vid: str, name: str) -> bool:
    blob = f"{vid} {name}".casefold()
    return any(marker in blob for marker in RUSSIAN_VOICE_MARKERS)


class DummyTTS(TTS):
    def __init__(self) -> None:
        self.texts: list[str] = []

    @property
    def name(self) -> str:
        return "dummy"

    def available(self) -> bool:
        return False

    def list_voices(self) -> list[tuple[str, str]]:
        return []

    def synthesize(
        self, text: str, rate: float = 1.0, voice_id: str = ""
    ) -> bytes | None:
        self.texts.append(text)
        return silence_wav(0.04)


class Pyttsx3TTS(TTS):
    def __init__(self) -> None:
        self._engine = None
        self._ok = False
        self._error = ""

    @property
    def name(self) -> str:
        return "pyttsx3"

    def available(self) -> bool:
        return self._ok

    @property
    def last_error(self) -> str:
        return self._error

    def ensure(self) -> None:
        if self._engine is not None or self._error:
            return
        try:
            import pyttsx3
        except ImportError as exc:
            self._error = str(exc)
            return
        try:
            engine = pyttsx3.init()
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            return
        self._engine = engine
        self._ok = True
        self._pick_russian()

    def _pick_russian(self) -> None:
        if self._engine is None:
            return
        try:
            voices = self._engine.getProperty("voices") or []
        except Exception:
            return
        for voice in voices:
            if is_russian_voice(voice):
                self._engine.setProperty("voice", voice.id)
                return

    def list_voices(self) -> list[tuple[str, str]]:
        self.ensure()
        if self._engine is None:
            return []
        out: list[tuple[str, str]] = []
        try:
            voices = self._engine.getProperty("voices") or []
        except Exception:
            return []
        for voice in voices:
            if not is_russian_voice(voice):
                continue
            vid = str(getattr(voice, "id", "") or "")
            if vid:
                out.append((vid, _voice_title(voice)))
        return out

    def synthesize(
        self, text: str, rate: float = 1.0, voice_id: str = ""
    ) -> bytes | None:
        self.ensure()
        if self._engine is None:
            return None
        try:
            if voice_id:
                self._engine.setProperty("voice", voice_id)
            base = 175
            self._engine.setProperty(
                "rate", max(80, min(400, int(base * max(0.4, min(2.5, rate)))))
            )
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="doteye_tts_")
            os.close(fd)
            try:
                self._engine.save_to_file(text, path)
                self._engine.runAndWait()
                # SAPI иногда завершает runAndWait до того, как дописывает
                # WAV. Короткое ожидание не позволяет передать в winsound
                # пустой файл и превратить «озвучиваю» в тишину.
                data = b""
                until = time.monotonic() + 2.0
                while time.monotonic() < until:
                    data = Path(path).read_bytes()
                    if is_wav(data) and len(data) > 44:
                        break
                    time.sleep(0.05)
            finally:
                Path(path).unlink(missing_ok=True)
            return data if is_wav(data) else None
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            return None


class Sapi5TTS(TTS):
    """Синтез напрямую через SAPI (win32com), минуя сломанный event-pump pyttsx3.

    pyttsx3 использует свой цикл сообщений SAPI, который зависает на повторном
    runAndWait() в том же потоке: первое «озвучивание» работает, следующее
    блокируется навсегда. Прямой SpVoice.Speak() в файл стабилен при повторных
    вызовах в одном выделенном потоке с инициализированным COM.
    """

    def __init__(self) -> None:
        self._voice = None
        self._stream = None
        self._ok = False
        self._error = ""
        self._voices: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "sapi5"

    def available(self) -> bool:
        return self._ok

    @property
    def last_error(self) -> str:
        return self._error

    def ensure(self) -> None:
        if self._voice is not None or self._error:
            return
        try:
            import pythoncom  # noqa: F401
            import win32com.client
        except ImportError as exc:
            self._error = str(exc)
            return
        try:
            pythoncom.CoInitialize()
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            # Прогрев нужен: первый Speak асинхронный, дальше - синхронный.
            self._voice = voice
            self._stream = stream
            self._ok = True
            self._voices = self._collect_voices(voice)
            self._pick_russian()
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)

    def _collect_voices(self, voice: object) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        try:
            for token in voice.GetVoices():
                vid = str(token.Id)
                name = str(token.GetDescription() or "")
                if not vid:
                    continue
                title = name or vid
                if _is_russian_token(vid, name):
                    out.append((vid, title))
        except Exception:  # noqa: BLE001
            return []
        return out

    def _pick_russian(self) -> None:
        for vid, _title in self._voices:
            try:
                token = next(t for t in self._voice.GetVoices() if str(t.Id) == vid)  # type: ignore[union-attr]
                self._voice.Voice = token  # type: ignore[union-attr]
                return
            except Exception:  # noqa: BLE001
                continue

    def _set_voice(self, voice_id: str) -> None:
        if not voice_id or self._voice is None:
            return
        try:
            for token in self._voice.GetVoices():
                if str(token.Id) == voice_id:
                    self._voice.Voice = token
                    return
        except Exception:  # noqa: BLE001
            return

    def _set_rate(self, rate: float) -> None:
        if self._voice is None:
            return
        # SAPI Rate - от -10 до 10; используем плавную шкалу.
        value = int(round(max(-10.0, min(10.0, (rate - 1.0) * 5.0))))
        try:
            self._voice.Rate = value
        except Exception:  # noqa: BLE001
            return

    def list_voices(self) -> list[tuple[str, str]]:
        self.ensure()
        return list(self._voices)

    def synthesize(
        self, text: str, rate: float = 1.0, voice_id: str = ""
    ) -> bytes | None:
        self.ensure()
        if self._voice is None or self._stream is None:
            return None
        path = ""
        try:
            if voice_id:
                self._set_voice(voice_id)
            self._set_rate(rate)
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="doteye_tts_")
            os.close(fd)
            stream = self._stream
            stream.Open(path, 3)  # SSFMCreateForWrite
            try:
                self._voice.AudioOutputStream = stream
                self._voice.Speak(text)
            finally:
                self._voice.AudioOutputStream = None
                stream.Close()
            data = Path(path).read_bytes()
            return data if is_wav(data) and len(data) > 44 else None
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            return None
        finally:
            if path:
                Path(path).unlink(missing_ok=True)


class EspeakTTS(TTS):
    def __init__(self) -> None:
        self._bin = shutil.which("espeak-ng") or shutil.which("espeak") or ""

    @property
    def name(self) -> str:
        return "espeak"

    def available(self) -> bool:
        return bool(self._bin)

    def ensure(self) -> None:
        if not self._bin:
            self._bin = shutil.which("espeak-ng") or shutil.which("espeak") or ""

    def list_voices(self) -> list[tuple[str, str]]:
        self.ensure()
        if not self._bin:
            return []
        return [("ru", "espeak ru")]

    def synthesize(
        self, text: str, rate: float = 1.0, voice_id: str = ""
    ) -> bytes | None:
        self.ensure()
        if not self._bin:
            return None
        voice = voice_id or "ru"
        wpm = max(80, min(400, int(175 * max(0.4, min(2.5, rate)))))
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="doteye_espeak_")
        os.close(fd)
        try:
            proc = subprocess.run(
                [self._bin, "-v", voice, "-s", str(wpm), "-w", path, text],
                capture_output=True,
                timeout=30,
                check=False,
            )
            if proc.returncode != 0:
                return None
            data = Path(path).read_bytes()
            return data if is_wav(data) else None
        except (OSError, subprocess.TimeoutExpired):
            return None
        finally:
            Path(path).unlink(missing_ok=True)


class ChainTTS(TTS):
    """Первый доступный бэкенд: pyttsx3, espeak, dummy."""

    def __init__(self, backends: list[TTS]) -> None:
        self._backends = backends
        self._active: TTS | None = None
        self._last_error = ""

    @property
    def name(self) -> str:
        if self._active is not None:
            return self._active.name
        return "tts"

    def available(self) -> bool:
        return self._active is not None and self._active.available()

    @property
    def last_error(self) -> str:
        if self._active is not None and self._active.available():
            return self._active.last_error
        return self._last_error

    def ensure(self) -> None:
        if self._active is not None:
            return
        for backend in self._backends:
            try:
                backend.ensure()
            except Exception as exc:  # noqa: BLE001
                print(f"[tts] {backend.name}: {exc}")
                continue
            if backend.available():
                self._active = backend
                print(f"[tts] backend={backend.name}")
                return
            if backend.last_error:
                self._last_error = backend.last_error
        self._active = DummyTTS()
        print("[tts] нет синтезатора, только сирена")

    def list_voices(self) -> list[tuple[str, str]]:
        self.ensure()
        if self._active is None:
            return []
        return self._active.list_voices()

    def synthesize(
        self, text: str, rate: float = 1.0, voice_id: str = ""
    ) -> bytes | None:
        self.ensure()
        if self._active is None:
            return None
        return self._active.synthesize(text, rate=rate, voice_id=voice_id)


def build_tts() -> TTS:
    return ChainTTS([Sapi5TTS(), Pyttsx3TTS(), EspeakTTS(), DummyTTS()])
