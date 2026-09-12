"""Тесты runtime: тихие часы, источники камер."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from doteye.camera import parse_sources
from doteye.config import ConfigurationError, Settings, validate_camera_source, validate_remote_url
from doteye.runtime import Runtime, in_quiet_hours
from doteye.storage import Storage


def test_in_quiet_hours_daytime() -> None:
    now = datetime(2026, 1, 1, 12, 0)
    assert in_quiet_hours("22:00-07:00", now) is False
    assert in_quiet_hours("09:00-18:00", now) is True


def test_in_quiet_hours_overnight() -> None:
    night = datetime(2026, 1, 1, 23, 30)
    morning = datetime(2026, 1, 1, 6, 0)
    noon = datetime(2026, 1, 1, 12, 0)
    assert in_quiet_hours("22:00-07:00", night) is True
    assert in_quiet_hours("22:00-07:00", morning) is True
    assert in_quiet_hours("22:00-07:00", noon) is False


def test_in_quiet_hours_off() -> None:
    now = datetime(2026, 1, 1, 1, 0)
    assert in_quiet_hours("", now) is False
    assert in_quiet_hours("выкл", now) is False
    assert in_quiet_hours("00:00-00:00", now) is True


def test_parse_sources() -> None:
    assert parse_sources("0") == ["0"]
    assert parse_sources("0|rtsp://a/b") == ["0", "rtsp://a/b"]
    assert parse_sources("  ") == ["0"]


def test_runtime_voice_settings(tmp_path: Path) -> None:
    settings = Settings()
    st = Storage(tmp_path / "voice-rt.db")
    rt = Runtime(settings, st)
    assert rt.voice_enabled is True
    assert rt.voice_clear_on == "both"
    rt.voice_enabled = False
    rt.voice_volume = 0.4
    rt.voice_clear_on = "known"
    rt.set_voice_phrase("welcome", "Здравствуй, {name}.")
    assert rt.voice_enabled is False
    assert rt.voice_volume == 0.4
    assert rt.voice_clear_on == "known"
    assert "Здравствуй" in rt.voice_phrase("welcome")
    st.close()


def test_runtime_arm_and_remote(tmp_path: Path) -> None:
    settings = Settings(armed=True, remote_processing=False, remote_url="")
    st = Storage(tmp_path / "r.db")
    rt = Runtime(settings, st)
    assert rt.armed is True
    rt.armed = False
    assert rt.armed is False
    rt.remote_url = "http://127.0.0.1:8099"
    rt.remote_processing = True
    assert rt.remote_processing is True
    assert rt.remote_url.endswith(":8099")
    st.close()


def test_open_access_requires_development() -> None:
    with pytest.raises(ConfigurationError, match="ALLOW_OPEN_ACCESS"):
        Settings(environment="production", allow_open_access=True)
    assert Settings(environment="development", allow_open_access=True).allow_open_access


def test_network_sources_require_exact_allowlist() -> None:
    allowed = frozenset({"camera.local", "inference.local"})
    assert validate_camera_source(
        "0|rtsp://camera.local/stream", allowed_hosts=allowed, environment="production",
    ) == "0|rtsp://camera.local/stream"
    assert validate_remote_url(
        "https://inference.local:8443", allowed_hosts=allowed, environment="production",
    ) == "https://inference.local:8443"
    with pytest.raises(ConfigurationError, match="DOTEYE_ALLOWED_URL_HOSTS"):
        validate_camera_source(
            "rtsp://other.local/stream", allowed_hosts=allowed, environment="production",
        )
    with pytest.raises(ConfigurationError, match="credentials"):
        validate_camera_source(
            "rtsp://user:pass@camera.local/stream", allowed_hosts=allowed, environment="production",
        )


def test_runtime_rejects_untrusted_persisted_url(tmp_path: Path) -> None:
    settings = Settings(environment="production", camera_source="0")
    st = Storage(tmp_path / "runtime.db")
    rt = Runtime(settings, st)
    st.set("camera_source", "rtsp://untrusted.local/stream")
    assert rt.camera_source == "0"
    with pytest.raises(ConfigurationError, match="DOTEYE_ALLOWED_URL_HOSTS"):
        rt.camera_source = "rtsp://untrusted.local/stream"
    st.close()


def test_runtime_rejects_unbounded_numeric_overrides(tmp_path: Path) -> None:
    settings = Settings(imgsz=640, events_max=500, events_ttl_days=14)
    st = Storage(tmp_path / "numeric.db")
    rt = Runtime(settings, st)
    st.set("imgsz", "999999")
    st.set("events_max", "-1")
    st.set("events_ttl_days", "inf")
    assert rt.imgsz == 640
    assert rt.events_max == 500
    assert rt.events_ttl_days == 14
    st.close()
