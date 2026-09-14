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


def test_settings_camera_names_are_padded_and_validated() -> None:
    from doteye.config import ConfigurationError as CfgError
    from doteye.config import Settings

    settings = Settings(camera_source="0|1", camera_names=("Вход",))
    assert settings.camera_names == ("Вход", "")
    with pytest.raises(CfgError, match="DOTEYE_CAMERA_NAMES"):
        Settings(camera_source="0", camera_names=("a", "b"))


def test_runtime_voice_settings(tmp_path: Path) -> None:
    settings = Settings()
    st = Storage(tmp_path / "voice-rt.db")
    rt = Runtime(settings, st)
    assert rt.voice_enabled is True
    assert rt.voice_presence is True
    assert rt.voice_clear_on == "known"
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


def test_remote_ca_certificate_must_be_a_file(tmp_path: Path) -> None:
    certificate = tmp_path / "remote.crt"
    certificate.write_text("PEM", encoding="utf-8")
    assert Settings(remote_ca_cert=str(certificate)).remote_ca_cert == str(certificate)
    with pytest.raises(ConfigurationError, match="REMOTE_CA_CERT"):
        Settings(remote_ca_cert=str(tmp_path / "missing.crt"))


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


def test_runtime_privacy_modes_keep_legacy_default(tmp_path: Path) -> None:
    st = Storage(tmp_path / "privacy.db")
    rt = Runtime(Settings(privacy_outbound=True), st)
    assert rt.privacy_mode == "silhouette"
    rt.privacy_mode = "face"
    rt.privacy_blocks = 6
    assert rt.privacy_mode == "face"
    assert rt.privacy_blocks == 6
    rt.privacy_mode = "off"
    assert rt.privacy_outbound is False
    st.close()


def test_runtime_camera_names_default_and_override(tmp_path: Path) -> None:
    settings = Settings(camera_source="0|1", camera_names=("Вход", ""))
    st = Storage(tmp_path / "names.db")
    rt = Runtime(settings, st)
    assert rt.camera_sources() == ["0", "1"]
    assert rt.camera_names == ["Вход", ""]
    assert rt.camera_label(0) == "Вход"
    assert rt.camera_label(1) == "1"
    rt.set_camera_name(1, "Склад")
    assert rt.camera_names == ["Вход", "Склад"]
    assert rt.camera_label(1) == "Склад"
    rt.set_camera_name(1, "")
    assert rt.camera_label(1) == "1"
    st.close()


def test_runtime_camera_rejects_long_name(tmp_path: Path) -> None:
    settings = Settings(camera_source="0")
    st = Storage(tmp_path / "long.db")
    rt = Runtime(settings, st)
    with pytest.raises(ConfigurationError, match="имя камеры"):
        rt.set_camera_name(0, "x" * 41)
    st.close()


def test_runtime_camera_overrides(tmp_path: Path) -> None:
    settings = Settings(camera_source="0|1", camera_parallel=False)
    st = Storage(tmp_path / "ov.db")
    rt = Runtime(settings, st)
    assert rt.camera_parallel is False
    assert rt.camera_cooldown(0) == settings.cooldown_seconds
    assert rt.camera_enabled(0) is True
    assert rt.camera_notify_exit(0) is settings.notify_exit

    rt.set_camera_override(0, "cooldown_seconds", 5.0)
    rt.set_camera_override(0, "enabled", False)
    rt.set_camera_override(0, "notify_exit", True)
    rt.set_camera_override(0, "detect_mode", "identity")
    assert rt.camera_cooldown(0) == 5.0
    assert rt.camera_enabled(0) is False
    assert rt.camera_notify_exit(0) is True
    assert rt.camera_detect_mode(0) == "identity"
    assert rt.camera_cooldown(1) == settings.cooldown_seconds

    rt.clear_camera_override(0, "cooldown_seconds")
    assert rt.camera_cooldown(0) == settings.cooldown_seconds
    with pytest.raises(ConfigurationError, match="неизвестная настройка"):
        rt.set_camera_override(0, "bogus", 1)
    st.close()


def test_runtime_camera_override_ignores_corrupt_json(tmp_path: Path) -> None:
    settings = Settings(camera_source="0")
    st = Storage(tmp_path / "corrupt.db")
    rt = Runtime(settings, st)
    st.set("camera_overrides", "{not json")
    assert rt.camera_override(0) == {}
    assert rt.camera_cooldown(0) == settings.cooldown_seconds
    st.close()
