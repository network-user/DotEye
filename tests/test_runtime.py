"""Тесты runtime: тихие часы, источники камер."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from doteye.camera import parse_sources
from doteye.config import Settings
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
