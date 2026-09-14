"""Проверки генератора remote-конфигурации."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_generator():
    path = Path(__file__).resolve().parents[1] / "deploy" / "deploy_remote.py"
    spec = importlib.util.spec_from_file_location("deploy_remote_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ip_compose_enables_tls_without_manual_edits(tmp_path: Path, monkeypatch) -> None:
    generator = _load_generator()
    monkeypatch.setattr(generator, "DEPLOY_DIR", tmp_path)
    generator.write_compose(caddy=False, direct_ip=True, model="yolov8s.pt", device="cuda")
    compose = (tmp_path / "docker-compose.remote.yml").read_text(encoding="utf-8")
    assert '"8099:8099"' in compose
    assert '"--tls-cert", "/run/tls/server.crt"' in compose
    assert '"--tls-key", "/run/tls/server.key"' in compose
    assert "./tls:/run/tls:ro" in compose
    assert "./models:/models" in compose
    assert "working_dir: /models" in compose
    assert "PYTHONPATH: /app" in compose
    assert '"--model", "yolov8s.pt", "--device", "cuda"' in compose
