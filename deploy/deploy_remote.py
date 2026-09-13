"""Интерактивный генератор конфигов remote-инференса DotEye.

Спрашивает, как remote-сервер доступен из интернета (свой домен / IP / ngrok),
и пишет .env + docker-compose.remote.yml. После этого деплой - одна команда:

    docker compose -f deploy/docker-compose.remote.yml up -d --build

Запуск (в корне репозитория):
    python deploy/deploy_remote.py
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf8"}:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy"
ENV_PATH = DEPLOY_DIR / ".env.remote"
MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]


def ask(text: str, default: str = "") -> str:
    prompt = f"{text} [{default}]: " if default else f"{text}: "
    value = input(prompt).strip()
    return value or default


def ask_bool(text: str) -> bool:
    value = input(f"{text} [y/N]: ").strip().lower()
    return value in {"y", "yes", "да", "д"}


def pick_model() -> str:
    print("\nМодель (меньше = быстрее, больше = точнее):")
    for i, name in enumerate(MODELS, 1):
        print(f"  {i}. {name}")
    while True:
        raw = ask("Выбери номер", "1")
        try:
            idx = int(raw)
        except ValueError:
            print("Введи номер из списка.")
            continue
        if 1 <= idx <= len(MODELS):
            return MODELS[idx - 1]
        print("Номер вне диапазона.")


def write_env(domain: str, key: str, model: str, device: str) -> None:
    lines = [
        "# DotEye remote - сгенерировано deploy_remote.py. Не коммить этот файл.",
        "",
        f"DOTEYE_DOMAIN={domain}",
        f"DOTEYE_CRYPTO_KEY={key}",
        f"DOTEYE_MODEL_PATH={model}",
        f"DOTEYE_DEVICE={device}",
        "",
    ]
    ENV_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_compose(caddy: bool) -> None:
    """Собственный compose рядом со скриптом (тот же каталог deploy/)."""
    # Только одна схема: environment + env_file не комбинируются у `remote`,
    # поэтому ключ/модель/устройство всегда тянутся из .env.remote.
    ports = '    expose:\n      - "8099"\n' if caddy else '    ports:\n      - "8099:8099"\n'

    remote = (
        "services:\n"
        "  remote:\n"
        "    build:\n"
        "      context: ..\n"
        "    image: doteye-remote:latest\n"
        "    restart: unless-stopped\n"
        + ports
        + "    env_file:\n      - .env.remote\n"
        + "    environment:\n"
        + "      DOTEYE_ENV: production\n"
        + "    command:\n"
        + "      [\n"
        + '        "python", "-m", "doteye.remote_server",\n'
        + '        "--host", "0.0.0.0", "--port", "8099",\n'
        + '        "--model", "${DOTEYE_MODEL_PATH:-yolov8n.pt}",\n'
        + '        "--device", "${DOTEYE_DEVICE:-cpu}",\n'
        + "      ]\n"
        + "    read_only: true\n"
        + "    tmpfs:\n"
        + "      - /tmp\n"
        + "    security_opt:\n"
        + "      - no-new-privileges:true\n"
    )

    caddy_svc = ""
    if caddy:
        caddy_svc = (
            "\n"
            "  caddy:\n"
            "    image: caddy:2-alpine\n"
            "    restart: unless-stopped\n"
            "    depends_on:\n"
            "      - remote\n"
            "    environment:\n"
            '      DOMAIN: ${DOTEYE_DOMAIN}\n'
            '    env_file:\n      - .env.remote\n'
            '    ports:\n      - "80:80"\n      - "443:443"\n'
            '    volumes:\n'
            '      - ./Caddyfile:/etc/caddy/Caddyfile:ro\n'
            '      - caddy_data:/data\n'
            '      - caddy_config:/config\n'
            "\n"
            "volumes:\n"
            "  caddy_data:\n"
            "  caddy_config:\n"
        )

    compose_path = DEPLOY_DIR / "docker-compose.remote.yml"
    compose_path.write_text(remote + caddy_svc, encoding="utf-8")


def main() -> None:
    print("DotEye remote deploy - генератор конфигурации\n")

    key = ask("DOTEYE_CRYPTO_KEY (пусто = сгенерировать)").strip()
    if not key:
        key = base64.b64encode(os.urandom(32)).decode()

    print("\nКак remote-сервер виден из интернета?")
    print("  1. Свой домен (Caddy поставит Let's Encrypt - рекомендую)")
    print("  2. Только IP (ngrok даст https-адрес, либо подними TLS сам)")
    print("  3. ngrok (домен сейчас не нужен)")
    while True:
        mode = ask("Режим", "1")
        if mode in {"1", "2", "3"}:
            break
        print("Введи 1, 2 или 3.")

    domain = ""
    caddy = False
    if mode == "1":
        # домен + Caddy auto-TLS
        domain = ask("Доменное имя", "eye.example.com")
        caddy = True
    elif mode == "2":
        domain = ask("Домен (можно оставить пустым)")
        caddy = False
    else:
        # ngrok
        caddy = False
        domain = ""

    device = ask("Устройство инференса (cpu/cuda)", "cpu")
    model = pick_model()

    write_env(domain, key, model, device)
    write_compose(caddy)

    print("\nФайлы готовы:")
    print(f"  {ENV_PATH}")
    print(f"  {DEPLOY_DIR / 'docker-compose.remote.yml'}")
    if caddy:
        print(f"  {DEPLOY_DIR / 'Caddyfile'}")
    print("\nДальше на сервере:")
    print("  cd <репозиторий>")
    print("  docker compose -f deploy/docker-compose.remote.yml up -d --build")
    print("  curl https://<домен>/health   # {" + '"status": "ok"' + "}")

    if not caddy:
        print("\nЛокально на машине с камерой (.env):")
        print("  DOTEYE_REMOTE_PROCESSING=1")
        print("  DOTEYE_REMOTE_URL=https://<внешний-адрес>")
        print("  DOTEYE_CRYPTO_KEY=<тот же ключ>")
        print("  DOTEYE_ALLOWED_URL_HOSTS=<внешний-хост>")

    print(
        f"\nКлюч записан в {ENV_PATH}. Скопируй его же в .env на машине с камерой."
    )


if __name__ == "__main__":
    main()
