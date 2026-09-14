"""Интерактивный генератор конфигов remote-инференса DotEye.

Предлагает выбор типа развёртывания:
  1. По домену - автоматический HTTPS через Let's Encrypt (Caddy)
  2. По IP-адресу - HTTPS через ngrok или самоподписанный сертификат

Генерирует .env.remote + docker-compose.remote.yml с автоматической настройкой
под выбранный тип. После этого деплой - одна команда:

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


def write_env(domain: str, key: str, model: str, device: str, ip_address: str = "") -> None:
    lines = [
        "# DotEye remote - сгенерировано deploy_remote.py. Не коммить этот файл.",
        "",
    ]
    if domain:
        lines.append(f"DOTEYE_DOMAIN={domain}")
    if ip_address:
        lines.append(f"# IP-адрес сервера: {ip_address}")
    lines.extend([
        f"DOTEYE_CRYPTO_KEY={key}",
        f"DOTEYE_MODEL_PATH={model}",
        f"DOTEYE_DEVICE={device}",
        "",
    ])
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
    print("=" * 70)
    print("DotEye Remote Deploy - генератор конфигурации")
    print("=" * 70)

    key = ask("\nDOTEYE_CRYPTO_KEY (пусто = сгенерировать)").strip()
    if not key:
        key = base64.b64encode(os.urandom(32)).decode()
        print(f"✓ Сгенерирован новый ключ")

    print("\n" + "=" * 70)
    print("Выбери тип развёртывания:")
    print("=" * 70)
    print("\n1. По ДОМЕНУ (рекомендуется)")
    print("   • Автоматический HTTPS через Let's Encrypt")
    print("   • Caddy настроит TLS сертификат сам")
    print("   • Требуется: доменное имя, указывающее на сервер")
    print("\n2. По IP-АДРЕСУ")
    print("   • HTTPS через ngrok или самоподписанный сертификат")
    print("   • Без автоматического TLS")
    print("   • Подходит для временного развёртывания или без домена")
    
    while True:
        mode = ask("\nВыбери тип (1 - домен / 2 - IP)", "1")
        if mode in {"1", "2"}:
            break
        print("Введи 1 или 2.")

    domain = ""
    use_caddy = False
    ip_address = ""
    
    if mode == "1":
        # Развёртывание по домену + Caddy auto-TLS
        print("\n" + "-" * 70)
        print("Настройка развёртывания по домену")
        print("-" * 70)
        domain = ask("Доменное имя (например, eye.example.com)", "eye.example.com")
        use_caddy = True
        print("✓ Caddy автоматически получит Let's Encrypt сертификат")
    else:
        # Развёртывание по IP-адресу
        print("\n" + "-" * 70)
        print("Настройка развёртывания по IP-адресу")
        print("-" * 70)
        ip_address = ask("IP-адрес сервера (для справки, можно оставить пустым)", "")
        use_caddy = False
        print("✓ Remote-сервер будет слушать localhost:8099")
        print("✓ Для HTTPS используй ngrok или самоподписанный сертификат")

    device = ask("\nУстройство инференса (cpu/cuda)", "cpu")
    model = pick_model()

    write_env(domain, key, model, device, ip_address)
    write_compose(use_caddy)

    print("\n" + "=" * 70)
    print("Файлы готовы:")
    print("=" * 70)
    print(f"  ✓ {ENV_PATH}")
    print(f"  ✓ {DEPLOY_DIR / 'docker-compose.remote.yml'}")
    if use_caddy:
        print(f"  ✓ {DEPLOY_DIR / 'Caddyfile'}")

    print("\n" + "=" * 70)
    print("Деплой на сервере:")
    print("=" * 70)
    print("  cd <репозиторий>")
    print("  docker compose -f deploy/docker-compose.remote.yml up -d --build")
    
    if use_caddy:
        print(f"  curl https://{domain}/health   # {'{'}\"status\": \"ok\"{'}'}")
    else:
        print("  curl http://127.0.0.1:8099/health   # {" + '"status": "ok"' + "}")

    if not use_caddy:
        print("\n" + "=" * 70)
        print("Настройка HTTPS (обязательно!):")
        print("=" * 70)
        print("\nВариант 1: ngrok (проще)")
        print("  На сервере запусти: ngrok http 8099")
        print("  ngrok выдаст адрес: https://xxxx-xxx-xxx-xxx.ngrok-free.app")
        print("  Этот адрес используй в DOTEYE_REMOTE_URL на машине с камерой")
        print("\nВариант 2: самоподписанный TLS")
        print("  1. Сгенерируй сертификат:")
        print("     mkdir -p deploy/tls")
        print("     openssl req -x509 -newkey rsa:4096 -nodes \\")
        print("       -keyout deploy/tls/server.key \\")
        print("       -out deploy/tls/server.crt \\")
        print(f"       -days 365 -subj \"/CN={ip_address or '<IP-адрес>'}\"")
        print("  2. Раскомментируй блок volumes и --tls-* в docker-compose.remote.yml")
        print("  3. Измени порты на \"8099:8099\" в docker-compose.remote.yml")
        print(f"  4. В DOTEYE_REMOTE_URL используй: https://{ip_address or '<IP-адрес>'}:8099")

    print("\n" + "=" * 70)
    print("Настройка на машине с камерой (.env):")
    print("=" * 70)
    print("  DOTEYE_REMOTE_PROCESSING=1")
    if use_caddy:
        print(f"  DOTEYE_REMOTE_URL=https://{domain}")
        print(f"  DOTEYE_ALLOWED_URL_HOSTS={domain}")
    else:
        print("  DOTEYE_REMOTE_URL=https://<ngrok-адрес или IP:8099>")
        print("  DOTEYE_ALLOWED_URL_HOSTS=<хост из REMOTE_URL>")
    print(f"  DOTEYE_CRYPTO_KEY={key}")

    print("\n" + "=" * 70)
    print(f"Ключ сохранён в {ENV_PATH}")
    print("Скопируй DOTEYE_CRYPTO_KEY в .env на машине с камерой.")
    print("=" * 70)


if __name__ == "__main__":
    main()
