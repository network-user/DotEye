"""Сервер remote-инференса DotEye.

Поднимает HTTP-сервер, который принимает зашифрованные кадры, детектит
людей выбранной моделью и возвращает боксы. Нужен, когда камера и инференс
разнесены (например, слабый ноутбук с камерой + машина с GPU).

Запуск:
    DOTEYE_CRYPTO_KEY=<тот же ключ, что у клиента> \\
        python -m doteye.remote_server --host 0.0.0.0 --port 8099 \\
        --tls-cert server.crt --tls-key server.key \\
        --model yolov8n.pt --device cpu

Клиент (камера) в .env:
    DOTEYE_REMOTE_PROCESSING=1
    DOTEYE_REMOTE_URL=https://<ip-сервера>:8099
    DOTEYE_CRYPTO_KEY=<тот же ключ>

Внешний хост требует TLS; для локальной отладки HTTP оставь loopback:
    python -m doteye.remote_server --host 127.0.0.1 --port 8099

Проверить, что сервер жив: curl http://127.0.0.1:8099/health
"""

from __future__ import annotations

import argparse
import ssl

from doteye.config import get_settings
from doteye.crypto import Crypto
from doteye.detector import build_detector
from doteye.models import YOLO_MODELS
from doteye.remote import RemoteServer, build_server_ssl_context, is_loopback_host


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DotEye remote inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "yolo", "yunet", "motion"])
    parser.add_argument("--model", default="yolov8n.pt", choices=list(YOLO_MODELS))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--face-model", default="", help="ONNX YuNet для backend=yunet")
    parser.add_argument("--workers", type=int, default=2, help="одновременные запросы инференса")
    parser.add_argument("--tls-cert", default="", help="PEM-сертификат для HTTPS")
    parser.add_argument("--tls-key", default="", help="PEM-ключ для HTTPS")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers должен быть в диапазоне 1..16")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf должен быть в диапазоне 0..1")
    if args.tls_cert and not args.tls_key or args.tls_key and not args.tls_cert:
        raise SystemExit("--tls-cert и --tls-key задаются вместе")
    settings = get_settings()

    try:
        ssl_context = (
            build_server_ssl_context(args.tls_cert, args.tls_key)
            if args.tls_cert else None
        )
    except (OSError, ssl.SSLError) as exc:
        raise SystemExit(f"не удалось загрузить TLS-сертификат: {exc}") from exc
    if ssl_context is None and not (is_loopback_host(args.host) or settings.environment == "development"):
        raise SystemExit(
            "plain HTTP разрешён только для loopback; для внешнего хоста "
            "укажи --tls-cert/--tls-key (или DOTEYE_ENV=development)"
        )

    try:
        crypto = Crypto(settings.crypto_key_env)
    except ValueError as exc:
        print(f"[remote] {exc}. Сгенерируй ключ: python -m doteye.crypto")
        return

    detector = build_detector(
        args.backend, args.model, args.device, args.conf, False, "", args.face_model
    )
    print(
        f"[remote] детектор: {detector.backend}, модель {args.model}, "
        f"device {args.device}"
    )

    scheme = "https" if ssl_context is not None else "http"
    server = RemoteServer(
        args.host, args.port, detector.detect, crypto,
        max_workers=args.workers, ssl_context=ssl_context,
    )
    print(f"[remote] слушаю {scheme}://{args.host}:{args.port} (POST /detect, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[remote] остановлено")
    finally:
        server.stop()
        detector.close()


if __name__ == "__main__":
    main()
