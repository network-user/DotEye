"""Сервер remote-инференса DotEye.

Поднимает HTTP-сервер, который принимает зашифрованные кадры, детектит
людей выбранной моделью и возвращает боксы. Нужен, когда камера и инференс
разнесены (например, слабый ноутбук с камерой + машина с GPU).

Запуск:
    DOTEYE_CRYPTO_KEY=<тот же ключ, что у клиента> \\
        python -m doteye.remote_server --host 0.0.0.0 --port 8099 \\
        --model yolov8n.pt --device cpu

Клиент (камера) в .env:
    DOTEYE_REMOTE_PROCESSING=1
    DOTEYE_REMOTE_URL=http://<ip-сервера>:8099
    DOTEYE_CRYPTO_KEY=<тот же ключ>

Проверить, что сервер жив: curl http://<ip>:8099/health
"""

from __future__ import annotations

import argparse

from doteye.config import get_settings
from doteye.crypto import Crypto
from doteye.detector import build_detector
from doteye.models import YOLO_MODELS
from doteye.remote import RemoteServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DotEye remote inference server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "yolo", "yunet", "motion"])
    parser.add_argument("--model", default="yolov8n.pt", choices=list(YOLO_MODELS))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--face-model", default="", help="ONNX YuNet для backend=yunet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

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

    server = RemoteServer(args.host, args.port, detector.detect, crypto)
    print(f"[remote] слушаю http://{args.host}:{args.port} (POST /detect, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[remote] остановлено")
    finally:
        server.stop()
        detector.close()


if __name__ == "__main__":
    main()
