"""Remote-инференс: HTTP-транспорт поверх AES-256-GCM.

Схема: клиент (ноутбук с камерой) шлёт зашифрованный кадр, сервер
расшифровывает, прогоняет детектор и возвращает боксы. Кадры никогда не
идут по сети в открытом виде.

Протокол (JSON):
    POST /detect
    {"frame": "<base64(AES-GCM JPEG)>"}
    -> {"boxes": [[x1, y1, x2, y2], ...]}

Сервер на stdlib http.server - без лишних зависимостей. Для продакшена
можно поставить за nginx/gunicorn, но для домашнего контура достаточно.
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import cv2
import numpy as np

from doteye.crypto import Crypto


def encode_frame(frame: np.ndarray, crypto: Crypto, quality: int = 85) -> str:
    """BGR-кадр -> base64 строка с зашифрованным JPEG."""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("не удалось закодировать кадр")
    return base64.b64encode(crypto.encrypt(buf.tobytes())).decode()


def decode_frame(payload_b64: str, crypto: Crypto) -> np.ndarray:
    """base64 с зашифрованным JPEG -> BGR-кадр."""
    encrypted = base64.b64decode(payload_b64)
    jpeg = crypto.decrypt(encrypted)
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("не удалось декодировать кадр")
    return frame


class RemoteClient:
    """Клиентская часть: шлёт кадр, получает боксы."""

    def __init__(self, url: str, crypto: Crypto, timeout: float = 10.0) -> None:
        self._url = url.rstrip("/")
        self._crypto = crypto
        self._timeout = timeout

    def _post_detect(self, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self._url}/detect",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        data = self._post_detect({"frame": encode_frame(frame, self._crypto)})
        return [tuple(int(v) for v in box) for box in data.get("boxes", [])]

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self._url}/health", timeout=self._timeout
            ) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False


def make_handler(
    detect: Callable[[np.ndarray], list[tuple[int, int, int, int]]],
    crypto: Crypto,
) -> type[BaseHTTPRequestHandler]:
    """Собрать класс-обработчик с зашитыми зависимостями.

    Замыкание, а не атрибут класса: иначе функция из атрибута превращается
    в bound method и получает лишний self.
    """

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            print(f"[remote] {self.address_string()} {fmt % args}")

        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/detect":
                self._send_json(404, {"error": "not found"})
                return

            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
                frame = decode_frame(data["frame"], crypto)
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"error": f"bad request: {exc}"})
                return

            try:
                boxes = detect(frame)
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": f"detect failed: {exc}"})
                return

            self._send_json(200, {"boxes": [list(b) for b in boxes]})

    return _Handler


class RemoteServer:
    """Обёртка над ThreadingHTTPServer: поднимает сервер в потоке."""

    def __init__(
        self,
        host: str,
        port: int,
        detect_handler: Callable[[np.ndarray], list[tuple[int, int, int, int]]],
        crypto: Crypto,
    ) -> None:
        self._server = ThreadingHTTPServer(
            (host, port), make_handler(detect_handler, crypto)
        )
        self._thread: threading.Thread | None = None

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def start(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
