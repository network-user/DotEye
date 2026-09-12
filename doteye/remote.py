"""Remote-инференс: HTTP-транспорт поверх AES-GCM.

Схема: клиент (ноутбук с камерой) шлёт зашифрованный кадр, сервер
расшифровывает, прогоняет детектор и возвращает боксы. Кадры никогда не
идут по сети в открытом виде.

Протокол (JSON):
    POST /detect
    Header: X-DotEye-Token: <sha256(key||doteye-remote)>
    {"frame": "<base64(AES-GCM JPEG)>"}
    -> {"boxes": [[x1, y1, x2, y2], ...]}

Keep-alive через http.client. HTTPS берётся из URL.
Сервер на stdlib http.server - без лишних зависимостей.
"""

from __future__ import annotations

import base64
import http.client
import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import urlparse

import cv2
import numpy as np

from doteye.crypto import Crypto

TOKEN_HEADER = "X-DotEye-Token"


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
    """Клиентская часть: шлёт кадр, получает боксы. Соединение переиспользуется."""

    def __init__(
        self,
        url: str,
        crypto: Crypto,
        timeout: float = 10.0,
        insecure: bool = False,
    ) -> None:
        self._url = url.rstrip("/")
        self._crypto = crypto
        self._timeout = timeout
        self._insecure = insecure
        parsed = urlparse(self._url if "://" in self._url else "http://" + self._url)
        self._scheme = parsed.scheme or "http"
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or (443 if self._scheme == "https" else 80)
        self._conn: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()

    def _connect(self) -> http.client.HTTPConnection:
        if self._scheme == "https":
            context = ssl._create_unverified_context() if self._insecure else ssl.create_default_context()
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                self._host, self._port, timeout=self._timeout, context=context,
            )
        else:
            conn = http.client.HTTPConnection(
                self._host, self._port, timeout=self._timeout,
            )
        return conn

    def _request(self, method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
        headers = {"X-DotEye-Token": self._crypto.auth_token()}
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        with self._lock:
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    if self._conn is None:
                        self._conn = self._connect()
                    self._conn.request(method, path, body=body, headers=headers)
                    resp = self._conn.getresponse()
                    data = resp.read()
                    return resp.status, data
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    self.close()
                    if attempt == 1:
                        raise
            raise last_exc or RuntimeError("remote request failed")

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        payload = json.dumps({"frame": encode_frame(frame, self._crypto)}).encode()
        status, data = self._request("POST", "/detect", payload)
        if status == 401:
            raise RuntimeError("remote: отказ в токене")
        if status >= 400:
            raise RuntimeError(f"remote HTTP {status}: {data[:200]!r}")
        parsed = json.loads(data or b"{}")
        return [tuple(int(v) for v in box) for box in parsed.get("boxes", [])]

    def health(self) -> bool:
        try:
            status, _ = self._request("GET", "/health")
            return status == 200
        except (OSError, http.client.HTTPException, json.JSONDecodeError):
            return False

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def make_handler(
    detect: Callable[[np.ndarray], list[tuple[int, int, int, int]]],
    crypto: Crypto,
) -> type[BaseHTTPRequestHandler]:
    """Собрать класс-обработчик с зашитыми зависимостями.

    Замыкание, а не атрибут класса: иначе функция из атрибута превращается
    в bound method и получает лишний self.
    """
    expected = crypto.auth_token()

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

        def _token_ok(self) -> bool:
            got = self.headers.get(TOKEN_HEADER) or self.headers.get("x-doteye-token")
            return got == expected

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/detect":
                self._send_json(404, {"error": "not found"})
                return
            if not self._token_ok():
                self._send_json(401, {"error": "unauthorized"})
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
        max_workers: int = 2,
    ) -> None:
        workers = max(1, min(16, int(max_workers)))
        limiter = threading.BoundedSemaphore(workers)

        def limited(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
            with limiter:
                return detect_handler(frame)

        self._server = ThreadingHTTPServer(
            (host, port), make_handler(limited, crypto)
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
