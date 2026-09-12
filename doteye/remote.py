"""Защищённый HTTP-транспорт remote-инференса поверх AES-GCM."""

from __future__ import annotations

import base64
import hmac
import http.client
import ipaddress
import json
import math
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

import cv2
import numpy as np

from doteye.crypto import Crypto

TOKEN_HEADER = "X-DotEye-Token"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_ENCRYPTED_FRAME_BYTES = 1500 * 1024
MAX_JPEG_BYTES = 1200 * 1024
MAX_FRAME_PIXELS = 8_000_000
MAX_RESPONSE_BYTES = 128 * 1024
MAX_BOXES = 256
SOCKET_TIMEOUT_SECONDS = 15.0


def encode_frame(frame: np.ndarray, crypto: Crypto, quality: int = 85) -> str:
    """BGR-кадр -> base64 строка с зашифрованным JPEG."""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("не удалось закодировать кадр")
    encrypted = crypto.encrypt(buf.tobytes())
    if len(encrypted) > MAX_ENCRYPTED_FRAME_BYTES:
        raise ValueError("зашифрованный кадр слишком большой")
    return base64.b64encode(encrypted).decode()


def decode_frame(payload_b64: str, crypto: Crypto) -> np.ndarray:
    """base64 зашифрованного JPEG -> BGR-кадр."""
    if not isinstance(payload_b64, str) or len(payload_b64) > MAX_REQUEST_BYTES:
        raise ValueError("некорректный размер кадра")
    try:
        encrypted = base64.b64decode(payload_b64, validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError("некорректный base64 кадра") from exc
    if len(encrypted) > MAX_ENCRYPTED_FRAME_BYTES:
        raise ValueError("зашифрованный кадр слишком большой")
    jpeg = crypto.decrypt(encrypted)
    if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
        raise ValueError("JPEG кадра слишком большой")
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("не удалось декодировать кадр")
    if frame.shape[0] * frame.shape[1] > MAX_FRAME_PIXELS:
        raise ValueError("разрешение кадра слишком большое")
    return frame


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_boxes(payload: Any) -> list[tuple[int, int, int, int]]:
    if not isinstance(payload, dict) or set(payload) != {"boxes"}:
        raise ValueError("remote вернул некорректный JSON")
    boxes = payload["boxes"]
    if not isinstance(boxes, list) or len(boxes) > MAX_BOXES:
        raise ValueError("remote вернул некорректные боксы")
    validated: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("remote вернул некорректный бокс")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
            raise ValueError("remote вернул некорректный бокс")
        if not all(math.isfinite(float(value)) for value in box):
            raise ValueError("remote вернул некорректный бокс")
        x1, y1, x2, y2 = (int(value) for value in box)
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("remote вернул некорректный бокс")
        validated.append((x1, y1, x2, y2))
    return validated


class RemoteClient:
    """Клиент remote-инференса с keep-alive и ограничениями ответа."""

    def __init__(
        self, url: str, crypto: Crypto, timeout: float = 10.0, insecure: bool = False,
    ) -> None:
        self._url = url.rstrip("/")
        self._crypto = crypto
        self._timeout = timeout
        self._insecure = insecure
        parsed = urlparse(self._url if "://" in self._url else "http://" + self._url)
        self._scheme = parsed.scheme.lower() or "http"
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or (443 if self._scheme == "https" else 80)
        if self._scheme not in {"http", "https"} or parsed.username or parsed.password:
            raise ValueError("remote URL должен быть HTTP(S) URL без учётных данных")
        # HTTP оставлен только для локального development и Docker healthcheck.
        if self._scheme == "http" and not _is_loopback_host(self._host):
            raise ValueError("remote HTTP разрешён только для localhost; используй HTTPS или VPN")
        self._conn: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()

    def _connect(self) -> http.client.HTTPConnection:
        if self._scheme == "https":
            context = ssl._create_unverified_context() if self._insecure else ssl.create_default_context()
            return http.client.HTTPSConnection(self._host, self._port, timeout=self._timeout, context=context)
        return http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)

    def _request(self, method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
        headers = {TOKEN_HEADER: self._crypto.auth_token(), "Connection": "close"}
        if body is not None:
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(body))})
        with self._lock:
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    if self._conn is None:
                        self._conn = self._connect()
                    self._conn.request(method, path, body=body, headers=headers)
                    response = self._conn.getresponse()
                    content_length = response.getheader("Content-Length")
                    if content_length is not None and int(content_length) > MAX_RESPONSE_BYTES:
                        raise ValueError("remote ответ слишком большой")
                    data = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(data) > MAX_RESPONSE_BYTES:
                        raise ValueError("remote ответ слишком большой")
                    return response.status, data
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    self.close()
                    if attempt == 1:
                        raise
            raise last_exc or RuntimeError("remote request failed")

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        payload = json.dumps({"frame": encode_frame(frame, self._crypto)}).encode()
        if len(payload) > MAX_REQUEST_BYTES:
            raise ValueError("remote запрос слишком большой")
        status, data = self._request("POST", "/detect", payload)
        if status == 401:
            raise RuntimeError("remote: отказ в токене")
        if status >= 400:
            raise RuntimeError(f"remote HTTP {status}: {data[:200]!r}")
        try:
            return _validate_boxes(json.loads(data))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("remote вернул некорректный ответ") from exc

    def health(self) -> bool:
        try:
            status, _ = self._request("GET", "/health")
            return status == 200
        except (OSError, ValueError, http.client.HTTPException, json.JSONDecodeError):
            return False

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None


def make_handler(
    detect: Callable[[np.ndarray], list[tuple[int, int, int, int]]], crypto: Crypto,
) -> type[BaseHTTPRequestHandler]:
    """Собрать HTTP handler с закрытыми зависимостями."""
    expected = crypto.auth_token()

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(SOCKET_TIMEOUT_SECONDS)

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            print(f"[remote] {self.address_string()} {fmt % args}")

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass

        def _token_ok(self) -> bool:
            got = self.headers.get(TOKEN_HEADER) or self.headers.get(TOKEN_HEADER.lower())
            return isinstance(got, str) and hmac.compare_digest(got, expected)

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
            try:
                content_length = self.headers.get("Content-Length")
                if content_length is None:
                    raise ValueError("Content-Length обязателен")
                length = int(content_length)
                if length < 1 or length > MAX_REQUEST_BYTES:
                    raise ValueError("некорректный размер запроса")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("неполное тело запроса")
                data = json.loads(raw)
                if not isinstance(data, dict) or set(data) != {"frame"}:
                    raise ValueError("некорректный JSON запроса")
                frame = decode_frame(data["frame"], crypto)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self._send_json(400, {"error": "bad request"})
                return
            try:
                boxes = detect(frame)
                safe_boxes = _validate_boxes({"boxes": [list(box) for box in boxes]})
            except Exception:  # noqa: BLE001
                self._send_json(500, {"error": "detect failed"})
                return
            self._send_json(200, {"boxes": [list(box) for box in safe_boxes]})

    return _Handler


class _BoundedHTTPServer(ThreadingHTTPServer):
    """Ограничивает число одновременных соединений до запуска потока."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(self, *args: Any, max_connections: int, **kwargs: Any) -> None:
        self._connection_limiter = threading.BoundedSemaphore(max_connections)
        super().__init__(*args, **kwargs)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._connection_limiter.acquire(blocking=False):
            self.shutdown_request(request)
            return

        def run() -> None:
            try:
                self.process_request_thread(request, client_address)
            finally:
                self._connection_limiter.release()

        threading.Thread(target=run, daemon=True).start()


class RemoteServer:
    """Обёртка над HTTP-сервером remote-инференса."""

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

        self._server = _BoundedHTTPServer(
            (host, port), make_handler(limited, crypto), max_connections=workers * 2,
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
