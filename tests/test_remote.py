"""Тесты remote-инференса: кодек кадров и клиент-сервер через HTTP."""

from __future__ import annotations

import http.client
import numpy as np
import pytest

from doteye.crypto import Crypto, generate_key_b64
from doteye.detector import RemoteDetector
from doteye.remote import RemoteClient, RemoteServer, decode_frame, encode_frame


@pytest.fixture()
def crypto() -> Crypto:
    return Crypto(generate_key_b64())


def test_encode_decode_roundtrip(crypto: Crypto) -> None:
    frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    payload = encode_frame(frame, crypto)
    restored = decode_frame(payload, crypto)
    assert restored.shape == frame.shape


def test_decode_with_wrong_key_fails(crypto: Crypto) -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    payload = encode_frame(frame, crypto)
    other = Crypto(generate_key_b64())
    with pytest.raises(Exception):
        decode_frame(payload, other)


def test_client_server_roundtrip(crypto: Crypto) -> None:
    calls: list[int] = []

    def detect(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        calls.append(frame.shape[0])
        return [(1, 2, 3, 4), (5, 6, 7, 8)]

    server = RemoteServer("127.0.0.1", 0, detect, crypto)
    port = server._server.server_address[1]
    server.start()
    try:
        client = RemoteClient(f"http://127.0.0.1:{port}", crypto, timeout=5.0)
        assert client.health() is True
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        boxes = client.detect(frame)
        assert boxes == [(1, 2, 3, 4), (5, 6, 7, 8)]
        assert calls == [32]
    finally:
        server.stop()


def test_remote_detector_returns_empty_without_crypto() -> None:
    detector = RemoteDetector("http://127.0.0.1:1", crypto=None)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


def test_remote_detector_survives_bad_server(crypto: Crypto) -> None:
    detector = RemoteDetector("http://127.0.0.1:1", crypto=crypto, timeout=0.2)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert detector.detect(frame) == []
    assert detector.last_error


def test_remote_fallback_used_when_server_down(crypto: Crypto) -> None:
    class Local:
        backend = "yolo"

        def detect(self, frame):
            return [(2, 2, 4, 4)]

        def close(self):
            pass

    detector = RemoteDetector(
        "http://127.0.0.1:1", crypto=crypto, timeout=0.2, fallback=Local(),
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert detector.detect(frame) == [(2, 2, 4, 4)]
    assert detector.using_fallback is True
    detector.close()


def test_detect_rejects_bad_token(crypto: Crypto) -> None:
    from doteye.crypto import generate_key_b64
    from doteye.remote import RemoteClient, RemoteServer

    def detect(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        return [(1, 1, 2, 2)]

    server = RemoteServer("127.0.0.1", 0, detect, crypto)
    port = server._server.server_address[1]
    server.start()
    try:
        other = Crypto(generate_key_b64())
        client = RemoteClient(f"http://127.0.0.1:{port}", other, timeout=2.0)
        with pytest.raises(RuntimeError, match="токен|401|HTTP"):
            client.detect(np.zeros((8, 8, 3), dtype=np.uint8))
    finally:
        server.stop()


def test_client_rejects_unencrypted_remote_url(crypto: Crypto) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        RemoteClient("http://192.0.2.1:8099", crypto)


def test_client_rejects_invalid_boxes(crypto: Crypto) -> None:
    server = RemoteServer("127.0.0.1", 0, lambda _frame: [(4, 4, 2, 2)], crypto)
    port = server._server.server_address[1]
    server.start()
    try:
        client = RemoteClient(f"http://127.0.0.1:{port}", crypto)
        with pytest.raises(RuntimeError, match="HTTP 500"):
            client.detect(np.zeros((8, 8, 3), dtype=np.uint8))
    finally:
        server.stop()


def test_client_validates_remote_response(crypto: Crypto) -> None:
    client = RemoteClient("http://127.0.0.1:8099", crypto)
    client._request = lambda *_args: (200, b'{"boxes": [[1, 2, 1, 4]]}')  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="некорректный"):
        client.detect(np.zeros((8, 8, 3), dtype=np.uint8))


def test_server_rejects_oversized_body_before_reading(crypto: Crypto) -> None:
    server = RemoteServer("127.0.0.1", 0, lambda _frame: [], crypto)
    port = server._server.server_address[1]
    server.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
        import secrets
        import time
        from doteye.remote import (
            NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, WORKER_ID_HEADER,
        )
        ts = str(int(time.time()))
        nonce = secrets.token_hex(16)
        sig = crypto.sign_request("POST", "/detect", ts, nonce, b"")
        conn.request(
            "POST",
            "/detect",
            body=b"",
            headers={
                "Content-Length": str(2 * 1024 * 1024 + 1),
                WORKER_ID_HEADER: "w_test",
                TIMESTAMP_HEADER: ts,
                NONCE_HEADER: nonce,
                SIGNATURE_HEADER: sig,
            },
        )
        assert conn.getresponse().status == 400
    finally:
        server.stop()
