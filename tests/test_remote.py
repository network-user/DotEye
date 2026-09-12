"""Тесты remote-инференса: кодек кадров и клиент-сервер через HTTP."""

from __future__ import annotations

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
