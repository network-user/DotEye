"""Тесты криптослоя: roundtrip, ключ, детект подмены."""

from __future__ import annotations

import base64
import os

import pytest

from doteye.crypto import Crypto, generate_key_b64


def test_roundtrip() -> None:
    crypto = Crypto(generate_key_b64())
    data = b"secret payload"
    assert crypto.decrypt(crypto.encrypt(data)) == data


def test_nonce_is_random() -> None:
    crypto = Crypto(generate_key_b64())
    data = b"same"
    assert crypto.encrypt(data) != crypto.encrypt(data)


def test_missing_key() -> None:
    with pytest.raises(ValueError):
        Crypto("")


def test_wrong_length_key() -> None:
    with pytest.raises(ValueError):
        Crypto(base64.b64encode(os.urandom(16)).decode())


def test_auth_token_stable_and_not_the_key() -> None:
    key = generate_key_b64()
    crypto = Crypto(key)
    token = crypto.auth_token()
    assert len(token) == 64
    assert crypto.auth_token() == token
    assert token != key


def test_tampered_payload_fails() -> None:
    crypto = Crypto(generate_key_b64())
    payload = bytearray(crypto.encrypt(b"data"))
    payload[-1] ^= 0x01
    with pytest.raises(Exception):
        crypto.decrypt(bytes(payload))
