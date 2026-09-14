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


def test_placeholder_key_rejected() -> None:
    zeros = base64.b64encode(b"\x00" * 32).decode()
    with pytest.raises(ValueError, match="слабый"):
        Crypto(zeros)


def test_sign_and_verify_roundtrip() -> None:
    crypto = Crypto(generate_key_b64())
    sig = crypto.sign_request("POST", "/detect", "123", "nonce", b"body")
    assert crypto.verify_signature("POST", "/detect", "123", "nonce", b"body", sig)


def test_verify_rejects_tampered_method() -> None:
    crypto = Crypto(generate_key_b64())
    sig = crypto.sign_request("POST", "/detect", "123", "nonce", b"body")
    assert not crypto.verify_signature("GET", "/detect", "123", "nonce", b"body", sig)


def test_signature_depends_on_body() -> None:
    crypto = Crypto(generate_key_b64())
    a = crypto.sign_request("POST", "/detect", "123", "nonce", b"a")
    b = crypto.sign_request("POST", "/detect", "123", "nonce", b"b")
    assert a != b


def test_signature_depends_on_nonce() -> None:
    crypto = Crypto(generate_key_b64())
    a = crypto.sign_request("POST", "/detect", "123", "n1", b"body")
    b = crypto.sign_request("POST", "/detect", "123", "n2", b"body")
    assert a != b


def test_verify_rejects_wrong_signature_type() -> None:
    crypto = Crypto(generate_key_b64())
    assert not crypto.verify_signature("POST", "/detect", "123", "nonce", b"body", None)


def test_tampered_payload_fails() -> None:
    crypto = Crypto(generate_key_b64())
    payload = bytearray(crypto.encrypt(b"data"))
    payload[-1] ^= 0x01
    with pytest.raises(Exception):
        crypto.decrypt(bytes(payload))
