"""Криптография: AES-256-GCM для кадров, embeddings и передачи на remote-сервер.

Ключ - 32 байта, передаётся в base64 через DOTEYE_CRYPTO_KEY.
Никогда не логировать ключ и IV.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class Crypto:
    def __init__(self, key_b64: str) -> None:
        if not key_b64:
            raise ValueError("DOTEYE_CRYPTO_KEY не задан")
        try:
            self._key = base64.b64decode(key_b64, validate=True)
        except ValueError as exc:
            raise ValueError("DOTEYE_CRYPTO_KEY должен быть корректным base64") from exc
        if len(self._key) != 32:
            raise ValueError("DOTEYE_CRYPTO_KEY должен декодироваться в 32 байта")
        if len(set(self._key)) == 1:
            raise ValueError(
                "DOTEYE_CRYPTO_KEY слабый (placeholder); сгенерируй: python -m doteye.crypto"
            )
        self._aesgcm = AESGCM(self._key)

    def encrypt(self, plaintext: bytes, aad: bytes | None = None) -> bytes:
        nonce = os.urandom(12)
        ct = self._aesgcm.encrypt(nonce, plaintext, aad)
        return nonce + ct

    def decrypt(self, payload: bytes, aad: bytes | None = None) -> bytes:
        if len(payload) < 28:
            raise ValueError("зашифрованные данные слишком короткие")
        nonce, ct = payload[:12], payload[12:]
        return self._aesgcm.decrypt(nonce, ct, aad)

    def _signing_key(self) -> bytes:
        """Отдельный HMAC-ключ (domain separation от AES-ключа)."""
        return hashlib.sha256(self._key + b"doteye-remote-signing-v1").digest()

    def sign_request(
        self,
        method: str,
        path: str,
        timestamp: str,
        nonce: str,
        body: bytes | None = None,
    ) -> str:
        """HMAC-SHA256 подпись по канонической строке METHOD/PATH/TIMESTAMP/NONCE/SHA256(BODY)."""
        body_sha = hashlib.sha256(body or b"").hexdigest()
        payload = f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{body_sha}".encode()
        return hmac.new(self._signing_key(), payload, hashlib.sha256).hexdigest()

    def verify_signature(
        self,
        method: str,
        path: str,
        timestamp: str,
        nonce: str,
        body: bytes | None,
        signature: str,
    ) -> bool:
        """Сравнить подпись в постоянном времени."""
        if not isinstance(signature, str):
            return False
        expected = self.sign_request(method, path, timestamp, nonce, body)
        return hmac.compare_digest(expected, signature)


def generate_key_b64() -> str:
    return base64.b64encode(os.urandom(32)).decode()


# Вспомогательная утилита, чтобы сгенерировать ключ для .env:
#   python -m doteye.crypto
if __name__ == "__main__":
    print(generate_key_b64())
