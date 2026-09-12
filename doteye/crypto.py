"""Криптография: AES-256-GCM для кадров, embeddings и передачи на remote-сервер.

Ключ - 32 байта, передаётся в base64 через DOTEYE_CRYPTO_KEY.
Никогда не логировать ключ и IV.
"""

from __future__ import annotations

import base64
import hashlib
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

    def auth_token(self) -> str:
        """Производный токен для заголовка remote, не сам ключ."""
        return hashlib.sha256(self._key + b"doteye-remote").hexdigest()


def generate_key_b64() -> str:
    return base64.b64encode(os.urandom(32)).decode()


# Вспомогательная утилита, чтобы сгенерировать ключ для .env:
#   python -m doteye.crypto
if __name__ == "__main__":
    print(generate_key_b64())
