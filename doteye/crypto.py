"""Криптография: AES-256-GCM для кадров, embeddings и передачи на remote-сервер.

Ключ - 32 байта, передаётся в base64 через DOTEYE_CRYPTO_KEY.
Никогда не логировать ключ и IV.
"""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class Crypto:
    def __init__(self, key_b64: str) -> None:
        if not key_b64:
            raise ValueError("DOTEYE_CRYPTO_KEY не задан")
        self._key = base64.b64decode(key_b64)
        if len(self._key) != 32:
            raise ValueError("DOTEYE_CRYPTO_KEY должен декодироваться в 32 байта")
        self._aesgcm = AESGCM(self._key)

    def encrypt(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(12)
        ct = self._aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ct

    def decrypt(self, payload: bytes) -> bytes:
        nonce, ct = payload[:12], payload[12:]
        return self._aesgcm.decrypt(nonce, ct, None)


def generate_key_b64() -> str:
    return base64.b64encode(os.urandom(32)).decode()


# Вспомогательная утилита, чтобы сгенерировать ключ для .env:
#   python -m doteye.crypto
if __name__ == "__main__":
    print(generate_key_b64())
