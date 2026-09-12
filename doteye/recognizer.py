"""Распознавание личности (identity mode) поверх детекции.

Лицо -> embedding -> сравнение с зарегистрированными людьми из Storage.
Embeddings хранятся в Storage зашифрованными (crypto.py), расшифровка и
сопоставление - в pipeline.py.

Основная реализация - insightface `buffalo_l` (512-мерный normed embedding).
Зависимость опциональная: если insightface/onnxruntime не установлены,
`build_recognizer()` вернёт DummyRecognizer с available() == False.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Recognizer(ABC):
    @abstractmethod
    def available(self) -> bool:
        """True, если реализация умеет строить embeddings."""

    @abstractmethod
    def embed(self, frame: np.ndarray) -> bytes | None:
        """Вернуть сериализованный float32 embedding лица или None."""

    @abstractmethod
    def distance(self, a: bytes, b: bytes) -> float:
        """Дистанция между двумя embeddings: 0 = идентичны, больше = дальше."""

    @property
    def name(self) -> str:
        return type(self).__name__


class DummyRecognizer(Recognizer):
    """Заглушка: identity mode выключен на уровне распознавания."""

    def available(self) -> bool:
        return False

    def embed(self, frame: np.ndarray) -> bytes | None:
        return None

    def distance(self, a: bytes, b: bytes) -> float:
        return 1.0


class InsightFaceRecognizer(Recognizer):
    """insightface FaceAnalysis (buffalo_l), CPU или CUDA."""

    def __init__(self, device: str = "cpu") -> None:
        from insightface.app import FaceAnalysis

        self._device = device
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def available(self) -> bool:
        return True

    def embed(self, frame: np.ndarray) -> bytes | None:
        faces = self._app.get(frame)
        if not faces:
            return None
        face = max(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        )
        return np.asarray(face.normed_embedding, dtype="<f4").tobytes()

    def distance(self, a: bytes, b: bytes) -> float:
        va = np.frombuffer(a, dtype="<f4")
        vb = np.frombuffer(b, dtype="<f4")
        if va.size == 0 or vb.size == 0 or va.size != vb.size:
            return 1.0
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9
        return float(1.0 - np.dot(va, vb) / denom)


def build_recognizer(device: str = "cpu") -> Recognizer:
    try:
        return InsightFaceRecognizer(device=device)
    except Exception as exc:
        print(f"[recognizer] insightface недоступен ({exc}); identity -> DummyRecognizer")
        return DummyRecognizer()
