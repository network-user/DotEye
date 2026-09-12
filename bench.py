"""Бенчмарк детекторов DotEye.

Гоняет выбранный детектор на синтетических кадрах и печатает FPS и
среднее время инференса. Помогает подобрать модель под железо.

Запуск:
    python bench.py                       # текущий бэкенд из env/auto
    python bench.py --backend motion
    python bench.py --backend yolo --model yolov8n.pt --device cpu
    python bench.py --frames 60 --width 640 --height 480
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from doteye.detector import build_detector, yolo_available
from doteye.models import YOLO_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Бенчмарк детекторов DotEye")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "yolo", "yunet", "motion"])
    parser.add_argument("--model", default="yolov8n.pt", choices=list(YOLO_MODELS))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--face-model", default="", help="ONNX YuNet для backend=yunet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend in ("auto", "yolo") and not yolo_available():
        print("[bench] ultralytics не установлен, yolo недоступен")

    detector = build_detector(
        args.backend, args.model, args.device, args.conf, False, "", args.face_model
    )
    print(f"[bench] backend={detector.backend} model={args.model} device={args.device}")
    print(f"[bench] кадр {args.width}x{args.height}, {args.frames} замеров")

    frame = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        detector.detect(frame)

    times: list[float] = []
    for _ in range(args.frames):
        start = time.perf_counter()
        detector.detect(frame)
        times.append(time.perf_counter() - start)

    detector.close()

    avg = sum(times) / len(times)
    p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
    print(f"[bench] среднее: {avg * 1000:.1f} мс  ({1 / avg:.1f} FPS)")
    print(f"[bench] p95:     {p95 * 1000:.1f} мс")
    print(f"[bench] мин/макс: {min(times) * 1000:.1f} / {max(times) * 1000:.1f} мс")


if __name__ == "__main__":
    main()
