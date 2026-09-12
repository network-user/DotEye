# DotEye - образ для камеры или remote-сервера инференса.
# Базовый образ: python:3.12-slim. Для ARM (Raspberry Pi) тот же Dockerfile
# собирается под arm64 без изменений (см. docker-compose.yml, platform).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgl1/libglib2 нужны opencv; ffmpeg - для RTSP/MJPEG;
# espeak-ng и alsa-utils - локальная тревога и TTS в контейнере.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg espeak-ng alsa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY doteye ./doteye
COPY remote_server.py bench.py run.py ./

# Непривилегированный пользователь.
RUN useradd -m -u 1000 doteye && mkdir -p /app/data && chown -R doteye:doteye /app
USER doteye

# Порт remote-сервера (если запускается в этом режиме).
EXPOSE 8099

# По умолчанию - бот+пайплайн. Для remote-сервера переопредели command
# в docker-compose или: docker run ... python -m doteye.remote_server
CMD ["python", "-m", "doteye.main"]
