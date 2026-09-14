# Деплой DotEye

Три способа: Docker Compose, systemd, Raspberry Pi (ARM). Что выбрать:

- одна машина с камерой - Docker Compose (`doteye`);
- сервер без Docker - systemd;
- камера на слабом железе + инференс на сервере - remote-режим.

Общее для всех: заполни `.env` (см. `.env.example`), ключ `DOTEYE_CRYPTO_KEY`
должен совпадать у клиента и remote-сервера. Не коммить `.env`.

## 1. Docker Compose

```bash
cp .env.example .env      # заполни токен, admin ids, ключ
python -m doteye.crypto   # сгенерировать DOTEYE_CRYPTO_KEY (нужен venv)

docker compose up -d --build doteye    # бот + пайплайн с камерой
docker compose logs -f doteye
```

Вебка пробрасывается через `devices: /dev/video0`. Если камера внешняя по
RTSP - убери блок `devices` и укажи `DOTEYE_CAMERA_SOURCE=rtsp://...`.
Камера остаётся в LAN: RTSP/MJPEG и веб-морду в интернет не публикуй
(см. отказ от ответственности в корневом README).
Сирена и TTS в контейнере на Linux требуют `/dev/snd` хоста (строка в
`docker-compose.yml` закомментирована).

GPU (NVIDIA): раскомментируй `deploy.resources` в `docker-compose.yml`,
поставь `nvidia-container-toolkit` и `DOTEYE_DEVICE=cuda`.

### Remote-инференс через Compose

Remote-порт остаётся на loopback. Для камеры на другой машине используй TLS reverse proxy: прямой HTTP-порт не публикуй в сеть.

На сервере с GPU положи `server.crt`/`server.key` в каталог `tls/` рядом с
`docker-compose.yml`, затем:

```bash
docker compose --profile remote up -d --build remote
curl -k https://localhost:8099/health     # {"status": "ok"}
```

Compose запускает сервер с `--tls-cert/--tls-key`; без сертификата контейнер
не поднимется (plain HTTP вне loopback отвергается). Для локального режима
без TLS убери `--tls-*` из `command` и оставь `--host 127.0.0.1`.

Свой TLS без reverse proxy - передай сертификат самому серверу
(plain HTTP вне loopback отклоняется):

```bash
python -m doteye.remote_server --host 0.0.0.0 --port 8099 \
    --tls-cert server.crt --tls-key server.key --model yolov8n.pt --device cuda
```

На машине с камерой в `.env`:

```
DOTEYE_REMOTE_PROCESSING=1
DOTEYE_REMOTE_URL=https://<remote-host>
DOTEYE_ALLOWED_URL_HOSTS=<remote-host>
# Нужен только для self-signed TLS по IP: путь к доверенному PEM-файлу.
DOTEYE_REMOTE_CA_CERT=/абсолютный/путь/server.crt
DOTEYE_CRYPTO_KEY=<тот же ключ>
```

Без Docker remote-сервер поднимается так:

```bash
python -m doteye.remote_server --host 127.0.0.1 --port 8099 --model yolov8n.pt --device cuda
```

Для внешнего хоста добавь `--tls-cert server.crt --tls-key server.key`.

### Автодеплой remote без VPN (домен или IP)

Remote-инференс выносится на VPS, а камера и бот остаются дома. Кадры уходят
по `https://` — VPN не обязателен.

Интерактивный генератор пишет `.env.remote` и `docker-compose.remote.yml`:

```bash
python deploy/deploy_remote.py
```

Скрипт предложит выбрать тип развёртывания:

**1. По ДОМЕНУ (рекомендуется)**
- Автоматический HTTPS через Let's Encrypt
- Caddy настроит TLS сертификат сам
- Требуется: доменное имя, указывающее на сервер
- Compose поднимет два сервиса: `remote` (инференс) + `caddy` (TLS-терминатор, порты 80/443)

**2. По IP-АДРЕСУ**
- Самоподписанный TLS с IP в SAN; DotEye проверяет явно переданный PEM-файл
- Нет автоматического выпуска и продления сертификата
- Временный вариант: для постоянного публичного сервиса предпочтителен домен
- Скрипт генерирует готовый Compose и выводит команду `openssl`

Скрипт также спросит:
- `DOTEYE_CRYPTO_KEY` — пусто = сгенерирует; ключ обязан совпасть с машиной камеры;
- устройство (`cpu`/`cuda`) и модель YOLO.

Затем на сервере (в корне репозитория):

```bash
docker compose -f deploy/docker-compose.remote.yml up -d --build
curl https://<домен>/health    # {"status": "ok"}
```

На машине с камерой в `.env`:

```
DOTEYE_REMOTE_PROCESSING=1
DOTEYE_REMOTE_URL=https://<домен-или-IP>:<порт>
DOTEYE_ALLOWED_URL_HOSTS=<домен-или-IP>
# Только для IP с self-signed TLS: абсолютный путь к скопированному server.crt.
DOTEYE_REMOTE_CA_CERT=/абсолютный/путь/server.crt
DOTEYE_CRYPTO_KEY=<тот же ключ>
```

Для IP-варианта генератор создаёт Compose с TLS. Выпусти сертификат командой из
его вывода, скопируй с VPS только `server.crt` на машину с камерой и укажи путь
к нему в `DOTEYE_REMOTE_CA_CERT`. `server.key` остаётся только на VPS.
Контейнер запускается от непривилегированного пользователя с GID `1000`, поэтому
после выпуска сертификата выполни `chown root:1000 deploy/tls/server.key` и
`chmod 640 deploy/tls/server.key`; сертификату достаточно `chmod 644`.
Выбранная YOLO-модель при первом старте скачивается в `deploy/models`; этот
каталог генератор создаёт с владельцем UID/GID `1000`, поэтому read-only контейнер
сохраняет жёсткую файловую изоляцию, но может обновлять кеш модели.
Не выставляй plain HTTP-порт remote в интернет и не включай `DOTEYE_REMOTE_INSECURE`.

## 2. systemd (Linux, без Docker)

```bash
sudo useradd -m -u 1000 doteye
sudo mkdir -p /opt/doteye && sudo chown doteye:doteye /opt/doteye
# скопируй код в /opt/doteye, затем:
cd /opt/doteye
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env            # заполни
.venv/bin/python -m doteye.crypto   # ключ -> в .env

sudo cp deploy/doteye.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now doteye
systemctl status doteye
journalctl -u doteye -f
```

Пути в unit-файле (`/opt/doteye`, пользователь `doteye`) поменяй под себя.
Для камеры пользователю нужен доступ к `/dev/video*` (группа `video`):
`sudo usermod -aG video doteye`.

## 3. Raspberry Pi (ARM)

Образ собирается из того же Dockerfile под arm64. Учти:

- Модель: `yolov8n.pt` или `motion`. На Pi 4 torch работает, но медленно;
  ставь `DOTEYE_DETECTOR=motion` или `yunet` (OpenCV, без torch).
- `DOTEYE_DEVICE=cpu` (GPU у Pi нет).
- Увеличь интервал: `DOTEYE_DETECTION_INTERVAL=2` и
  `DOTEYE_COOLDOWN_SECONDS=60`, чтобы не грузить CPU.
- Память: собери с `--build-arg` при необходимости, либо используй
  `opencv-python-headless` (в образе уже slim-вариант).

Сборка на самой Pi:

```bash
docker compose up -d --build doteye
```

Кросс-сборка с x86 через buildx:

```bash
docker buildx build --platform linux/arm64 -t doteye:arm64 .
```

Быстрый бенчмарк перед выбором модели:

```bash
docker compose run --rm doteye python bench.py --backend motion
docker compose run --rm doteye python bench.py --backend yolo --model yolov8n.pt
```
