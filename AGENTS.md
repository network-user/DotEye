# AGENTS.md

> Инструкции для AI coding agents. Человеческий обзор - в [README.md](README.md).
> Сгенерировано скиллом `sync-project-rules`. Источник правды - код репозитория.

## Профиль проекта

- **Тип:** bot (Telegram-бот на aiogram 3)
- **Аудитория:** internal
- **Runtime:** Python 3.12
- **Монорепо:** нет

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
cp .env.example .env              # заполни DOTEYE_BOT_TOKEN
python -m doteye.crypto           # сгенерировать DOTEYE_CRYPTO_KEY
python -m doteye.main
```

Точка входа: `doteye/main.py` (или `python run.py`).

## Сборка и проверки

| Действие | Команда |
|----------|---------|
| Установка | `pip install -r requirements.txt` |
| Установка (dev + тесты) | `pip install -r requirements-dev.txt` |
| Запуск | `python -m doteye.main` |
| Remote-сервер | `python -m doteye.remote_server --host 127.0.0.1 --port 8099` |
| Компиляция (быстрая проверка) | `python -m py_compile doteye/main.py doteye/bot.py doteye/camera.py doteye/config.py doteye/crypto.py doteye/detector.py doteye/models.py doteye/pipeline.py doteye/recognizer.py doteye/remote.py doteye/remote_server.py doteye/runtime.py doteye/storage.py doteye/tracker.py doteye/zones.py doteye/annotate.py doteye/audio.py doteye/tts.py doteye/voice.py remote_server.py run.py bench.py` |
| Тесты | `python -m pytest tests -q` |
| Бенчмарк | `python bench.py` |
| Docker | `docker compose up -d --build doteye` |
| Lint / typecheck | — |

## Структура репозитория

```
doteye/
├── main.py        точка входа: собирает пайплайн + бота, SQLite outbox, shutdown
├── config.py      Settings из env, валидация и allowlist сетевых URL
├── runtime.py     env-настройки + переопределения из чата (Storage)
├── models.py      каталог YOLO-моделей с описаниями для панели
├── bot.py         aiogram 3: роутер, FSM-диалоги, админ-панель, уведомления
├── camera.py      источники кадров: вебка / RTSP / MJPEG, reader-поток и reconnect backoff
├── detector.py    детекция: yolo | yunet | motion, auto-деградация, remote
├── remote.py      ограниченный HTTP transport + AES-GCM: клиент и сервер remote-инференса
├── remote_server.py модульная точка входа remote inference
├── recognizer.py  лицо -> embedding (insightface) + DummyRecognizer fallback
├── crypto.py      AES-256-GCM (кадры, embeddings, remote)
├── tracker.py     IoU-трекер входа/выхода
├── zones.py       ROI кадра (0..1)
├── annotate.py    кроп бокса и рамки на JPEG
├── audio.py       сирена WAV, плеер winsound/aplay/afplay
├── tts.py         pyttsx3 / espeak-ng / dummy
├── voice.py       тревога, фразы, ручная озвучка (отдельный поток)
├── pipeline.py    цикл камера -> трек enter/exit -> событие, кэш кадра, prune
└── storage.py     SQLite WAL: people, embeddings, events, settings, notification outbox
tests/             pytest: crypto, storage, pipeline, detector, models, bot, remote, tracker, zones, audio, voice
bench.py           бенчмарк детектора: FPS и время инференса
remote_server.py   точка входа remote-сервера инференса
run.py             альтернативная точка входа
Dockerfile         образ; docker-compose.yml; deploy/ (systemd, инструкция)
docs/cover.svg     обложка DotBioSite
```

## Соглашения

- **Язык документации:** русский. Комментарии в коде - русский.
- **Стиль кода:** следовать существующим файлам; type hints обязательны, `from __future__ import annotations`.
- **Именование:** нижний регистр с подчёркиванием; ABC-слои с `<Layer>` + `build_<layer>()` фабриками.
- **Хранение:** SQLite без ORM (stdlib `sqlite3`), схема в `storage.py`.
- **Слои отделены:** камера / детектор / распознавание / пайплайн / голос каждый в своём модуле, связаны интерфейсами ABC.

## Переменные окружения

| Переменная | Назначение |
|------------|------------|
| `DOTEYE_BOT_TOKEN` | токен Telegram-бота |
| `DOTEYE_ADMIN_IDS` | id админов через запятую |
| `DOTEYE_ENV` | `production` по умолчанию; `development` нужен для dev-only опций |
| `DOTEYE_ALLOW_OPEN_ACCESS` | `1` = пустой список админов пускает всех, только при `DOTEYE_ENV=development` |
| `DOTEYE_ALLOWED_URL_HOSTS` | точный CSV allowlist хостов сетевых камер и remote |
| `DOTEYE_DETECT_MODE` | `presence` \| `identity` |
| `DOTEYE_CAMERA_SOURCE` | `0` = вебка, rtsp/http; несколько через `\|` |
| `DOTEYE_DETECTOR` | `auto` \| `yolo` \| `yunet` \| `motion` |
| `DOTEYE_MODEL_PATH` | путь к YOLO-модели |
| `DOTEYE_FACE_MODEL` | путь к ONNX-модели YuNet (для детектора yunet) |
| `DOTEYE_DEVICE` | `cpu` \| `cuda` \| `mps` |
| `DOTEYE_MIN_CONFIDENCE` | порог детекции 0..1 |
| `DOTEYE_COOLDOWN_SECONDS` | пауза повторного входа, сек |
| `DOTEYE_JPEG_QUALITY` | качество JPEG для кадров событий |
| `DOTEYE_EVENTS_LIMIT` | сколько событий показывает `/events` |
| `DOTEYE_FACE_THRESHOLD` | порог дистанции embedding (identity) |
| `DOTEYE_ARMED` | `1` = охрана включена |
| `DOTEYE_QUIET_HOURS` | тихие часы `HH:MM-HH:MM` |
| `DOTEYE_NOTIFY_EXIT` | `1` = уведомлять о выходе |
| `DOTEYE_IMGSZ` | размер входа YOLO, по умолчанию 640 |
| `DOTEYE_EVENTS_MAX` | лимит строк событий в БД |
| `DOTEYE_EVENTS_TTL_DAYS` | TTL событий, дни |
| `DOTEYE_ZONES` | JSON зон кадра |
| `DOTEYE_REMOTE_PROCESSING` | `1` = вынос инференса на сервер |
| `DOTEYE_REMOTE_URL` | HTTPS-адрес remote-сервера; HTTP допустим только для loopback/development |
| `DOTEYE_REMOTE_FALLBACK` | `1` = локальный детектор при падении remote |
| `DOTEYE_REMOTE_INSECURE` | `1` = не проверять TLS remote, только при `DOTEYE_ENV=development` |
| `DOTEYE_CRYPTO_KEY` | base64 32 байта для AES-GCM |
| `DOTEYE_VOICE` | `1` = авто-фразы (приветствие, тревога по событию) |
| `DOTEYE_VOICE_ALARM` | `1` = тревога по незнакомцу |
| `DOTEYE_VOICE_SIREN` | `1` = двухтональная сирена |
| `DOTEYE_VOICE_SPEECH` | `1` = TTS |
| `DOTEYE_VOICE_WELCOME` | `1` = «добро пожаловать, {name}» |
| `DOTEYE_VOICE_GOODBYE` | `1` = прощание при выходе известного |
| `DOTEYE_VOICE_PRESENCE` | `1` = фраза при любом входе в presence |
| `DOTEYE_VOICE_ARMED_ANNOUNCE` | `1` = озвучивать вкл/выкл охраны |
| `DOTEYE_VOICE_ALARM_ON_PRESENCE` | `1` = тревога на любой вход в presence |
| `DOTEYE_VOICE_MUTE_QUIET` | `1` = глушить приветствия в тихие часы (сирена нет) |
| `DOTEYE_VOICE_REPEAT_SECONDS` | период повтора фразы тревоги, сек |
| `DOTEYE_VOICE_TIMEOUT_SECONDS` | автоснятие, сек (`0` = пока не снимут) |
| `DOTEYE_VOICE_GRACE_SECONDS` | пауза после ухода незнакомца, сек |
| `DOTEYE_VOICE_CLEAR_ON` | `both` \| `known` \| `exit` |
| `DOTEYE_VOICE_VOLUME` | громкость 0..1 |
| `DOTEYE_VOICE_RATE` | скорость речи 0.4..2.5 |
| `DOTEYE_VOICE_TTS_VOICE` | id голоса TTS |
| `DOTEYE_VOICE_COOLDOWN_SECONDS` | пауза повторного приветствия, сек |

Не читай `.env`. Не коммить секреты.

## Что делать агенту

- Перед правками прочитай затронутые файлы и соседний код.
- После изменений запусти `python -m py_compile doteye/*.py run.py` и `python -m pytest tests -q`.
- **README-sync:** при глобальных изменениях (новые/удалённые модули, зависимости, команды, смена архитектуры или runtime) обнови `README.md` через `generate-readme` и `AGENTS.md` через `sync-project-rules` - в том числе пересчёт LoC. Мелкие правки README не трогают. Блок `<!-- disclaimer:start -->…<!-- disclaimer:end -->` в README - авторский: при регенерации перенеси дословно, как блок аудита.
- Не латай разметку README вручную - перегенерируй скиллом.
- Минимальный diff - не рефактори несвязанный код.
- Числа, пути, версии - только из репозитория.

## Чего не делать

- Не выдумывать команды, зависимости, env, API endpoints.
- Не менять `LICENSE` без явного запроса пользователя.
- Не коммитить секреты, токены, `.env`.
- Не удалять маркеры `<!-- loc:start -->` / `<!-- loc:end -->` и `<!-- disclaimer:start -->` / `<!-- disclaimer:end -->` в README.
- Не логировать `DOTEYE_CRYPTO_KEY` и IV (в `crypto.py`).
- Не писать «ставь в прод» и «стек свободный»: Ultralytics AGPL-3.0, веса InsightFace research-only; код репозитория All Rights Reserved.
- Не рекомендовать `identity` в офисе, подъезде, на улице; не учить проброс RTSP/MJPEG камеры в WAN.

## Документация

- [README.md](README.md) - запуск, команды бота, стек, архитектура, отказ от ответственности, лицензия
- [LICENSE](LICENSE) - All Rights Reserved

## DotCore

Проект следует стандарту DotCore: плоский технический README, SVG-обложка DotBioSite, LoC-бейдж. «Обнови README» → `generate-readme`; «обнови правила» → `sync-project-rules`.
