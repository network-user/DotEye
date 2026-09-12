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
| Запуск | `python -m doteye.main` |
| Компиляция (быстрая проверка) | `python -m py_compile doteye/main.py doteye/bot.py doteye/camera.py doteye/config.py doteye/crypto.py doteye/detector.py doteye/models.py doteye/pipeline.py doteye/recognizer.py doteye/runtime.py doteye/storage.py run.py` |
| Тесты | `python -m pytest tests -q` |
| Lint / typecheck | — |

## Структура репозитория

```
doteye/
├── main.py        точка входа: собирает пайплайн + бота, очередь событий, shutdown
├── config.py      Settings из env, дефолты
├── runtime.py     env-настройки + переопределения из чата (Storage)
├── models.py      каталог YOLO-моделей с описаниями для панели
├── bot.py         aiogram 3: роутер, FSM-диалоги, админ-панель, уведомления
├── camera.py      источники кадров: вебка / RTSP / MJPEG (фабрика)
├── detector.py    детекция: yolo | yunet | motion, auto-деградация, remote-заглушка
├── recognizer.py  лицо -> embedding (insightface) + DummyRecognizer fallback
├── crypto.py      AES-256-GCM (кадры, embeddings, remote)
├── pipeline.py    цикл камера -> детекция -> событие, пересборка, кулдаун
└── storage.py     SQLite (thread-safe): people, events, settings
tests/             pytest: crypto, storage, pipeline, detector
run.py             альтернативная точка входа
docs/cover.svg     обложка DotBioSite
```

## Соглашения

- **Язык документации:** русский. Комментарии в коде - русский.
- **Стиль кода:** следовать существующим файлам; type hints обязательны, `from __future__ import annotations`.
- **Именование:** нижний регистр с подчёркиванием; ABC-слои с `<Layer>` + `build_<layer>()` фабриками.
- **Хранение:** SQLite без ORM (stdlib `sqlite3`), схема в `storage.py`.
- **Слои отделены:** камера / детектор / распознавание / пайплайн каждый в своём модуле, связаны интерфейсами ABC.

## Переменные окружения

| Переменная | Назначение |
|------------|------------|
| `DOTEYE_BOT_TOKEN` | токен Telegram-бота |
| `DOTEYE_ADMIN_IDS` | id админов через запятую |
| `DOTEYE_DETECT_MODE` | `presence` \| `identity` |
| `DOTEYE_CAMERA_SOURCE` | `0` = вебка, либо rtsp/http адрес |
| `DOTEYE_DETECTOR` | `auto` \| `yolo` \| `yunet` \| `motion` |
| `DOTEYE_MODEL_PATH` | путь к YOLO-модели |
| `DOTEYE_FACE_MODEL` | путь к ONNX-модели YuNet (для детектора yunet) |
| `DOTEYE_DEVICE` | `cpu` \| `cuda` \| `mps` |
| `DOTEYE_MIN_CONFIDENCE` | порог детекции 0..1 |
| `DOTEYE_COOLDOWN_SECONDS` | пауза между уведомлениями |
| `DOTEYE_JPEG_QUALITY` | качество JPEG для кадров событий |
| `DOTEYE_EVENTS_LIMIT` | сколько событий показывает `/events` |
| `DOTEYE_FACE_THRESHOLD` | порог дистанции embedding (identity) |
| `DOTEYE_REMOTE_PROCESSING` | `1` = вынос инференса на сервер |
| `DOTEYE_REMOTE_URL` | адрес remote-сервера |
| `DOTEYE_CRYPTO_KEY` | base64 32 байта для AES-GCM |

Не читай `.env`. Не коммить секреты.

## Что делать агенту

- Перед правками прочитай затронутые файлы и соседний код.
- После изменений запусти `python -m py_compile doteye/*.py run.py`.
- **README-sync:** при глобальных изменениях (новые/удалённые модули, зависимости, команды, смена архитектуры или runtime) обнови `README.md` через `generate-readme` и `AGENTS.md` через `sync-project-rules` - в том числе пересчёт LoC. Мелкие правки README не трогают.
- Не латай разметку README вручную - перегенерируй скиллом.
- Минимальный diff - не рефактори несвязанный код.
- Числа, пути, версии - только из репозитория.

## Чего не делать

- Не выдумывать команды, зависимости, env, API endpoints.
- Не менять `LICENSE` без явного запроса пользователя.
- Не коммитить секреты, токены, `.env`.
- Не удалять маркеры `<!-- loc:start -->` / `<!-- loc:end -->` в README.
- Не логировать `DOTEYE_CRYPTO_KEY` и IV (в `crypto.py`).

## Документация

- [README.md](README.md) - запуск, команды бота, стек, архитектура, план работ
- [LICENSE](LICENSE) - All Rights Reserved

## DotCore

Проект следует стандарту DotCore: плоский технический README, SVG-обложка DotBioSite, LoC-бейдж. «Обнови README» → `generate-readme`; «обнови правила» → `sync-project-rules`.
