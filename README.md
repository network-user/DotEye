# .ядро

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Platform-Telegram%20%7C%20Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat" alt="Platform" />
  <img src="https://img.shields.io/badge/Category-Bot-orange?style=flat" alt="Category" />
  <!-- loc:start --><img src="https://img.shields.io/badge/lines_of_code-7813-lightgrey?style=flat" alt="7813 lines of code" /><!-- loc:end -->
</p>

<img src="docs/cover.svg" width="720" alt="DotEye">

DotEye - лабораторный Telegram-бот, который определяет, кто зашёл в комнату. Подключает вебку, IP-камеру или несколько источников сразу, детектирует людей через YOLO и шлёт уведомления только на вход и выход. Настройка ведётся в чате: охрана, тихие часы, зоны кадра, эталоны лиц, голос и тревога; не продукт охраны, см. [отказ от ответственности](#отказ-от-ответственности).

## Что внутри

- **Охрана и тихие часы**: кнопка вкл/выкл и интервал `HH:MM-HH:MM`, чтобы ночью не спамить Telegram.
- **Вход/выход по трекеру**: IoU-сопровождение боксов, событие когда человек появился или исчез, а не пока стоит в кадре.
- **Несколько людей в кадре**: каждый бокс кропается и в identity сравнивается со всеми эталонами.
- **Кадр с подписью**: на фото в Telegram рисуются бокс, имя и уверенность. У неизвестного кнопки «Это кто?» и «Снять тревогу».
- **Голос и тревога**: сирена плюс TTS из динамиков машины. Незнакомец в режиме identity запускает тревогу; человек из списка в кадре снимает её. Приветствие, прощание и ручная фраза («отойди от двери») настраиваются в панели.
- **Панель**: люди и события списками с кнопками, пагинация, `/cancel` для FSM, здоровье камеры и фактический бэкенд детектора.
- **Несколько камер и зоны**: источники через `|`, отдельный reader с последним кадром для каждой камеры, reconnect backoff и ROI в координатах 0..1.
- **Remote с fallback**: ограниченный HTTP API, AES-GCM, проверка схемы ответов, локальный fallback и HTTPS/VPN для внешней сети.
- **Доставка событий**: SQLite outbox с зашифрованным JPEG, идемпотентностью и повторной отправкой после ошибки Telegram.

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

cp .env.example .env              # заполни DOTEYE_BOT_TOKEN и DOTEYE_ADMIN_IDS
python -m doteye.crypto           # сгенерировать DOTEYE_CRYPTO_KEY

python -m doteye.main
```

Для разработки и тестов:

```bash
pip install -r requirements-dev.txt   # тянет runtime + pytest
python -m pytest tests -q
python bench.py                        # бенчмарк детектора
```

Русский TTS на Windows берёт SAPI (пакет «Ирина» или другой ru-голос). На Linux нужен `espeak-ng`. Если синтезатора нет, сирена всё равно играет.

## Команды бота (в чате Telegram)

Доступ - только для id из `DOTEYE_ADMIN_IDS`. Пустой список никого не пускает. Для локальной отладки `DOTEYE_ALLOW_OPEN_ACCESS=1` требует `DOTEYE_ENV=development`.
При старте бот шлёт админам статус и кнопку панели.

| Команда | Назначение |
|---------|-----------|
| `/panel` | админ-панель на inline-кнопках |
| `/start` | приветствие |
| `/status` | текущие настройки и здоровье |
| `/mode` | переключить presence / identity |
| `/camera` | источник (0, rtsp/http; несколько через `\|`) |
| `/detector` | бэкенд: auto / yolo / yunet / motion |
| `/device` | устройство: cpu / cuda / mps |
| `/model` | выбрать YOLO-модель |
| `/confidence` | порог детекции 0..1 |
| `/cooldown` | пауза повторного входа, сек |
| `/people` | список известных людей |
| `/add` | добавить человека (имя + фото лица) |
| `/remove` | удалить человека по имени |
| `/events` | события с кадрами, по страницам |
| `/say` | сказать вслух через динамики |
| `/alarm` | ручная тревога; `/alarm off` снимает |
| `/cancel` | отменить текущий ввод |
| `/help` | справка |

## Админ-панель

`/panel` открывает inline-панель: охрана, режим, детектор (с подписью следующего), устройство, YOLO-модель, превью из кэша кадра, здоровье камеры, люди, события, камера, remote, порог, кулдаун, интервал, порог лица, тихие часы, уведомления о выходе, зоны, голос и тревога.

Подменю «Голос / тревога»: авто-фразы, сирена, приветствие/прощание, повтор и таймаут, громкость, скорость, фразы с `{name}`, выбор TTS-голоса, «Сказать вслух», тест сирены, ручной пуск и снятие.

Выбор модели (`/model` или кнопка «YOLO-модель») показывает карточки: размер весов, число параметров, примерная скорость и точность. Смена модели переключает детектор на `yolo` и пересобирает его на ходу.

Кнопка «Это кто?» на неизвестном входе привязывает событие к человеку и пишет новый эталон с кропа. У человека можно хранить несколько фото.

Тревога по незнакомцу работает в режиме `identity` (нужны эталоны лиц). В `presence` её можно включить отдельно: любой вход считается чужим, снять сможет только кнопка, таймаут или уход из кадра.

| Модель | Размер | Параметры | Скорость (CPU) | Точность | Когда брать |
|--------|--------|-----------|----------------|----------|-------------|
| YOLOv8n | ~6 МБ | 3.2M | высокая | базовая | слабое железо, Raspberry Pi |
| YOLOv8s | ~22 МБ | 11.2M | средняя | хорошая | ноутбук без GPU |
| YOLOv8m | ~50 МБ | 25.9M | низкая | высокая | ПК с GPU |
| YOLOv8l | ~84 МБ | 43.7M | очень низкая | очень высокая | сервер/GPU |
| YOLOv8x | ~131 МБ | 68.2M | минимальная | максимальная | максимум точности на GPU |

Каждая следующая модель крупнее, точнее и медленнее. На CPU бери n/s, с GPU - m и выше.

## Стек

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/aiogram_3-26A5E4?style=for-the-badge&logo=telegram&logoColor=white" alt="aiogram" />
  <img src="https://img.shields.io/badge/Ultralytics_YOLO-111820?style=for-the-badge" alt="Ultralytics YOLO" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/cryptography-555555?style=for-the-badge" alt="cryptography" />
  <img src="https://img.shields.io/badge/pyttsx3-555555?style=for-the-badge" alt="pyttsx3" />
  <img src="https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
</p>

## Тесты

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

Покрыты crypto, storage (WAL, prune, несколько эталонов), pipeline (вход/выход, identity по кропу, тихие часы, кэш превью, снятие тревоги поздним распознаванием), tracker, zones, каталог моделей, фабрика детекторов, remote (токен, fallback), голос (стейт-машина тревоги, TTS-заглушка) и бот (панель, доступ, callback-хендлеры).

## Бенчмарк

```bash
python bench.py                                  # текущий/auto бэкенд
python bench.py --backend yolo --model yolov8n.pt
```

Печатает среднее время инференса, FPS и p95 на синтетических кадрах.

## Remote-инференс

Камера и инференс могут жить на разных машинах: кадр шифруется AES-256-GCM и уходит POST-ом, заголовок `X-DotEye-Token` - производный от ключа. HTTPS берётся из URL. Если сервер недоступен, клиент уходит на локальный детектор (если включён fallback) и шлёт алерт в чат.

На сервере:

```bash
export DOTEYE_CRYPTO_KEY=<тот же ключ, что у клиента>
python -m doteye.remote_server --host 127.0.0.1 --port 8099 \
    --model yolov8n.pt --device cuda
curl http://localhost:8099/health    # {"status": "ok"}
```

Камера остаётся в LAN. Remote - только через VPN или TLS reverse proxy; прямой HTTP у клиента разрешён на loopback. Не публикуй RTSP, MJPEG и веб-морду камеры в интернет. Имя remote-хоста и сетевых камер - в точный CSV allowlist `DOTEYE_ALLOWED_URL_HOSTS`.

На машине с камерой:

```
DOTEYE_REMOTE_PROCESSING=1
DOTEYE_REMOTE_URL=https://<remote-host>
DOTEYE_ALLOWED_URL_HOSTS=<remote-host>
DOTEYE_REMOTE_FALLBACK=1
DOTEYE_CRYPTO_KEY=<тот же ключ>
```

Протокол: `POST /detect` + токен, тело `{"frame": "<base64(AES-GCM JPEG)>"}` -> `{"boxes": [[x1,y1,x2,y2], ...]}`, плюс `GET /health`. Сервер ограничивает размер запросов, изображения, ответов, соединений и время чтения.

## Деплой

- `Dockerfile` - образ для бота и remote-сервера (Python 3.12 slim + OpenCV/ffmpeg/espeak-ng, непривилегированный пользователь).
- `docker-compose.yml` - сервис `doteye` (камера + инференс) и `remote` (профиль `remote`).
- `deploy/doteye.service` - unit для systemd.
- `deploy/README.md` - Compose, systemd, Raspberry Pi (ARM).

```bash
cp .env.example .env && python -m doteye.crypto
docker compose up -d --build doteye
```

Сирена и TTS играют на машине с пайплайном (там, где динамики). В Docker на Linux для звука нужен `/dev/snd` хоста.

## Архитектура

Бот и пайплайн в одном процессе. Telegram на aiogram 3, OpenCV/YOLO выполняются вне event loop. У каждой камеры свой reader, а события до отправки в Telegram сохраняются в долговечный SQLite outbox. Голос и сирена идут в отдельном потоке: шаг пайплайна не ждёт динамики.

```
doteye/
├── main.py        точка входа: пайплайн + бот + голос, SQLite outbox, shutdown
├── config.py      Settings из env, валидация и allowlist сетевых URL
├── runtime.py     env + переопределения из чата (Storage)
├── models.py      каталог YOLO-моделей для панели
├── bot.py         aiogram 3: панель, FSM, «это кто?», голос, уведомления
├── camera.py      вебка / RTSP / MJPEG, reader-поток и reconnect backoff
├── detector.py    yolo | yunet | motion, motion-gate, remote+fallback
├── tracker.py     IoU-трекер входа и выхода
├── zones.py       ROI кадра в координатах 0..1
├── annotate.py    кроп бокса и рамки на JPEG
├── remote.py      ограниченный transport, AES-GCM, проверка ответов и лимиты
├── recognizer.py  insightface (CPU/CUDA) + DummyRecognizer
├── crypto.py      AES-256-GCM, auth_token для remote
├── pipeline.py    камеры -> трек -> событие, кэш кадра, prune БД
├── audio.py       сирена WAV, winsound/aplay/afplay
├── tts.py         pyttsx3 / espeak-ng / dummy
├── voice.py       стейт-машина тревоги, фразы, ручная озвучка
├── remote_server.py точка входа пакета для remote inference
└── storage.py     SQLite WAL: people, embeddings, events, settings, outbox
tests/             pytest
bench.py           бенчмарк детектора
remote_server.py   точка входа remote-сервера
run.py             альтернативная точка входа
Dockerfile         образ; docker-compose.yml; deploy/
```

Поток данных:

```
камеры (camera.py)
   │  кадр BGR (кэш для превью)
   ▼
motion-gate + детектор  ── yolo / yunet / motion / remote
   │  боксы, фильтр зон
   ▼
IoU-трекер ── enter / active / exit
   │
   ▼
(identity) кроп бокса -> embedding -> min по эталонам человека
   │
   ├── annotate JPEG -> AES-GCM -> storage -> outbox/retry -> Telegram
   └── voice.observe -> сирена/TTS в отдельном потоке
         неизвестный -> тревога
         известный в кадре -> снятие + «добро пожаловать»
         /say и панель -> ручная фраза
```

Инварианты:

- Настройки: `Settings` (env) + `Runtime` (чат поверх env).
- Кадры и embeddings только зашифрованными (AES-256-GCM).
- SQLite без ORM, WAL, индексы и периодический prune по `events_max` и `events_ttl_days`.
- Сетевые URL проходят allowlist; production не допускает open access и небезопасный TLS.
- Уведомление сначала фиксируется в outbox, поэтому временный сбой Telegram не теряет событие.
- Событие на появление/исчезновение трека, кулдаун гасит дребезг.
- Превью не вызывает второй `VideoCapture.read()`.
- Детектор деградирует yolo -> yunet -> motion; remote падает в fallback, а не в «никого нет».
- TTS и сирена не блокируют `pipeline.step()`; ручная озвучка из чата работает без охраны.

<!-- disclaimer:start -->
## Отказ от ответственности

Лабораторный код, не система охраны и не юридическая консультация. Поставляется как есть. Автор не отвечает за то, кого снимают и куда уходит кадр.

Ниже - рамка ограничений, не разрешение «можно / нельзя».

**Персональные данные и биометрия.** Детекция человека (YOLO `person`, YuNet, motion) и распознавание лица (InsightFace, режим `identity`) - разные юридические слои. Фото становится биометрией, когда его используют, чтобы установить личность (ст. 11 152-ФЗ, позиция Роскомнадзора 2024). Режим `identity` с галереей «мама / незнакомец» в эту формулировку попадает: эталон, embedding, сравнение, имя в Telegram. Режим `presence` без имён ближе к обычному видеонаблюдению, но кадр узнаваемого человека всё равно персональные данные. AES-GCM в SQLite 152-ФЗ не отменяет.

Домашнее исключение 152-ФЗ узкое: исключительно личные и семейные нужды и без нарушения прав субъектов. Гость, курьер, сосед в Telegram это исключение выбивают. Офис, подъезд, ресепшен, улица - вне scope репозитория. Типовой регламент VPS («нет нарушений закона») такое не покрывает.

Публикация чужого лица - отдельно ст. 152.1 ГК. Скрины реальных людей без согласия нельзя тащить в README, статьи, issues и демо. Свои тестовые кадры, ты сам, манекен, синтетика.

**Лицензии зависимостей.** Код репозитория - All Rights Reserved. Это не делает стек свободным.

- Ultralytics YOLO (`ultralytics>=8.1` в `requirements.txt`) - AGPL-3.0. По позиции Ultralytics закрытый прод без Enterprise-лицензии и без публикации полного исходника производного продукта с AGPL не сходится.
- InsightFace: код библиотеки MIT, предобученные веса (buffalo и остальные) - non-commercial research only; коммерция требует отдельной лицензии у InsightFace. В `requirements.txt` пакет закомментирован, режим `identity` его всё равно подразумевает.

Фраза «ставь в прод, стек свободный» ломается об эти лицензии ещё до 152-ФЗ.

**Камеры в интернет.** Не публикуй RTSP/MJPEG/веб-морду камеры в WAN и не описывай такой проброс как рабочую схему. [CameraSwarm](https://hunt.io/blog/operation-cameraswarm-dahua-cameras-compromised) (Hunt.io, август 2026): 14 530+ камер Dahua за 35 дней, снэпшоты в Telegram, бэкдор на ~1900 устройствах; часть доступна по серийнику через облачный релей без адреса устройства. Камера живёт в LAN. Allowlist `DOTEYE_ALLOWED_URL_HOSTS` - не замена закрытой сети.
<!-- disclaimer:end -->

## Лицензия

© 2026 .ядро. Все права защищены.

Проприетарный код. Использование, копирование, изменение и распространение запрещены без письменного разрешения автора. Исходный код открыт только для ознакомления. См. [LICENSE](LICENSE).

Зависимости живут под своими лицензиями (Ultralytics AGPL-3.0, веса InsightFace - research-only). См. [отказ от ответственности](#отказ-от-ответственности).
