> Последний прогон: silver-otter · 2026-09-13. Снимок: [2026-09-13-silver-otter.md](2026-09-13-silver-otter.md) · история: [docs/audit/](.)

# Security Audit · Silver Otter · 2026-09-13

| Поле | Значение |
|------|----------|
| Статус | PASSED |
| Прогон | silver-otter |
| Уровень | full |
| Охват | leaks + code |
| Модель | deepseek-v4-pro |
| Дата | 2026-09-13 |

## Сводка

Трек A · Секреты/ключи:   0  (Crit 0 / High 0)
Трек A · PII/экспозиция:   0 Critical/High/Medium (Low/Info: 5)
Трек A · История git:      0
Трек B · Инъекции/exec:    0
Трек B · Authz/крипто:     0
Трек B · Зависимости:      0
Инфра/CI:                  0 Critical/High/Medium (Low/Info: 3)

Severity: Crit 0 · High 0 · Med 0 · Low 1 · Info 8
Готовность: 10/10
Вердикт: PASSED

## Находки

| Severity | Категория | Файл:строка | Описание | Рекомендация |
|----------|-----------|-------------|----------|--------------|
| Low | .gitignore | .gitignore | Не было паттернов `*.pem`/`*.key`/`*.p12`/`*.pfx`; ключей/сертификатов в дереве нет, но гигиена требует закрыть | Добавлены паттерны ключей/сертификатов (устранено) |
| Info | PII | doteye/bot.py:1809 | Внутренний IP как placeholder в подсказке UI, не реальный хост | Оставить; можно обезличить пример |
| Info | supply-chain | requirements.txt | Верхние границы добавлены для opencv/numpy/cryptography; для aiogram/ultralytics/insightface/onnxruntime/pyttsx3/pywin32 lockfile желателен, не обязателен | lockfile (uv/pip-tools) + require-hashes как бэклог |
| Info | deps | requirements.txt | ultralytics исторически в инцидентах PyPI; pip-audit гоняется в CI | Держать pip-audit в CI |
| Info | tls | remote.py:126 | ssl._create_unverified_context гейтится DOTEYE_ENV=development; в проде недостижим | Не включать DOTEYE_REMOTE_INSECURE в проде |
| Info | auth | remote.py / crypto.py | Remote-токен производный от AES-ключа, hmac.compare_digest; ротации/nonce нет, импакт низкий без ключа | Опционально HMAC с timestamp/nonce |
| Info | ssrf | config.py:106 | Loopback проходит без allowlist; источники задаёт только админ | Для remote требовать явный allowlist и для loopback |
| Info | infra | Dockerfile:5 | Базовый образ python:3.12-slim без диджеста | Закреплять по диджесту при релизе |
| Info | infra | .github/workflows/checks.yml | CI чист: нет pull_request_target, permissions contents: read, есть pip-audit | — |

## Устранено в этом прогоне

| Было | Severity | Файл | Как исправлено |
|------|----------|------|----------------|
| Remote-сервер слушал plain HTTP, auth-токен открытым текстом; нет TLS | Medium | remote_server.py, doteye/remote.py | Добавлены `--tls-cert`/`--tls-key`, `build_server_ssl_context` (TLS 1.2+), `_TLSHTTPServer`; plain HTTP вне loopback отвергается |
| `cryptography>=42.0` без верхней границы, старые ветки с advisory | Medium | requirements.txt | `cryptography>=43.0,<51` |
| `numpy>=1.24`, `opencv-python>=4.8` допускали major-обновления, ломающие ABI | Medium | requirements.txt | `numpy>=1.24,<2.0`, `opencv-python>=4.8,<5.0` |
| TLS-рукопожатие без таймаута блокировало accept-loop (DoS) | High | doteye/remote.py | Таймаут сокета до `wrap_socket`, листенеру установлен таймаут |
| Compose remote-профиль падал бы из-за TLS-гейта | Low | docker-compose.yml, deploy/README.md | Compose передаёт `--tls-*`, монтирует `tls/` |
| Нет паттернов ключей/сертификатов в .gitignore | Low | .gitignore | Добавлены `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.log` |
