# Security Audit · Amber Falcon · 2026-09-13

| Поле | Значение |
|------|----------|
| Статус | PASSED WITH WARNINGS |
| Прогон | amber-falcon |
| Уровень | full |
| Охват | leaks + code |
| Модель | deepseek-v4-pro |
| Дата | 2026-09-13 |

## Сводка

Трек A · Секреты/ключи:   0  (Crit 0 / High 0)
Трек A · PII/экспозиция:   0 Critical/High/Medium (Low/Info: 6)
Трек A · История git:      0
Трек B · Инъекции/exec:    0 Critical/High/Medium (Info: 2)
Трек B · Authz/крипто:     3 Medium (TLS, static token)
Трек B · Зависимости:      3 Medium (unpinned/version range)
Инфра/CI:                  0 Critical/High/Medium (Low/Info: 7)

Severity: Crit 0 · High 0 · Med 3 · Low 7 · Info 13
Готовность: 8/10
Вердикт: PASSED WITH WARNINGS

## Находки

| Severity | Категория | Файл:строка | Описание | Рекомендация |
|----------|-----------|-------------|----------|--------------|
| Medium | tls | remote_server.py:33 | Remote-сервер слушает plain HTTP, auth-токен передаётся открытым текстом; статичен, не ротируется | Не связывать с 0.0.0.0 без TLS/VPN; добавить --tls-cert/--tls-key или обратный прокси |
| Medium | known-vuln | requirements.txt | cryptography закреплён только снизу (>=42.0), старые ветки имеют advisory | Держать актуальную ветку, зафиксировать верхнюю границу |
| Medium | outdated | requirements.txt | numpy>=1.24 и opencv-python>=4.8 допускают major-обновления, ломающие ABI | Ограничить диапазон, закрепить проверенную пару версий |
| Low | tls | remote.py:118 | ssl._create_unverified_context() отключает проверку TLS (гейт только development) | Оставить гейт; не включать DOTEYE_REMOTE_INSECURE в прод |
| Low | docker-latest | Dockerfile:5 | Базовый образ python:3.12-slim не зафиксирован по диджесту | Закрепить тег/диджест |
| Low | compose-secret | docker-compose.yml:17 | env_file монтирует .env целиком | Не писать в образ (уже в .dockerignore); отдельный файл с нужными переменными |
| Low | compose-secret | docker-compose.yml:48 | remote-порт наружу только 127.0.0.1; внутри 0.0.0.0 | Держать loopback наружу, не публиковать напрямую |
| Low | unpinned | requirements.txt | Все зависимости без верхних границ и хешей | Добавить верхние границы/хеши (pip-tools, uv lock) |
| Low | no-lockfile | requirements.txt | Нет lockfile и --require-hashes | Генерировать lockfile; CI уже гоняет pip-audit |
| Low | typosquat | requirements.txt | Транзитивные ultralytics-* имена требуют проверки происхождения | Зафиксировать в lockfile, проверить источник |
| Info | зарезервировано | — | — | — |
