"""Слой хранения (SQLite): известные люди, события входа, настройки.

Паттерн Data Access Object без ORM - обычный sqlite3 stdlib.
Соединение открыто с check_same_thread=False, потому что пайплайн работает
в отдельном потоке (asyncio.to_thread), а бот - в event loop; доступ
сериализуется через threading.Lock.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    embedding BLOB,          -- устар.: первый эталон, дублируется в person_embeddings
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS person_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES people (id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,       -- NULL = неизвестный
    frame BLOB,              -- зашифрованный кадр (JPEG без рамок)
    confidence REAL,
    detected_at TEXT NOT NULL,
    event_type TEXT,         -- enter | exit
    boxes TEXT,              -- JSON боксов
    camera_source TEXT,
    zone TEXT,
    FOREIGN KEY (person_id) REFERENCES people (id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS notification_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    admin_id INTEGER NOT NULL,
    dedup_key TEXT NOT NULL UNIQUE,
    payload TEXT NOT NULL,
    jpeg BLOB,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    sent_at TEXT,
    last_error TEXT,
    FOREIGN KEY (event_id) REFERENCES events (id),
    UNIQUE(event_id, admin_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payload BLOB NOT NULL,       -- AES-GCM: только обезличенный тип действия
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_person_embeddings_person_id
    ON person_embeddings(person_id);
CREATE INDEX IF NOT EXISTS idx_events_detected_at ON events(detected_at);
CREATE INDEX IF NOT EXISTS idx_events_person_id ON events(person_id);
CREATE INDEX IF NOT EXISTS idx_notification_outbox_due
    ON notification_outbox(sent_at, next_attempt_at, id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);
"""

_EVENT_COLUMNS = {
    "confidence": "REAL",
    "event_type": "TEXT",
    "boxes": "TEXT",
    "camera_source": "TEXT",
    "zone": "TEXT",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()
        self._people_revision = 0

    def _migrate(self) -> None:
        """Добавить колонки и перенести embeddings со старой схемы."""
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(events)")}
        for name, spec in _EVENT_COLUMNS.items():
            if name not in cols:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {name} {spec}")

        people_cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(people)")}
        if "embedding" not in people_cols:
            self._conn.execute("ALTER TABLE people ADD COLUMN embedding BLOB")

        existing = {
            (row["person_id"], row["embedding"])
            for row in self._conn.execute(
                "SELECT person_id, embedding FROM person_embeddings"
            )
        }
        for row in self._conn.execute(
            "SELECT id, embedding FROM people WHERE embedding IS NOT NULL"
        ):
            key = (row["id"], row["embedding"])
            if key in existing:
                continue
            self._conn.execute(
                "INSERT INTO person_embeddings (person_id, embedding, created_at) "
                "VALUES (?, ?, ?)",
                (row["id"], row["embedding"], _now()),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def people_revision(self) -> int:
        """Монотонная версия эталонов для недорогой инвалидации кэша pipeline."""
        with self._lock:
            return self._people_revision

    # -- people ---------------------------------------------------------

    def add_person(self, name: str, embedding: bytes | None = None) -> int:
        return self.upsert_person(name, embedding)

    def upsert_person(self, name: str, embedding: bytes | None) -> int:
        """Добавить человека или обновить/добавить эталон по имени."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM people WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                cur = self._conn.execute(
                    "INSERT INTO people (name, embedding, created_at) VALUES (?, ?, ?)",
                    (name, embedding, _now()),
                )
                person_id = int(cur.lastrowid)
            else:
                person_id = int(row["id"])
                if embedding is not None:
                    self._conn.execute(
                        "UPDATE people SET embedding = ? WHERE id = ?",
                        (embedding, person_id),
                    )
            if embedding is not None:
                self._conn.execute(
                    "INSERT INTO person_embeddings (person_id, embedding, created_at) "
                    "VALUES (?, ?, ?)",
                    (person_id, embedding, _now()),
                )
            self._conn.commit()
            if embedding is not None:
                self._people_revision += 1
            return person_id

    def add_embedding(self, person_id: int, embedding: bytes) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO person_embeddings (person_id, embedding, created_at) "
                "VALUES (?, ?, ?)",
                (person_id, embedding, _now()),
            )
            self._conn.execute(
                "UPDATE people SET embedding = ? WHERE id = ?",
                (embedding, person_id),
            )
            self._conn.commit()
            self._people_revision += 1
            return int(cur.lastrowid)

    def get_person(self, name: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM people WHERE name = ?", (name,)
            ).fetchone()

    def get_person_by_id(self, person_id: int) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM people WHERE id = ?", (person_id,)
            ).fetchone()

    def list_people(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT p.id, p.name, p.created_at, "
                "  (SELECT COUNT(*) FROM person_embeddings pe "
                "    WHERE pe.person_id = p.id) AS face_count, "
                "  (SELECT COUNT(*) FROM person_embeddings pe "
                "    WHERE pe.person_id = p.id) > 0 AS has_face "
                "FROM people p ORDER BY p.name"
            ).fetchall()

    def list_people_paged(self, limit: int, offset: int) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT p.id, p.name, p.created_at, "
                "  (SELECT COUNT(*) FROM person_embeddings pe "
                "    WHERE pe.person_id = p.id) AS face_count, "
                "  (SELECT COUNT(*) FROM person_embeddings pe "
                "    WHERE pe.person_id = p.id) > 0 AS has_face "
                "FROM people p ORDER BY p.name LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

    def count_people(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM people").fetchone()
            return int(row[0]) if row else 0


    def list_people_with_embeddings(self) -> list[dict[str, Any]]:
        with self._lock:
            people = self._conn.execute(
                "SELECT id, name FROM people ORDER BY name"
            ).fetchall()
            emb_rows = self._conn.execute(
                "SELECT person_id, embedding FROM person_embeddings"
            ).fetchall()
        grouped: dict[int, list[bytes]] = {}
        for row in emb_rows:
            grouped.setdefault(int(row["person_id"]), []).append(row["embedding"])
        out: list[dict[str, Any]] = []
        for person in people:
            embeddings = grouped.get(int(person["id"]), [])
            if not embeddings:
                continue
            out.append({
                "id": int(person["id"]),
                "name": person["name"],
                "embeddings": embeddings,
            })
        return out

    def embeddings_for(self, person_id: int) -> list[bytes]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT embedding FROM person_embeddings WHERE person_id = ? ORDER BY id",
                (person_id,),
            ).fetchall()
        return [row["embedding"] for row in rows]

    def delete_person(self, name: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM people WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                return False
            person_id = int(row["id"])
            self._conn.execute(
                "UPDATE events SET person_id = NULL WHERE person_id = ?", (person_id,)
            )
            self._conn.execute(
                "DELETE FROM person_embeddings WHERE person_id = ?", (person_id,)
            )
            self._conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
            self._conn.commit()
            self._people_revision += 1
            return True

    def delete_person_by_id(self, person_id: int) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT name FROM people WHERE id = ?", (person_id,)
            ).fetchone()
        if row is None:
            return False
        return self.delete_person(row["name"])

    # -- events ---------------------------------------------------------

    def add_event(
        self,
        person_id: int | None,
        frame: bytes | None,
        confidence: float | None = None,
        event_type: str | None = "enter",
        boxes: str | None = None,
        camera_source: str | None = None,
        zone: str | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (person_id, frame, confidence, detected_at, "
                "event_type, boxes, camera_source, zone) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    person_id, frame, confidence, _now(),
                    event_type, boxes, camera_source, zone,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_event(self, event_id: int) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT e.*, p.name AS person_name FROM events e "
                "LEFT JOIN people p ON p.id = e.person_id WHERE e.id = ?",
                (event_id,),
            ).fetchone()

    def update_event_person(self, event_id: int, person_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE events SET person_id = ? WHERE id = ?",
                (person_id, event_id),
            )
            self._conn.commit()

    def recent_events(self, limit: int = 20, offset: int = 0) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT e.*, p.name AS person_name FROM events e "
                "LEFT JOIN people p ON p.id = e.person_id "
                "ORDER BY e.id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

    def count_events(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
            return int(row["n"]) if row else 0

    # -- encrypted audit log ------------------------------------------

    def add_audit_entry(self, encrypted_payload: bytes, keep: int = 1000) -> None:
        """Сохранить зашифрованный, обезличенный тип действия администратора."""
        if not encrypted_payload:
            raise ValueError("audit payload must not be empty")
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log (payload, created_at) VALUES (?, ?)",
                (encrypted_payload, _now()),
            )
            self._conn.execute(
                "DELETE FROM audit_log WHERE id < COALESCE("
                "(SELECT id FROM audit_log ORDER BY id DESC LIMIT 1 OFFSET ?), 0)",
                (max(1, int(keep)) - 1,),
            )
            self._conn.commit()

    def recent_audit_entries(self, limit: int = 30) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT id, payload, created_at FROM audit_log ORDER BY id DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()

    # -- durable notification outbox -----------------------------------

    def enqueue_notification(
        self,
        dedup_key: str,
        event_id: int | None,
        admin_id: int,
        payload: str,
        jpeg: bytes | None = None,
    ) -> int:
        """Сохранить уведомление до отправки, идемпотентно по ``dedup_key``.

        Для событий ключ обычно содержит event_id и admin_id. Для служебных
        alert без event_id вызывающий код передаёт свой UUID или другой
        устойчивый ключ дедупликации.
        """
        if not dedup_key:
            raise ValueError("dedup_key must not be empty")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO notification_outbox "
                "(event_id, admin_id, dedup_key, payload, jpeg, next_attempt_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                (event_id, int(admin_id), dedup_key, payload, jpeg, _now()),
            )
            if cur.lastrowid:
                notification_id = int(cur.lastrowid)
            else:
                row = self._conn.execute(
                    "SELECT id FROM notification_outbox WHERE dedup_key = ?",
                    (dedup_key,),
                ).fetchone()
                if row is None:
                    # event/admin is a second idempotency guard for older
                    # callers which accidentally construct another key.
                    row = self._conn.execute(
                        "SELECT id FROM notification_outbox "
                        "WHERE event_id IS ? AND admin_id = ?",
                        (event_id, int(admin_id)),
                    ).fetchone()
                if row is None:
                    raise RuntimeError("notification outbox insert was not persisted")
                notification_id = int(row["id"])
            self._conn.commit()
            return notification_id

    def claim_due_notifications(
        self, now: str | None = None, limit: int = 20
    ) -> list[sqlite3.Row]:
        """Вернуть ещё не доставленные сообщения, срок повтора которых наступил.

        Один sender в main обрабатывает список последовательно, поэтому не
        нужен ненадёжный lock-lease протокол. Статус меняется только после
        успешной отправки или планирования retry.
        """
        if limit <= 0:
            return []
        due_at = now or _now()
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM notification_outbox "
                "WHERE sent_at IS NULL AND next_attempt_at <= ? "
                "ORDER BY next_attempt_at, id LIMIT ?",
                (due_at, int(limit)),
            ).fetchall()

    def mark_notification_sent(self, notification_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM notification_outbox WHERE id = ?",
                (int(notification_id),),
            )
            self._conn.commit()

    def retry_notification(
        self, notification_id: int, next_attempt_at: str, error: str | None
    ) -> None:
        """Запланировать повтор. Ошибка ограничена, чтобы не раздувать SQLite."""
        with self._lock:
            self._conn.execute(
                "UPDATE notification_outbox "
                "SET attempts = attempts + 1, next_attempt_at = ?, last_error = ? "
                "WHERE id = ? AND sent_at IS NULL",
                (next_attempt_at, (error or "")[:500], int(notification_id)),
            )
            self._conn.commit()

    def prune_events(self, max_count: int | None = None, ttl_days: float | None = None) -> int:
        """Удалить старые события. Возвращает число удалённых строк."""
        deleted = 0
        with self._lock:
            # Старые версии оставляли доставленные outbox-строки с JPEG и
            # ссылкой на событие. Они не нужны после отправки и мешают prune.
            sent_cleanup = self._conn.execute(
                "DELETE FROM notification_outbox WHERE sent_at IS NOT NULL"
            ).rowcount or 0
            if ttl_days is not None and ttl_days > 0:
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(days=float(ttl_days))
                ).isoformat()
                cur = self._conn.execute(
                    "DELETE FROM events WHERE detected_at < ?", (cutoff,)
                )
                deleted += cur.rowcount or 0
            if max_count is not None and max_count > 0:
                # Avoid materialising every retained id (and SQLite's bind
                # limit) when the event archive is large.
                cutoff_row = self._conn.execute(
                    "SELECT id FROM events ORDER BY id DESC LIMIT 1 OFFSET ?",
                    (int(max_count) - 1,),
                ).fetchone()
                if cutoff_row is not None:
                    cur = self._conn.execute(
                        "DELETE FROM events WHERE id < ?", (int(cutoff_row["id"]),)
                    )
                    deleted += cur.rowcount or 0
            if deleted or sent_cleanup:
                self._conn.commit()
            if deleted:
                # Убирает освобождённые страницы из WAL без тяжёлого VACUUM в
                # горячем цикле детектора. Основной файл SQLite переиспользует
                # свободное место под следующие события.
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return deleted

    # -- settings -------------------------------------------------------

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    def get(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default
