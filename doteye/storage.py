"""Слой хранения (SQLite): известные люди, события входа, настройки.

Паттерн Data Access Object без ORM - обычный sqlite3 stdlib.
Соединение открыто с check_same_thread=False, потому что пайплайн работает
в отдельном потоке (asyncio.to_thread), а бот - в event loop; доступ
сериализуется через threading.Lock.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    embedding BLOB,          -- зашифрованный face embedding (identity mode)
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,       -- NULL = неизвестный (presence mode без распознавания)
    frame BLOB,              -- зашифрованный кадр (JPEG)
    confidence REAL,         -- уверенность распознавания (identity mode)
    detected_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES people (id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


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
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Добавить колонки, которых нет в БД со старой схемой."""
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(events)")}
        if "confidence" not in cols:
            self._conn.execute("ALTER TABLE events ADD COLUMN confidence REAL")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- people ---------------------------------------------------------

    def add_person(self, name: str, embedding: bytes | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO people (name, embedding, created_at) VALUES (?, ?, ?)",
                (name, embedding, _now()),
            )
            self._conn.commit()
            return cur.lastrowid

    def upsert_person(self, name: str, embedding: bytes | None) -> int:
        """Добавить человека или обновить его embedding по имени."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM people WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                cur = self._conn.execute(
                    "INSERT INTO people (name, embedding, created_at) VALUES (?, ?, ?)",
                    (name, embedding, _now()),
                )
                self._conn.commit()
                return cur.lastrowid
            self._conn.execute(
                "UPDATE people SET embedding = ? WHERE id = ?", (embedding, row["id"])
            )
            self._conn.commit()
            return row["id"]

    def get_person(self, name: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM people WHERE name = ?", (name,)
            ).fetchone()

    def list_people(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT id, name, created_at, embedding IS NOT NULL AS has_face "
                "FROM people ORDER BY name"
            ).fetchall()

    def list_people_with_embeddings(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT id, name, embedding FROM people WHERE embedding IS NOT NULL"
            ).fetchall()

    def delete_person(self, name: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM people WHERE name = ?", (name,))
            self._conn.commit()
            return cur.rowcount > 0

    # -- events ---------------------------------------------------------

    def add_event(
        self,
        person_id: int | None,
        frame: bytes | None,
        confidence: float | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (person_id, frame, confidence, detected_at) "
                "VALUES (?, ?, ?, ?)",
                (person_id, frame, confidence, _now()),
            )
            self._conn.commit()
            return cur.lastrowid

    def recent_events(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT e.*, p.name AS person_name FROM events e "
                "LEFT JOIN people p ON p.id = e.person_id "
                "ORDER BY e.id DESC LIMIT ?",
                (limit,),
            ).fetchall()

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
