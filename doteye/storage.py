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
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

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

    def prune_events(self, max_count: int | None = None, ttl_days: float | None = None) -> int:
        """Удалить старые события. Возвращает число удалённых строк."""
        deleted = 0
        with self._lock:
            if ttl_days is not None and ttl_days > 0:
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(days=float(ttl_days))
                ).isoformat()
                cur = self._conn.execute(
                    "DELETE FROM events WHERE detected_at < ?", (cutoff,)
                )
                deleted += cur.rowcount or 0
            if max_count is not None and max_count > 0:
                keep = [
                    int(row["id"])
                    for row in self._conn.execute(
                        "SELECT id FROM events ORDER BY id DESC LIMIT ?",
                        (int(max_count),),
                    ).fetchall()
                ]
                if keep:
                    placeholders = ",".join("?" * len(keep))
                    cur = self._conn.execute(
                        f"DELETE FROM events WHERE id NOT IN ({placeholders})",
                        keep,
                    )
                    deleted += cur.rowcount or 0
                else:
                    cur = self._conn.execute("DELETE FROM events")
                    deleted += cur.rowcount or 0
            if deleted:
                self._conn.commit()
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
