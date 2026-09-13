"""Тесты Storage: люди, события, настройки, миграция схемы."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

from doteye.storage import Storage


def make_storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "test.db")


def test_settings_roundtrip(tmp_path: Path) -> None:
    st = make_storage(tmp_path)
    assert st.get("missing") is None
    assert st.get("missing", "fallback") == "fallback"
    st.set("detect_mode", "identity")
    st.set("detect_mode", "presence")
    assert st.get("detect_mode") == "presence"
    st.close()


def test_people_crud(tmp_path: Path) -> None:
    st = make_storage(tmp_path)
    st.upsert_person("alice", b"emb")
    assert st.get_person("alice")["name"] == "alice"

    st.upsert_person("alice", b"emb2")
    assert st.get_person("alice")["embedding"] == b"emb2"

    assert len(st.list_people()) == 1
    assert st.list_people()[0]["has_face"] == 1
    assert st.list_people()[0]["face_count"] == 2
    people = st.list_people_with_embeddings()
    assert len(people) == 1
    assert len(people[0]["embeddings"]) == 2

    assert st.delete_person("alice") is True
    assert st.delete_person("alice") is False
    st.close()


def test_events_join_person(tmp_path: Path) -> None:
    st = make_storage(tmp_path)
    pid = st.upsert_person("bob", None)
    st.add_event(pid, b"frame", 0.9)
    st.add_event(None, b"frame2", None)

    rows = st.recent_events()
    assert len(rows) == 2
    assert rows[0]["person_name"] is None
    assert rows[1]["person_name"] == "bob"
    assert rows[1]["confidence"] == 0.9
    st.close()


def test_encrypted_audit_entries_are_retained_without_plaintext(tmp_path: Path) -> None:
    st = Storage(tmp_path / "audit.db")
    st.add_audit_entry(b"encrypted-action", keep=2)
    st.add_audit_entry(b"encrypted-action-2", keep=2)
    st.add_audit_entry(b"encrypted-action-3", keep=2)
    rows = st.recent_audit_entries()
    assert len(rows) == 2
    assert rows[0]["payload"] == b"encrypted-action-3"
    st.close()


def test_migration_adds_confidence(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            frame BLOB,
            detected_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    st = Storage(db)
    st.add_event(None, b"x", 0.5)
    assert st.recent_events()[0]["confidence"] == 0.5
    st.close()


def test_prune_and_pagination(tmp_path: Path) -> None:
    st = make_storage(tmp_path)
    for i in range(5):
        st.add_event(None, f"f{i}".encode(), event_type="enter")
    assert st.count_events() == 5
    page = st.recent_events(limit=2, offset=0)
    assert len(page) == 2
    deleted = st.prune_events(max_count=3, ttl_days=None)
    assert deleted >= 2
    assert st.count_events() == 3
    st.close()


def test_wal_enabled(tmp_path: Path) -> None:
    st = make_storage(tmp_path)
    mode = st._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    st.close()


def test_notification_outbox_is_durable_and_idempotent(tmp_path: Path) -> None:
    st = make_storage(tmp_path)
    event_id = st.add_event(None, b"frame")
    first = st.enqueue_notification("event:1:42", event_id, 42, "alert", b"jpeg")
    assert st.enqueue_notification("event:1:42", event_id, 42, "alert", b"jpeg") == first
    due = st.claim_due_notifications(limit=10)
    assert len(due) == 1
    assert due[0]["event_id"] == event_id
    assert due[0]["jpeg"] == b"jpeg"

    later = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    st.retry_notification(first, later, "temporary error")
    assert st.claim_due_notifications(limit=10) == []
    due = st.claim_due_notifications(now=(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat())
    assert due[0]["attempts"] == 1
    assert due[0]["last_error"] == "temporary error"
    st.mark_notification_sent(first)
    assert st.claim_due_notifications(now=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat()) == []
    assert st._conn.execute("SELECT COUNT(*) FROM notification_outbox").fetchone()[0] == 0
    st.close()
