from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterator


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = Lock()
        self._ensure_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connection() as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT,
                    consent INTEGER DEFAULT 0,
                    consent_timestamp TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    query TEXT,
                    response TEXT,
                    created_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    score INTEGER,
                    created_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )

    def upsert_user(self, user_id: int, username: str | None) -> None:
        username = username or "anonymous"
        with self._connection() as conn, conn:
            conn.execute(
                """
                INSERT INTO users (id, username)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET username=excluded.username
                """,
                (user_id, username),
            )

    def record_consent(self, user_id: int) -> None:
        timestamp = datetime.utcnow().isoformat()
        with self._connection() as conn, conn:
            conn.execute(
                "UPDATE users SET consent=1, consent_timestamp=? WHERE id=?",
                (timestamp, user_id),
            )

    def has_consent(self, user_id: int) -> bool:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT consent FROM users WHERE id=?",
                (user_id,),
            ).fetchone()
        return bool(row and row[0])

    def log_interaction(self, user_id: int, query: str, response: str) -> None:
        timestamp = datetime.utcnow().isoformat()
        with self._connection() as conn, conn:
            conn.execute(
                "INSERT INTO logs (user_id, query, response, created_at) VALUES (?, ?, ?, ?)",
                (user_id, query, response, timestamp),
            )

    def record_test_score(self, user_id: int, score: int) -> None:
        timestamp = datetime.utcnow().isoformat()
        with self._connection() as conn, conn:
            conn.execute(
                "INSERT INTO tests (user_id, score, created_at) VALUES (?, ?, ?)",
                (user_id, score, timestamp),
            )

    def get_stats(self) -> dict[str, int]:
        with self._connection() as conn:
            users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            consensual = conn.execute(
                "SELECT COUNT(*) FROM users WHERE consent=1"
            ).fetchone()[0]
            logs = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
            tests = conn.execute("SELECT COUNT(*) FROM tests").fetchone()[0]
        return {
            "users": users,
            "consents": consensual,
            "logs": logs,
            "tests": tests,
        }

    def export_logs(self) -> list[tuple[int, str, str, str]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT user_id, query, response, created_at FROM logs ORDER BY created_at DESC"
            ).fetchall()
        return rows

    def delete_user(self, user_id: int) -> None:
        with self._connection() as conn, conn:
            conn.execute("DELETE FROM logs WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM tests WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM users WHERE id=?", (user_id,))
