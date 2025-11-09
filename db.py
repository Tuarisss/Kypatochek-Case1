"""SQLite persistence layer for users, interactions, and document usage."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

LOGGER = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat(sep=" ")


@dataclass
class BotUser:
    id: int
    telegram_id: int
    username: Optional[str]
    fio: Optional[str]
    profession: Optional[str]
    state: str
    first_seen: str
    last_active: Optional[str]
    consent_at: Optional[str]

    @property
    def is_active(self) -> bool:
        return self.state == "active"


@dataclass
class QuizSession:
    user_id: int
    question: str
    options: List[str]
    correct_index: int
    explanation: Optional[str]
    sources: List[str]
    questions_answered: int
    correct_answers: int


class BotDatabase:
    """Thread-safe SQLite helper."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON;")
        self._ensure_schema()
        self._apply_migrations()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _ensure_schema(self) -> None:
        LOGGER.info("Ensuring database schema at %s", self.db_path)
        schema = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER NOT NULL UNIQUE,
            username TEXT,
            fio TEXT,
            profession TEXT,
            state TEXT NOT NULL DEFAULT 'pending_consent',
            first_seen TEXT NOT NULL,
            last_active TEXT,
            last_state_change TEXT NOT NULL,
            consent_at TEXT
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            user_text TEXT,
            bot_text TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);

        CREATE TABLE IF NOT EXISTS document_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            doc_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_doc_usage_doc_path ON document_usage(doc_path);
        CREATE INDEX IF NOT EXISTS idx_doc_usage_user_id ON document_usage(user_id);

        CREATE TABLE IF NOT EXISTS quiz_sessions (
            user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            question TEXT NOT NULL,
            options TEXT NOT NULL,
            correct_index INTEGER NOT NULL,
            explanation TEXT,
            sources TEXT,
            created_at TEXT NOT NULL,
            questions_answered INTEGER NOT NULL DEFAULT 0,
            correct_answers INTEGER NOT NULL DEFAULT 0
        );
        """
        with self._lock:
            self._conn.executescript(schema)
            self._conn.commit()

    def _apply_migrations(self) -> None:
        self._ensure_user_columns()
        self._ensure_quiz_columns()

    def _ensure_user_columns(self) -> None:
        columns = self._get_table_columns("users")
        if "consent_at" not in columns:
            with self._lock:
                self._conn.execute("ALTER TABLE users ADD COLUMN consent_at TEXT")
                self._conn.commit()

    def _ensure_quiz_columns(self) -> None:
        columns = self._get_table_columns("quiz_sessions")
        if columns:
            if "questions_answered" not in columns:
                with self._lock:
                    self._conn.execute(
                        "ALTER TABLE quiz_sessions ADD COLUMN questions_answered INTEGER NOT NULL DEFAULT 0"
                    )
                    self._conn.commit()
            if "correct_answers" not in columns:
                with self._lock:
                    self._conn.execute(
                        "ALTER TABLE quiz_sessions ADD COLUMN correct_answers INTEGER NOT NULL DEFAULT 0"
                    )
                    self._conn.commit()

    def _get_table_columns(self, table_name: str) -> Set[str]:
        with self._lock:
            rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row["name"] for row in rows}

    def _row_to_user(self, row: sqlite3.Row) -> BotUser:
        return BotUser(
            id=row["id"],
            telegram_id=row["telegram_id"],
            username=row["username"],
            fio=row["fio"],
            profession=row["profession"],
            state=row["state"],
            first_seen=row["first_seen"],
            last_active=row["last_active"],
            consent_at=row["consent_at"],
        )

    def get_or_create_user(self, telegram_id: int, username: Optional[str] = None) -> BotUser:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)
            ).fetchone()
            if row:
                return self._row_to_user(row)
            now = _utcnow()
            self._conn.execute(
                """
                INSERT INTO users (telegram_id, username, state, first_seen, last_active, last_state_change)
                VALUES (?, ?, 'pending_consent', ?, ?, ?)
                """,
                (telegram_id, username, now, now, now),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)
            ).fetchone()
            return self._row_to_user(row)

    def update_user_profile(
        self,
        user_id: int,
        *,
        fio: Optional[str] = None,
        profession: Optional[str] = None,
    ) -> None:
        fields = []
        params: List[str] = []
        if fio is not None:
            fields.append("fio = ?")
            params.append(fio.strip())
        if profession is not None:
            fields.append("profession = ?")
            params.append(profession.strip())
        if not fields:
            return
        params.append(user_id)
        query = f"UPDATE users SET {', '.join(fields)} WHERE id = ?"
        with self._lock:
            self._conn.execute(query, params)
            self._conn.commit()

    def update_user_state(self, user_id: int, new_state: str) -> None:
        now = _utcnow()
        with self._lock:
            self._conn.execute(
                "UPDATE users SET state = ?, last_state_change = ?, last_active = ? WHERE id = ?",
                (new_state, now, now, user_id),
            )
            self._conn.commit()

    def mark_user_consent(self, user_id: int) -> None:
        now = _utcnow()
        with self._lock:
            self._conn.execute(
                """
                UPDATE users
                SET consent_at = COALESCE(consent_at, ?),
                    last_active = ?
                WHERE id = ?
                """,
                (now, now, user_id),
            )
            self._conn.commit()


    def update_last_active(self, user_id: int) -> None:
        now = _utcnow()
        with self._lock:
            self._conn.execute(
                "UPDATE users SET last_active = ? WHERE id = ?",
                (now, user_id),
            )
            self._conn.commit()

    def log_interaction(self, user_id: int, user_text: str, bot_text: str) -> None:
        now = _utcnow()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO interactions (user_id, user_text, bot_text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, user_text, bot_text, now),
            )
            self._conn.commit()

    def log_document_usage(self, user_id: int, doc_path: Path) -> None:
        now = _utcnow()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO document_usage (user_id, doc_path, created_at)
                VALUES (?, ?, ?)
                """,
                (user_id, str(doc_path), now),
            )
            self._conn.commit()

    def set_quiz_session(
        self,
        user_id: int,
        question: str,
        options: List[str],
        correct_index: int,
        explanation: Optional[str],
        sources: List[str],
        *,
        questions_answered: int = 0,
        correct_answers: int = 0,
    ) -> None:
        now = _utcnow()
        payload = json.dumps(options, ensure_ascii=False)
        sources_payload = json.dumps(sources, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO quiz_sessions (user_id, question, options, correct_index, explanation, sources, created_at, questions_answered, correct_answers)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    question = excluded.question,
                    options = excluded.options,
                    correct_index = excluded.correct_index,
                    explanation = excluded.explanation,
                    sources = excluded.sources,
                    created_at = excluded.created_at,
                    questions_answered = excluded.questions_answered,
                    correct_answers = excluded.correct_answers
                """,
                (
                    user_id,
                    question,
                    payload,
                    correct_index,
                    explanation,
                    sources_payload,
                    now,
                    questions_answered,
                    correct_answers,
                ),
            )
            self._conn.commit()

    def get_quiz_session(self, user_id: int) -> Optional[QuizSession]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT user_id, question, options, correct_index, explanation, sources
                FROM quiz_sessions
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if not row:
            return None
        try:
            answered = row["questions_answered"]
            correct = row["correct_answers"]
        except Exception:
            answered = 0
            correct = 0
        return QuizSession(
            user_id=row["user_id"],
            question=row["question"],
            options=json.loads(row["options"]),
            correct_index=row["correct_index"],
            explanation=row["explanation"],
            sources=json.loads(row["sources"]) if row["sources"] else [],
            questions_answered=answered,
            correct_answers=correct,
        )

    def clear_quiz_session(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM quiz_sessions WHERE user_id = ?", (user_id,))
            self._conn.commit()

    def update_quiz_stats(
        self, user_id: int, *, answered_delta: int, correct_delta: int
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE quiz_sessions
                SET questions_answered = questions_answered + ?,
                    correct_answers = correct_answers + ?
                WHERE user_id = ?
                """,
                (answered_delta, correct_delta, user_id),
            )
            self._conn.commit()

    def get_stats(
        self,
        *,
        limit_docs: int = 5,
        limit_recent_docs: int = 5,
        limit_users: int = 5,
    ) -> Dict[str, object]:
        with self._lock:
            total_users = self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            active_users = self._conn.execute(
                "SELECT COUNT(*) FROM users WHERE state = 'active'"
            ).fetchone()[0]
            pending_users = total_users - active_users
            total_interactions = self._conn.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()[0]
            top_docs = self._conn.execute(
                """
                SELECT doc_path, COUNT(*) as cnt
                FROM document_usage
                GROUP BY doc_path
                ORDER BY cnt DESC
                LIMIT ?
                """,
                (limit_docs,),
            ).fetchall()
            recent_doc_events = self._conn.execute(
                """
                SELECT doc_path, created_at, u.fio, u.profession, u.telegram_id
                FROM document_usage du
                JOIN users u ON u.id = du.user_id
                ORDER BY du.created_at DESC
                LIMIT ?
                """,
                (limit_recent_docs,),
            ).fetchall()
            user_rows = self._conn.execute(
                """
                SELECT fio, profession, first_seen, last_active, telegram_id, state
                FROM users
                ORDER BY last_active DESC
                LIMIT ?
                """,
                (limit_users,),
            ).fetchall()

        def _format_duration(first_seen: str, last_active: Optional[str]) -> str:
            if not first_seen or not last_active:
                return "n/a"
            start = datetime.fromisoformat(first_seen)
            end = datetime.fromisoformat(last_active)
            delta = end - start
            hours = int(delta.total_seconds() // 3600)
            minutes = int((delta.total_seconds() % 3600) // 60)
            if hours > 0:
                return f"{hours}ч {minutes}м"
            return f"{minutes}м"

        user_summaries = []
        for row in user_rows:
            user_summaries.append(
                {
                    "fio": row["fio"] or "Не указано",
                    "profession": row["profession"] or "Не указано",
                    "telegram_id": row["telegram_id"],
                    "state": row["state"],
                    "first_seen": row["first_seen"],
                    "last_active": row["last_active"],
                    "duration": _format_duration(row["first_seen"], row["last_active"]),
                }
            )

        top_docs_formatted = [
            {"doc_path": row["doc_path"], "count": row["cnt"]} for row in top_docs
        ]
        recent_docs_formatted = [
            {
                "doc_path": row["doc_path"],
                "created_at": row["created_at"],
                "fio": row["fio"] or "Не указано",
                "profession": row["profession"] or "",
                "telegram_id": row["telegram_id"],
            }
            for row in recent_doc_events
        ]
        return {
            "total_users": total_users,
            "active_users": active_users,
            "pending_users": pending_users,
            "total_interactions": total_interactions,
            "top_docs": top_docs_formatted,
            "recent_doc_events": recent_docs_formatted,
            "user_summaries": user_summaries,
        }
