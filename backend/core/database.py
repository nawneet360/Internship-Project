"""
core/database.py
----------------
Thin SQLite wrapper using Python's built-in sqlite3.
No ORM — just parameterised queries.

Tables
------
  users
    id            TEXT PRIMARY KEY  (UUID)
    username      TEXT UNIQUE NOT NULL
    email         TEXT UNIQUE NOT NULL
    password_hash TEXT NOT NULL     (PBKDF2-HMAC-SHA256)
    created_at    TEXT NOT NULL     (ISO-8601)
    is_active     INTEGER DEFAULT 1

  refresh_tokens
    token         TEXT PRIMARY KEY
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE
    expires_at    TEXT NOT NULL
    revoked       INTEGER DEFAULT 0

  predictions
    id            TEXT PRIMARY KEY
    user_id       TEXT REFERENCES users(id) ON DELETE SET NULL
    disaster_type TEXT NOT NULL
    risk_level    TEXT NOT NULL
    risk_score    REAL NOT NULL
    location      TEXT
    inputs_json   TEXT NOT NULL
    result_json   TEXT NOT NULL
    created_at    TEXT NOT NULL
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("disasterai.db")

_DB_PATH: Path | None = None


def init_db(db_path: Path) -> None:
    """Create tables if they don't exist. Call once at app startup."""
    global _DB_PATH
    _DB_PATH = Path(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                username      TEXT UNIQUE NOT NULL,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                is_active     INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS refresh_tokens (
                token      TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL,
                revoked    INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id            TEXT PRIMARY KEY,
                user_id       TEXT REFERENCES users(id) ON DELETE SET NULL,
                disaster_type TEXT NOT NULL,
                risk_level    TEXT NOT NULL,
                risk_score    REAL NOT NULL,
                location      TEXT,
                inputs_json   TEXT NOT NULL,
                result_json   TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(disaster_type);
        """)
    log.info("Database initialized at: %s", _DB_PATH)


@contextmanager
def _connect():
    """Yield a connection with foreign keys on and auto-commit."""
    if _DB_PATH is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Users ──────────────────────────────────────────────────────────────────────

def create_user(user_id: str, username: str, email: str,
                password_hash: str, created_at: str | None = None) -> dict:
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (id, username, email, password_hash, created_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (user_id, username.lower().strip(), email.lower().strip(), password_hash, created_at),
        )
    return {
        "id": user_id,
        "username": username.lower().strip(),
        "email": email.lower().strip(),
        "created_at": created_at,
    }


def get_user_by_email(email: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ? AND is_active = 1",
            (email.lower().strip(),),
        ).fetchone()


def get_user_by_username(username: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ? AND is_active = 1",
            (username.lower().strip(),),
        ).fetchone()


def get_user_by_identifier(identifier: str) -> sqlite3.Row | None:
    """Find user by either username or email."""
    val = identifier.lower().strip()
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE (username = ? OR email = ?) AND is_active = 1",
            (val, val),
        ).fetchone()


def get_user_by_id(user_id: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE id = ? AND is_active = 1",
            (user_id,),
        ).fetchone()


def email_exists(email: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
        return row is not None


def username_exists(username: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username.lower().strip(),)
        ).fetchone()
        return row is not None


# ── Refresh tokens ─────────────────────────────────────────────────────────────

def save_refresh_token(token: str, user_id: str, expires_at: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO refresh_tokens (token, user_id, expires_at, revoked) VALUES (?, ?, ?, 0)",
            (token, user_id, expires_at),
        )


def get_refresh_token(token: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM refresh_tokens WHERE token = ? AND revoked = 0",
            (token,),
        ).fetchone()


def revoke_refresh_token(token: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE refresh_tokens SET revoked = 1 WHERE token = ?", (token,)
        )


def revoke_all_user_tokens(user_id: str) -> None:
    """Log out user from all devices."""
    with _connect() as conn:
        conn.execute(
            "UPDATE refresh_tokens SET revoked = 1 WHERE user_id = ?", (user_id,)
        )


# ── Prediction Storage ─────────────────────────────────────────────────────────

def save_prediction(
    pred_id: str,
    user_id: str | None,
    disaster_type: str,
    risk_level: str,
    risk_score: float,
    location: str | None,
    inputs: dict,
    result: dict,
    created_at: str | None = None,
) -> None:
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO predictions (id, user_id, disaster_type, risk_level, risk_score, location, inputs_json, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pred_id,
                user_id,
                disaster_type,
                risk_level,
                float(risk_score),
                location,
                json.dumps(inputs),
                json.dumps(result),
                created_at,
            ),
        )


def get_predictions(
    user_id: str | None = None,
    disaster_type: str | None = None,
    limit: int = 20,
) -> list[dict]:
    query = "SELECT * FROM predictions WHERE 1=1"
    params: list = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    if disaster_type:
        query += " AND disaster_type = ?"
        params.append(disaster_type)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r["id"],
                "user_id": r["user_id"],
                "type": r["disaster_type"],
                "risk_level": r["risk_level"],
                "risk_score": r["risk_score"],
                "location": r["location"],
                "inputs": json.loads(r["inputs_json"]),
                "result": json.loads(r["result_json"]),
                "created_at": r["created_at"],
            })
        return result
