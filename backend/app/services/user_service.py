from __future__ import annotations

import re
import secrets
import sqlite3
import hashlib

import bcrypt
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.app.config import settings


DISPOSABLE_EMAIL_DOMAINS = {
    "mailinator.com",
    "10minutemail.com",
    "guerrillamail.com",
    "tempmail.com",
    "trashmail.com",
    "yopmail.com",
    "dispostable.com",
}


def normalize_email(email: str) -> str:
    return email.strip().lower()


def validate_email_policy(email: str) -> None:
    normalized = normalize_email(email)
    domain = normalized.split("@")[-1]
    if domain in DISPOSABLE_EMAIL_DOMAINS:
        raise ValueError("Disposable email domains are not allowed.")


def validate_password_strength(password: str) -> None:
    """Raise ``ValueError`` if *password* does not meet minimum complexity."""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain at least one uppercase letter.")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain at least one lowercase letter.")
    if not re.search(r"\d", password):
        raise ValueError("Password must contain at least one digit.")


def generate_verification_token() -> str:
    return secrets.token_urlsafe(48)


class UserService:
    """CRUD operations for user accounts (SQLite)."""

    def __init__(self) -> None:
        self._db_path = Path(settings.history_db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    email       TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    full_name   TEXT NOT NULL,
                    role        TEXT NOT NULL DEFAULT 'clinician',
                    created_at_utc TEXT NOT NULL,
                    is_active   INTEGER NOT NULL DEFAULT 1,
                    is_verified INTEGER NOT NULL DEFAULT 0,
                    verification_token TEXT,
                    verification_token_expires TEXT
                )
                """
            )
            self._migrate_add_verification_columns(conn)

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS revoked_tokens (
                    token_hash TEXT PRIMARY KEY,
                    revoked_at_utc TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS refresh_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at_utc TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0,
                    created_at_utc TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    def _migrate_add_verification_columns(self, conn: sqlite3.Connection) -> None:
        """Add verification columns to existing tables that lack them."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "is_verified" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER NOT NULL DEFAULT 0")
        if "verification_token" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN verification_token TEXT")
        if "verification_token_expires" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN verification_token_expires TEXT")

    # ---- CRUD ----

    def create(
        self,
        email: str,
        hashed_password: str,
        full_name: str,
        role: str = "clinician",
        verification_token: str | None = None,
        verification_token_expires: str | None = None,
    ) -> dict:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (email, hashed_password, full_name, role,
                                   created_at_utc, is_verified,
                                   verification_token, verification_token_expires)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (normalize_email(email), hashed_password, full_name, role, now,
                 verification_token, verification_token_expires),
            )
            conn.commit()
            return self.get_by_id(cursor.lastrowid)  # type: ignore[arg-type]

    def get_by_email(self, email: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (normalize_email(email),)).fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_id(self, user_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_verification_token(self, token: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE verification_token = ?", (token,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def mark_verified(self, user_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET is_verified = 1, verification_token = NULL, "
                "verification_token_expires = NULL WHERE id = ?",
                (user_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def set_verification_token(self, user_id: int, token: str, expires: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET verification_token = ?, verification_token_expires = ? WHERE id = ?",
                (token, expires, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0


    def update_profile(self, user_id: int, full_name: str) -> dict | None:
        with self._connect() as conn:
            conn.execute("UPDATE users SET full_name = ? WHERE id = ?", (full_name, user_id))
            conn.commit()
        return self.get_by_id(user_id)

    def update_password(self, user_id: int, new_hashed_password: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET hashed_password = ? WHERE id = ?",
                (new_hashed_password, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def deactivate_release_email(self, user_id: int) -> bool:
        """Deactivate user, revoke sessions, scramble password, and free the email for re-registration."""
        placeholder = f"deactivated.{secrets.token_hex(16)}@inactive.local"
        junk_hash = bcrypt.hashpw(secrets.token_urlsafe(24).encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        with self._connect() as conn:
            conn.execute("UPDATE refresh_sessions SET revoked = 1 WHERE user_id = ?", (user_id,))
            cursor = conn.execute(
                """
                UPDATE users SET
                    email = ?,
                    hashed_password = ?,
                    is_active = 0,
                    is_verified = 0,
                    verification_token = NULL,
                    verification_token_expires = NULL
                WHERE id = ?
                """,
                (placeholder, junk_hash, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_active_refresh_sessions(self, user_id: int) -> list[dict]:
        """Non-revoked, unexpired sessions for *user_id* (includes token_hash for server-side use only)."""
        now = datetime.now(tz=timezone.utc)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT token_hash, created_at_utc, expires_at_utc, revoked
                FROM refresh_sessions WHERE user_id = ?
                """,
                (user_id,),
            ).fetchall()

        out: list[dict] = []
        for row in rows:
            if int(row["revoked"]) == 1:
                continue
            try:
                exp = datetime.fromisoformat(str(row["expires_at_utc"]))
            except ValueError:
                continue
            if now > exp:
                continue
            out.append(
                {
                    "token_hash": str(row["token_hash"]),
                    "created_at_utc": str(row["created_at_utc"]),
                    "expires_at_utc": str(row["expires_at_utc"]),
                }
            )
        return sorted(out, key=lambda r: r["created_at_utc"], reverse=True)

    def revoke_all_refresh_sessions_for_user(self, user_id: int) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE refresh_sessions SET revoked = 1 WHERE user_id = ? AND revoked = 0",
                (user_id,),
            )
            conn.commit()
            return int(cursor.rowcount or 0)

    def store_refresh_session(self, user_id: int, refresh_token: str, expires_at_utc: str) -> None:
        token_hash = hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO refresh_sessions (token_hash, user_id, expires_at_utc, revoked, created_at_utc) VALUES (?, ?, ?, 0, ?)",
                (token_hash, user_id, expires_at_utc, now),
            )
            conn.commit()

    def is_refresh_session_active(self, refresh_token: str) -> bool:
        token_hash = hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT expires_at_utc, revoked FROM refresh_sessions WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
        if row is None or int(row["revoked"]) == 1:
            return False
        try:
            return datetime.now(tz=timezone.utc) <= datetime.fromisoformat(str(row["expires_at_utc"]))
        except ValueError:
            return False

    def revoke_refresh_session(self, refresh_token: str) -> None:
        token_hash = hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute("UPDATE refresh_sessions SET revoked = 1 WHERE token_hash = ?", (token_hash,))
            conn.commit()

    def revoke_token(self, token: str) -> None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO revoked_tokens (token_hash, revoked_at_utc) VALUES (?, ?)",
                (token_hash, now),
            )
            conn.commit()

    def is_token_revoked(self, token: str) -> bool:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            row = conn.execute("SELECT token_hash FROM revoked_tokens WHERE token_hash = ?", (token_hash,)).fetchone()
        return row is not None

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": int(row["id"]),
            "email": str(row["email"]),
            "hashed_password": str(row["hashed_password"]),
            "full_name": str(row["full_name"]),
            "role": str(row["role"]),
            "created_at_utc": str(row["created_at_utc"]),
            "is_active": bool(row["is_active"]),
            "is_verified": bool(row["is_verified"]),
            "verification_token": row["verification_token"],
            "verification_token_expires": row["verification_token_expires"],
        }


user_service = UserService()
