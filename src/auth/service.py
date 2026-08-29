"""Phase 23.1 — Registration and login service."""

from __future__ import annotations

from pathlib import Path

from src.auth.database import get_connection, init_db, utc_now_iso
from src.auth.models import User
from src.auth.security import (
    create_access_token,
    hash_password,
    validate_password_policy,
    verify_password,
)


class AuthError(Exception):
    pass


class AuthService:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path
        init_db(db_path)

    def _row_to_user(self, row) -> User:
        return User(
            id=row["id"],
            full_name=row["full_name"],
            email=row["email"],
            is_active=bool(row["is_active"]),
            role=row["role"],
            created_at=row["created_at"],
            last_login=row["last_login"],
        )

    def register(self, full_name: str, email: str, password: str, confirm_password: str) -> User:
        if not full_name.strip():
            raise AuthError("Name is required")
        if password != confirm_password:
            raise AuthError("Passwords do not match")
        validate_password_policy(password)
        email_norm = email.strip().lower()
        password_hash = hash_password(password)
        now = utc_now_iso()
        try:
            with get_connection(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO users (full_name, email, password_hash, is_active, role, created_at)
                    VALUES (?, ?, ?, 1, 'USER', ?)
                    """,
                    (full_name.strip(), email_norm, password_hash, now),
                )
                row = conn.execute(
                    "SELECT * FROM users WHERE email = ?", (email_norm,)
                ).fetchone()
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                raise AuthError("Email already registered") from exc
            raise
        return self._row_to_user(row)

    def login(self, email: str, password: str) -> tuple[User, str]:
        email_norm = email.strip().lower()
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email_norm,)).fetchone()
            if row is None:
                raise AuthError("Invalid email or password")
            if not bool(row["is_active"]):
                raise AuthError("Account is inactive")
            if not verify_password(password, row["password_hash"]):
                raise AuthError("Invalid email or password")
            now = utc_now_iso()
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, row["id"]))
            row = conn.execute("SELECT * FROM users WHERE id = ?", (row["id"],)).fetchone()
        user = self._row_to_user(row)
        # itsdangerous JSON serializer is happiest with string subject ids
        token = create_access_token(
            {"sub": str(user.id), "email": user.email, "role": user.role}
        )
        return user, token

    def get_user_by_id(self, user_id: int) -> User | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._row_to_user(row) if row else None


_service: AuthService | None = None


def get_auth_service() -> AuthService:
    global _service
    if _service is None:
        _service = AuthService()
    return _service
