"""Phase 23.1 — Auth configuration."""

from __future__ import annotations

import os
from pathlib import Path

from src.config import PROJECT_ROOT


def auth_db_path() -> Path:
    raw = os.environ.get("FORESIGHT_AUTH_DB_PATH", "data/auth/project_foresight_auth.db")
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def jwt_secret_key() -> str:
    key = os.environ.get("JWT_SECRET_KEY") or os.environ.get("SECRET_KEY")
    if not key:
        if os.environ.get("FORESIGHT_ENV", "development") == "production":
            raise RuntimeError("JWT_SECRET_KEY or SECRET_KEY must be set in production")
        key = "foresight-dev-only-change-in-production"
    return key


def user_auth_required_for_api() -> bool:
    raw = os.environ.get("FORESIGHT_USER_AUTH_REQUIRED")
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def jwt_expiry_seconds() -> int:
    return int(os.environ.get("FORESIGHT_JWT_EXPIRY_SECONDS", "86400"))
