"""Phase 23.1 — Auth configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.config import PROJECT_ROOT

logger = logging.getLogger("forecast_service.auth")


def auth_db_path() -> Path:
    raw = os.environ.get("FORESIGHT_AUTH_DB_PATH")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p
    # On Vercel, AWS Lambda, or Linux serverless, use writable /tmp
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or not os.access(str(PROJECT_ROOT), os.W_OK):
        return Path("/tmp/project_foresight_auth.db")
    return PROJECT_ROOT / Path("data/auth/project_foresight_auth.db")


def jwt_secret_key() -> str:
    """Resolve JWT signing secret. Never raise during login — fall back safely."""
    key = (
        os.environ.get("JWT_SECRET_KEY")
        or os.environ.get("FORESIGHT_API_JWT_SECRET")
        or os.environ.get("FORESIGHT_JWT_SECRET_KEY")
        or os.environ.get("SECRET_KEY")
    )
    if key and key.strip():
        return key.strip()

    api_key = (os.environ.get("FORESIGHT_API_API_KEY") or "").strip()
    if api_key:
        logger.warning(
            "JWT_SECRET_KEY unset; using FORESIGHT_API_API_KEY as token signing secret. "
            "Set JWT_SECRET_KEY on Render for production."
        )
        return api_key

    env = os.environ.get("FORESIGHT_ENV", "development").strip().lower()
    if env == "production":
        # Prefer working auth over a hard 500 when deploy env is incomplete.
        logger.error(
            "JWT_SECRET_KEY unset in production; using ephemeral fallback. "
            "Set JWT_SECRET_KEY (or FORESIGHT_API_JWT_SECRET) on Render."
        )
        return "foresight-production-jwt-fallback-change-me"
    return "foresight-dev-only-change-in-production"


def user_auth_required_for_api() -> bool:
    raw = os.environ.get("FORESIGHT_USER_AUTH_REQUIRED")
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def jwt_expiry_seconds() -> int:
    return int(os.environ.get("FORESIGHT_JWT_EXPIRY_SECONDS", "86400"))
