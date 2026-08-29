"""Configurable API-key authentication. Secrets are never hardcoded."""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Request

from src.config import api_api_key, api_auth_enabled, foresight_env

PUBLIC_PATHS = frozenset({"/health", "/ready", "/auth/register", "/auth/login", "/auth/logout"})


def auth_is_required() -> bool:
    """Production always requires auth. Development may bypass when disabled."""
    env = foresight_env()
    if env == "production":
        return True
    return api_auth_enabled()


def extract_api_key(request: Request) -> Optional[str]:
    header = request.headers.get("x-api-key")
    if header and header.strip():
        return header.strip()
    auth = request.headers.get("authorization")
    if not auth:
        return None
    scheme, _, remainder = auth.partition(" ")
    if scheme.lower() == "bearer" and remainder.strip():
        return remainder.strip()
    return None


def key_matches(provided: str | None, expected: str | None = None) -> bool:
    want = (expected if expected is not None else api_api_key()) or ""
    got = provided or ""
    if not want or not got:
        return False
    return hmac.compare_digest(got.encode("utf-8"), want.encode("utf-8"))


def is_public_path(path: str) -> bool:
    normalized = path.rstrip("/") or "/"
    if normalized in PUBLIC_PATHS:
        return True
    return path in PUBLIC_PATHS
