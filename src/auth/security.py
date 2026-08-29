"""Phase 23.1 — Password hashing and JWT tokens."""

from __future__ import annotations

import hashlib
import re
import secrets
from typing import Any

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from src.auth.config import jwt_expiry_seconds, jwt_secret_key

PASSWORD_MIN_LENGTH = 8
PASSWORD_POLICY_MSG = (
    "Password must be at least 8 characters and include at least one letter and one number."
)


def validate_password_policy(password: str) -> None:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(PASSWORD_POLICY_MSG)
    if not re.search(r"[A-Za-z]", password):
        raise ValueError(PASSWORD_POLICY_MSG)
    if not re.search(r"\d", password):
        raise ValueError(PASSWORD_POLICY_MSG)


def hash_password(password: str) -> str:
    validate_password_policy(password)
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 260_000
    )
    return f"pbkdf2_sha256$260000${salt}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algo, iterations, salt, digest_hex = password_hash.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        expected = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iterations),
        ).hex()
        return secrets.compare_digest(expected, digest_hex)
    except (ValueError, TypeError):
        return False


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(jwt_secret_key(), salt="project-foresight-auth")


def create_access_token(payload: dict[str, Any]) -> str:
    return _serializer().dumps(payload)


def decode_access_token(token: str) -> dict[str, Any]:
    return _serializer().loads(token, max_age=jwt_expiry_seconds())


def safe_decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        return decode_access_token(token)
    except (BadSignature, SignatureExpired):
        return None
