"""Phase 23.1 — FastAPI auth dependencies."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.config import user_auth_required_for_api
from src.auth.models import User
from src.auth.security import safe_decode_access_token
from src.auth.service import get_auth_service

_bearer = HTTPBearer(auto_error=False)


def _extract_bearer_token(
    credentials: HTTPAuthorizationCredentials | None,
) -> str | None:
    if credentials and credentials.scheme.lower() == "bearer":
        return credentials.credentials
    return None


async def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User | None:
    token = _extract_bearer_token(credentials)
    if not token:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
    if not token:
        return None
    payload = safe_decode_access_token(token)
    if not payload:
        return None
    user = get_auth_service().get_user_by_id(int(payload["sub"]))
    if not user or not user.is_active:
        return None
    return user


async def require_user_auth(
    user: User | None = Depends(get_current_user_optional),
) -> User | None:
    if not user_auth_required_for_api():
        return user
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


async def require_admin(
    user: User | None = Depends(get_current_user_optional),
) -> User | None:
    if not user_auth_required_for_api():
        return user
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Forbidden")
    return user
