"""Phase 23.1 — Authentication API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.auth.dependencies import get_current_user_optional
from src.auth.models import User
from src.auth.schemas import AuthResponse, LoginRequest, MessageResponse, RegisterRequest, UserPublic
from src.auth.service import AuthError, get_auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=MessageResponse)
def register(body: RegisterRequest):
    try:
        user = get_auth_service().register(
            body.full_name, body.email, body.password, body.confirm_password
        )
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MessageResponse(message=f"Registration successful. Account created for {user.email}.")


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest):
    try:
        user, token = get_auth_service().login(body.email, body.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        # Surface config failures as 503 instead of opaque 500
        raise HTTPException(
            status_code=503,
            detail="Authentication service misconfigured. Set JWT_SECRET_KEY on the API host.",
        ) from exc
    return AuthResponse(
        access_token=token,
        user=UserPublic(**user.to_public_dict()),
        message="Login successful",
    )


@router.get("/me", response_model=UserPublic)
def me(user: User | None = Depends(get_current_user_optional)):
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return UserPublic(**user.to_public_dict())


@router.post("/logout", response_model=MessageResponse)
def logout():
    return MessageResponse(message="Logout successful. Discard the access token on the client.")
