"""Phase 23.1 — Auth request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


def _basic_email_check(value: str) -> str:
    v = value.strip().lower()
    if "@" not in v or "." not in v.split("@")[-1]:
        raise ValueError("Valid email required")
    return v


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=120)
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return _basic_email_check(v)

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return _basic_email_check(v)


class UserPublic(BaseModel):
    id: int
    full_name: str
    email: str
    is_active: bool
    role: str
    created_at: str
    last_login: str | None = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
    message: str = "Authentication successful"


class MessageResponse(BaseModel):
    message: str
