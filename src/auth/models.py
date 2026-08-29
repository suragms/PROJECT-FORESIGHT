"""Phase 23.1 — User model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class User:
    id: int
    full_name: str
    email: str
    is_active: bool
    role: str
    created_at: str
    last_login: str | None

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "is_active": self.is_active,
            "role": self.role,
            "created_at": self.created_at,
            "last_login": self.last_login,
        }
