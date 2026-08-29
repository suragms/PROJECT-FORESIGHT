"""Phase 23.1 — User authentication (additive; separate from API-key middleware)."""

from src.auth.service import AuthService, get_auth_service

__all__ = ["AuthService", "get_auth_service"]
