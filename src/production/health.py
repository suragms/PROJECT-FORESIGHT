"""Lightweight liveness helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from src.config import APP_VERSION


def liveness() -> dict[str, str]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
