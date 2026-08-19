"""Readiness checks for registry, model hashes, and configuration."""

from __future__ import annotations

from typing import Any

from src.config import APP_VERSION
from src.forecasting.registry import load_registry, verify_hash
from src.production.config_validation import validate_runtime_config


def check_readiness() -> tuple[int, dict[str, Any]]:
    config_errors = validate_runtime_config()
    registry_verified = False
    models_verified = False
    detail = None
    try:
        recs = load_registry()
        registry_verified = bool(recs)
        for rec in recs:
            verify_hash(rec)
        models_verified = True
    except Exception as exc:
        detail = type(exc).__name__

    ready = registry_verified and models_verified and not config_errors
    body: dict[str, Any] = {
        "status": "ready" if ready else "not_ready",
        "version": APP_VERSION,
        "models_verified": models_verified,
        "registry_verified": registry_verified,
        "config_valid": not config_errors,
    }
    if not ready and detail:
        body["reason"] = detail
    if config_errors:
        body["config_errors"] = config_errors
    return (200 if ready else 503, body)
