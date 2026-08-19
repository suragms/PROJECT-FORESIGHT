"""Fail-safe runtime configuration validation for the forecast API."""

from __future__ import annotations

from typing import Any

from src.config import (
    VALID_ENVIRONMENTS,
    VALID_LOG_LEVELS,
    api_api_key,
    api_auth_enabled,
    api_host,
    api_max_batch,
    api_max_payload_bytes,
    api_port,
    foresight_env,
    log_level,
    rate_limit_enabled,
    rate_limit_forecast_requests,
    rate_limit_requests,
    rate_limit_window_seconds,
)


class ConfigValidationError(ValueError):
    """Invalid or unsafe runtime configuration."""


def validate_runtime_config() -> list[str]:
    """Return human-readable configuration errors. Empty list means valid."""
    errors: list[str] = []
    env = foresight_env()
    if env not in VALID_ENVIRONMENTS:
        errors.append(
            f"FORESIGHT_ENV must be one of {list(VALID_ENVIRONMENTS)}, got '{env}'"
        )

    host = api_host()
    if not host:
        errors.append("FORESIGHT_API_HOST must not be empty")

    try:
        port = api_port()
        if port < 1 or port > 65535:
            errors.append(f"FORESIGHT_API_PORT out of range: {port}")
    except (TypeError, ValueError):
        errors.append("FORESIGHT_API_PORT must be an integer")

    try:
        batch = api_max_batch()
        if batch < 1:
            errors.append("FORESIGHT_API_MAX_BATCH must be >= 1")
    except (TypeError, ValueError):
        errors.append("FORESIGHT_API_MAX_BATCH must be an integer")

    try:
        payload = api_max_payload_bytes()
        if payload < 1024:
            errors.append("FORESIGHT_API_MAX_PAYLOAD_BYTES must be >= 1024")
    except (TypeError, ValueError):
        errors.append("FORESIGHT_API_MAX_PAYLOAD_BYTES must be an integer")

    level = log_level()
    if level not in VALID_LOG_LEVELS:
        errors.append(
            f"FORESIGHT_LOG_LEVEL must be one of {list(VALID_LOG_LEVELS)}, got '{level}'"
        )

    auth_on = api_auth_enabled()
    key = api_api_key()
    if env == "production":
        if not auth_on:
            errors.append("production requires FORESIGHT_API_AUTH_ENABLED=true")
        if not key:
            errors.append("production requires FORESIGHT_API_API_KEY")
    elif auth_on and not key:
        errors.append("FORESIGHT_API_AUTH_ENABLED=true requires FORESIGHT_API_API_KEY")

    if rate_limit_enabled():
        try:
            req = rate_limit_requests()
            window = rate_limit_window_seconds()
            forecast = rate_limit_forecast_requests()
            if req < 1 or window < 1 or forecast < 1:
                errors.append("rate-limit integers must be >= 1")
            if forecast > req:
                errors.append(
                    "FORESIGHT_RATE_LIMIT_FORECAST_REQUESTS must be <= FORESIGHT_RATE_LIMIT_REQUESTS"
                )
        except (TypeError, ValueError):
            errors.append("rate-limit environment variables must be integers")

    return errors


def assert_runtime_config() -> None:
    errors = validate_runtime_config()
    if not errors:
        return
    env = foresight_env()
    joined = "; ".join(errors)
    if env == "production":
        raise ConfigValidationError(joined)
    # Development remains startable; callers should log the warnings.


def config_snapshot() -> dict[str, Any]:
    """Non-secret view of the active configuration."""
    return {
        "FORESIGHT_ENV": foresight_env(),
        "FORESIGHT_API_HOST": api_host(),
        "FORESIGHT_API_PORT": api_port(),
        "FORESIGHT_API_MAX_BATCH": api_max_batch(),
        "FORESIGHT_API_MAX_PAYLOAD_BYTES": api_max_payload_bytes(),
        "FORESIGHT_LOG_LEVEL": log_level(),
        "FORESIGHT_API_AUTH_ENABLED": api_auth_enabled(),
        "FORESIGHT_RATE_LIMIT_ENABLED": rate_limit_enabled(),
        "FORESIGHT_RATE_LIMIT_REQUESTS": rate_limit_requests(),
        "FORESIGHT_RATE_LIMIT_WINDOW_SECONDS": rate_limit_window_seconds(),
        "api_key_configured": bool(api_api_key()),
    }
