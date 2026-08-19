"""Structured JSON audit logging. Never logs secrets or full payloads."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("forecast_service.audit")

_CAPTURE_LOCK = threading.Lock()
_CAPTURE: list[dict[str, Any]] = []
_CAPTURING = False

_REDACT_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "x-api-key",
    "foresight_api_api_key",
    "features",
    "records",
    "payload",
}


def _redact(value: Any, key: str | None = None) -> Any:
    if key is not None and key.lower() in _REDACT_KEYS:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value[:20]]
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "…"
    return value


def audit(event: str, **fields: Any) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **{k: _redact(v, k) for k, v in fields.items()},
    }
    with _CAPTURE_LOCK:
        if _CAPTURING:
            _CAPTURE.append(record)
    logger.info(json.dumps(record, default=str, separators=(",", ":")))


def start_capture() -> None:
    global _CAPTURING
    with _CAPTURE_LOCK:
        _CAPTURE.clear()
        _CAPTURING = True


def stop_capture() -> list[dict[str, Any]]:
    global _CAPTURING
    with _CAPTURE_LOCK:
        _CAPTURING = False
        return list(_CAPTURE)


def captured_events() -> list[dict[str, Any]]:
    with _CAPTURE_LOCK:
        return list(_CAPTURE)
