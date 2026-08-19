"""In-process API operational metrics. Does not modify models."""

from __future__ import annotations

import threading
import time
from typing import Any

_lock = threading.Lock()
_state: dict[str, Any] = {
    "request_count": 0,
    "error_count": 0,
    "auth_failures": 0,
    "rate_limit_events": 0,
    "forecast_failures": 0,
    "latency_sum_ms": 0.0,
    "latency_count": 0,
    "batch_size_sum": 0,
    "batch_count": 0,
}


def reset() -> None:
    with _lock:
        for key in list(_state):
            _state[key] = 0 if not str(key).startswith("latency_sum") else 0.0
        _state["latency_sum_ms"] = 0.0


def incr(name: str, amount: int = 1) -> None:
    with _lock:
        _state[name] = int(_state.get(name, 0)) + amount


def observe_latency(duration_s: float) -> None:
    with _lock:
        _state["latency_sum_ms"] += duration_s * 1000.0
        _state["latency_count"] = int(_state["latency_count"]) + 1


def observe_batch(size: int) -> None:
    with _lock:
        _state["batch_size_sum"] = int(_state["batch_size_sum"]) + int(size)
        _state["batch_count"] = int(_state["batch_count"]) + 1


def snapshot() -> dict[str, Any]:
    with _lock:
        latency_count = int(_state["latency_count"])
        batch_count = int(_state["batch_count"])
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_count": int(_state["request_count"]),
            "error_count": int(_state["error_count"]),
            "error_rate": (
                round(int(_state["error_count"]) / int(_state["request_count"]), 4)
                if int(_state["request_count"])
                else 0.0
            ),
            "auth_failures": int(_state["auth_failures"]),
            "rate_limit_events": int(_state["rate_limit_events"]),
            "forecast_failures": int(_state["forecast_failures"]),
            "mean_latency_ms": (
                round(float(_state["latency_sum_ms"]) / latency_count, 3)
                if latency_count
                else None
            ),
            "mean_batch_size": (
                round(int(_state["batch_size_sum"]) / batch_count, 3)
                if batch_count
                else None
            ),
            "note": "In-process counters for this API worker. Not a live cloud APM export.",
        }
