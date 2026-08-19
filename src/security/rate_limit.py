"""In-memory sliding-window rate limiter for the API process."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        if limit <= 0 or window_seconds <= 0:
            return False
        now = time.monotonic()
        cutoff = now - float(window_seconds)
        with self._lock:
            q = self._events[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True


def client_key(host: str | None, path: str, bucket: str) -> str:
    return f"{host or 'unknown'}|{bucket}|{path}"
