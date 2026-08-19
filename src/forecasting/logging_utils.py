"""Structured application logging for Phase 12."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from src.config import APP_NAME, LOG_LEVEL


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        base = (
            f"ts={ts} logger={record.name} level={record.levelname} "
            f"msg={record.getMessage()}"
        )
        extras = []
        for k, v in record.__dict__.items():
            if k in (
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message", "taskName",
            ):
                continue
            extras.append(f"{k}={v}")
        if extras:
            base += " " + " ".join(extras)
        return base


def configure_logging(level: str | None = None) -> logging.Logger:
    log = logging.getLogger(APP_NAME)
    if log.handlers:
        return log
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(KeyValueFormatter())
    log.addHandler(handler)
    log.setLevel(getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO))
    log.propagate = False
    logging.getLogger("forecast_service").handlers = log.handlers
    logging.getLogger("forecast_service").setLevel(log.level)
    logging.getLogger("forecast_service.audit").handlers = log.handlers
    logging.getLogger("forecast_service.audit").setLevel(log.level)
    logging.getLogger("forecast_service.audit").propagate = False
    logging.getLogger("final_forecasting").handlers = log.handlers
    logging.getLogger("final_forecasting").setLevel(log.level)
    return log
