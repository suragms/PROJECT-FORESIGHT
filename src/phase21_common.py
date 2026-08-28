"""Phase 21 — shared utilities."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs")
P21_DIR = os.path.join(BASE, "data", "phase21")
P21_MON = os.path.join(P21_DIR, "monitoring")
P21_HIST = os.path.join(P21_MON, "history")
P19_FEAT = os.path.join(BASE, "data", "phase19", "features", "synthetic_weekly_features.parquet")
P20_FCST = os.path.join(BASE, "data", "phase20", "production_forecasts.parquet")
P20_RISK = os.path.join(BASE, "data", "phase20", "production_risk.parquet")
CONTRACT_PATH = os.path.join(DOCS, "phase20_feature_contract.json")
P20_REG = os.path.join(DOCS, "phase20_production_registry.json")
P20_PROV = os.path.join(DOCS, "phase20_promotion_provenance.json")

SUPPORTED_HORIZON = 6
OVERALL_WAPE_BASELINE = 13.96
H16_WAPE_BASELINE = 11.03

for d in [P21_MON, P21_HIST]:
    os.makedirs(d, exist_ok=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def classify_status(checks: list[str]) -> str:
    """Rule-based: FAIL -> CRITICAL component; any WARNING -> at least WATCH."""
    if any(c == "FAIL" for c in checks):
        return "FAIL"
    if any(c == "WARNING" for c in checks):
        return "WARNING"
    return "PASS"


def wape(actual, forecast) -> float:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    d = np.sum(np.abs(a))
    return float(np.sum(np.abs(a - f)) / d) if d > 0 else np.nan


def bias(actual, forecast) -> float:
    return float(np.mean(np.asarray(forecast, float) - np.asarray(actual, float)))
