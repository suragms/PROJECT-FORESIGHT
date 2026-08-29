"""Phase 21 monitoring API routes — additive observability only."""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, Depends, HTTPException

from src.auth.dependencies import require_admin
from src.phase21_common import P21_MON, DOCS
from src.phase21_integrity_monitoring import run_integrity_monitoring

router = APIRouter(dependencies=[Depends(require_admin)])


def _load_json(name: str) -> dict:
    path = os.path.join(P21_MON, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Monitoring artifact not found: {name}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/health")
def phase21_health():
    try:
        summary = _load_json("monitoring_summary.json")
    except HTTPException:
        return {"status": "NOT_RUN", "message": "Run python src/run_phase21.py first"}
    return {
        "health_score": summary.get("health_score"),
        "components": summary.get("components"),
        "timestamp": summary.get("timestamp"),
    }


@router.get("/monitoring/latest")
def phase21_monitoring_latest():
    return _load_json("monitoring_summary.json")


@router.get("/alerts")
def phase21_alerts():
    return _load_json("alerts.json")


@router.get("/integrity")
def phase21_integrity():
    baseline_path = os.path.join(DOCS, "phase21_production_integrity_baseline.json")
    baseline = json.load(open(baseline_path)) if os.path.exists(baseline_path) else None
    report = run_integrity_monitoring(baseline)
    return {"baseline": baseline, "current": report}
