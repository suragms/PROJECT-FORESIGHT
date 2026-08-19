"""Data-quality and freshness summaries from existing reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import (
    FINAL_FORECASTS_PATH,
    INVENTORY_RISK_PATH,
    OUTPUTS_MONITORING_DIR,
    PHASE11_META_PATH,
    PROJECT_ROOT,
)


def _json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def freshness() -> dict[str, Any]:
    mon = _json(OUTPUTS_MONITORING_DIR / "monitoring_summary.json") or {}
    acc = _json(OUTPUTS_MONITORING_DIR / "accuracy_monitoring_report.json") or {}
    meta = _json(Path(PHASE11_META_PATH)) or {}
    risk = Path(INVENTORY_RISK_PATH)
    fc = Path(FINAL_FORECASTS_PATH)
    forecast_generated = meta.get("executed_at_utc")
    if not forecast_generated and fc.exists():
        forecast_generated = datetime.fromtimestamp(fc.stat().st_mtime, tz=timezone.utc).isoformat()
    if not forecast_generated:
        forecast_generated = "NOT AVAILABLE"
    return {
        "label": "file snapshot — not live data",
        "data_as_of": mon.get("generated_at") or acc.get("generated_at") or "NOT AVAILABLE",
        "forecast_generated": forecast_generated,
        "monitoring_snapshot": mon.get("generated_at", "NOT AVAILABLE"),
        "inventory_risk_mtime_utc": (
            datetime.fromtimestamp(risk.stat().st_mtime, tz=timezone.utc).isoformat()
            if risk.exists() else "NOT AVAILABLE"
        ),
        "final_forecasts_exists": fc.exists(),
        "bi_generated_at": datetime.now(timezone.utc).isoformat(),
    }


def quality_scorecard() -> dict[str, Any]:
    dq = _json(OUTPUTS_MONITORING_DIR / "data_quality_report.json") or {}
    docs_dq = PROJECT_ROOT / "docs" / "data_quality_report.json"
    # Prefer monitoring forecast DQ; cleaning DQ is historical Phase 3.
    return {
        "source": "outputs/monitoring/data_quality_report.json",
        "generated_at": dq.get("generated_at", "NOT AVAILABLE"),
        "n_rows": dq.get("n_rows", "NOT AVAILABLE"),
        "n_columns": dq.get("n_columns", "NOT AVAILABLE"),
        "missing_required_columns": dq.get("missing_required_columns", []),
        "missing_value_rate_pct": dq.get("missing_value_rate_pct", {}),
        "duplicate_rate_pct": dq.get("duplicate_rate_pct", "NOT AVAILABLE"),
        "n_duplicates": dq.get("n_duplicates", "NOT AVAILABLE"),
        "invalid_negative_counts": dq.get("invalid_negative_counts", {}),
        "date_gaps": dq.get("date_gaps", "NOT AVAILABLE"),
        "categorical_consistency": dq.get("category_changes", {}),
        "leakage_columns_present": dq.get("leakage_columns_present", []),
        "schema_validity": "PASS" if not dq.get("missing_required_columns") else "FAIL",
        "feature_completeness_note": "Required forecast columns present if missing_required_columns is empty.",
        "no_invented_quality_score": True,
        "phase3_cleaning_report": str(docs_dq) if docs_dq.exists() else "NOT AVAILABLE",
    }
