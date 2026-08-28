"""
Phase 22 — Executive Dashboard Adapter
Combines Phase 20 production data and Phase 21 monitoring summaries.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from src.phase20_dashboard_adapter import dashboard_bundle, model_info_panel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs")
P21_MON = os.path.join(BASE, "data", "phase21", "monitoring")


def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_monitoring_summary() -> dict[str, Any]:
    summary = _load_json(os.path.join(P21_MON, "monitoring_summary.json"))
    if summary:
        return summary
    return {
        "health_score": "NOT_RUN",
        "components": {},
        "note": "Run python src/run_phase21.py to generate monitoring data",
    }


def business_impact_metrics(risk: pd.DataFrame) -> dict[str, Any]:
    if risk is None or len(risk) == 0:
        return {"status": "NOT AVAILABLE"}

    metrics: dict[str, Any] = {}
    if "sales_at_risk" in risk.columns:
        metrics["total_sales_at_risk"] = round(float(risk["sales_at_risk"].sum()), 2)
    else:
        metrics["total_sales_at_risk"] = "NOT AVAILABLE"

    if "locked_capital" in risk.columns:
        metrics["total_locked_capital"] = round(float(risk["locked_capital"].sum()), 2)
    else:
        metrics["total_locked_capital"] = "NOT AVAILABLE"

    if "stockout_risk_level" in risk.columns:
        metrics["at_risk_skus"] = int(
            risk["stockout_risk_level"].isin(["HIGH", "CRITICAL"]).sum()
        )
    else:
        metrics["at_risk_skus"] = "NOT AVAILABLE"

    if "overstock_risk_level" in risk.columns:
        metrics["high_overstock_skus"] = int(
            risk["overstock_risk_level"].isin(["HIGH", "SEVERE"]).sum()
        )
    else:
        metrics["high_overstock_skus"] = "NOT AVAILABLE"

    metrics["status"] = "AVAILABLE"
    return metrics


def executive_bundle() -> dict[str, Any]:
    bundle = dashboard_bundle()
    monitoring = load_monitoring_summary()
    risk = bundle["risk"]
    info = model_info_panel()

    contract = _load_json(os.path.join(DOCS, "phase20_feature_contract.json")) or {}

    return {
        "project_name": "PROJECT FORESIGHT",
        "system_status": "PRODUCTION PROMOTION COMPLETE",
        "monitoring_status": "MONITORING READY" if monitoring.get("health_score") != "NOT_RUN" else "NOT_RUN",
        "production_performance": "PENDING ACTUALS",
        "validation_overall_wape": 13.96,
        "validation_h16_wape": 11.03,
        "validation_label": "VALIDATION / BACKTEST",
        "model_info": info,
        "feature_count": contract.get("feature_count", 45),
        "forecasts": bundle["forecasts"],
        "risk": risk,
        "sku_count": bundle["sku_count"],
        "forecast_count": bundle["forecast_count"],
        "business_impact": business_impact_metrics(risk),
        "monitoring": monitoring,
        "monitoring_components": monitoring.get("components", {}),
        "alerts": monitoring.get("alerts", []),
    }
