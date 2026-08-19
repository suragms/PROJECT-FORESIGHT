"""KPI calculations from existing artifacts. Missing values stay NOT AVAILABLE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    INVENTORY_RISK_PATH,
    OUTPUTS_MONITORING_DIR,
    PHASE11_META_PATH,
)


def _json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_risk() -> pd.DataFrame | None:
    path = Path(INVENTORY_RISK_PATH)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def inventory_kpis(risk: pd.DataFrame | None) -> dict[str, Any]:
    if risk is None or risk.empty:
        return {
            "n_rows": 0,
            "extract_note": "NOT AVAILABLE",
            "stockout_critical_high": "NOT AVAILABLE",
            "overstock_severe": "NOT AVAILABLE",
            "overstock_moderate": "NOT AVAILABLE",
            "reorder_review_count": "NOT AVAILABLE",
            "no_exceptional_risk": "NOT AVAILABLE",
            "mean_days_of_supply": "NOT AVAILABLE",
        }
    stockout_c = int((risk["stockout_risk_level"] == "CRITICAL / HIGH").sum()) if "stockout_risk_level" in risk.columns else "NOT AVAILABLE"
    over_s = int((risk["overstock_risk_level"] == "SEVERE OVERSTOCK").sum()) if "overstock_risk_level" in risk.columns else "NOT AVAILABLE"
    over_m = int((risk["overstock_risk_level"] == "MODERATE OVERSTOCK").sum()) if "overstock_risk_level" in risk.columns else "NOT AVAILABLE"
    reorder = int(risk["reorder_triggered"].sum()) if "reorder_triggered" in risk.columns else "NOT AVAILABLE"
    if "stockout_risk_level" in risk.columns and "overstock_risk_level" in risk.columns:
        no_ex = int(
            ((risk["stockout_risk_level"] == "LOW / SAFE") & (risk["overstock_risk_level"] == "OPTIMAL")).sum()
        )
    else:
        no_ex = "NOT AVAILABLE"
    dos = float(risk["days_of_supply"].mean()) if "days_of_supply" in risk.columns else "NOT AVAILABLE"
    return {
        "n_rows": int(len(risk)),
        "extract_note": "1000-row reference extract — not the operational inventory universe",
        "stockout_critical_high": stockout_c,
        "overstock_severe": over_s,
        "overstock_moderate": over_m,
        "reorder_review_count": reorder,
        "no_exceptional_risk": no_ex,
        "mean_days_of_supply": None if dos == "NOT AVAILABLE" else round(dos, 4),
    }


def demand_kpis(risk: pd.DataFrame | None) -> dict[str, Any]:
    if risk is None or risk.empty:
        return {
            "total_recent_units": "NOT AVAILABLE",
            "avg_daily_demand": "NOT AVAILABLE",
            "demand_volatility_cv": "NOT AVAILABLE",
            "n_products": "NOT AVAILABLE",
        }
    total = float(risk["total_recent_units"].sum()) if "total_recent_units" in risk.columns else "NOT AVAILABLE"
    avg = float(risk["avg_daily_demand"].mean()) if "avg_daily_demand" in risk.columns else "NOT AVAILABLE"
    if "std_daily_demand" in risk.columns and "avg_daily_demand" in risk.columns:
        denom = max(float(risk["avg_daily_demand"].mean()), 1e-9)
        cv = float(risk["std_daily_demand"].mean()) / denom
    else:
        cv = "NOT AVAILABLE"
    n_prod = int(risk["sku_id"].nunique()) if "sku_id" in risk.columns else "NOT AVAILABLE"
    return {
        "total_recent_units": None if total == "NOT AVAILABLE" else round(total, 4),
        "avg_daily_demand": None if avg == "NOT AVAILABLE" else round(avg, 4),
        "demand_volatility_cv": None if cv == "NOT AVAILABLE" else round(cv, 4),
        "n_products": n_prod,
        "growth_on_extract": "Insufficient Evidence",
        "growth_rule": "Extract has no independent historical vs recent window; YoY is not computed here.",
    }


def forecast_kpis() -> dict[str, Any]:
    acc = _json(OUTPUTS_MONITORING_DIR / "accuracy_monitoring_report.json")
    meta = _json(Path(PHASE11_META_PATH)) or {}
    if not acc:
        return {"mae": "NOT AVAILABLE", "rmse": "NOT AVAILABLE", "wape": "NOT AVAILABLE", "bias": "NOT AVAILABLE"}
    overall = (acc.get("overall") or {}).get("metrics") or {}
    h1 = [r for r in (acc.get("by_dataset_horizon") or []) if int(r.get("horizon", -1)) == 1]
    coverage = "NOT AVAILABLE"
    return {
        "n_with_actuals": (acc.get("overall") or {}).get("n_with_actuals"),
        "mae": overall.get("MAE", "NOT AVAILABLE"),
        "rmse": overall.get("RMSE", "NOT AVAILABLE"),
        "wape": overall.get("WAPE", "NOT AVAILABLE"),
        "bias": overall.get("bias", "NOT AVAILABLE"),
        "interval_coverage": coverage,
        "interval_coverage_note": "P10/P90 are interval companions; live coverage is not claimed.",
        "uci_h1_model": meta.get("final_uci_model", "NOT AVAILABLE"),
        "synthetic_h1_model": meta.get("final_synthetic_model", "NOT AVAILABLE"),
        "h1_rows": h1,
        "generated_at": acc.get("generated_at"),
    }


def operations_kpis() -> dict[str, Any]:
    summary = _json(OUTPUTS_MONITORING_DIR / "monitoring_summary.json") or {}
    api = summary.get("api_metrics") or _json(OUTPUTS_MONITORING_DIR / "api_metrics.json") or {}
    return {
        "monitoring_generated_at": summary.get("generated_at", "NOT AVAILABLE"),
        "n_forecasts": summary.get("n_forecasts", "NOT AVAILABLE"),
        "n_alerts": summary.get("n_alerts", "NOT AVAILABLE"),
        "retraining": summary.get("retraining", "disabled"),
        "request_count": api.get("request_count", "NOT AVAILABLE"),
        "error_rate": api.get("error_rate", "NOT AVAILABLE"),
        "mean_latency_ms": api.get("mean_latency_ms", "NOT AVAILABLE"),
        "auth_failures": api.get("auth_failures", "NOT AVAILABLE"),
        "note": api.get("note", "File snapshot; not live APM."),
    }


def executive_kpi_row() -> dict[str, Any]:
    risk = load_risk()
    inv = inventory_kpis(risk)
    dem = demand_kpis(risk)
    fc = forecast_kpis()
    ops = operations_kpis()
    row = {"layer": "executive_kpi", "decision_support_only": True}
    row.update({f"demand_{k}": v for k, v in dem.items() if k != "h1_rows"})
    row.update({f"inventory_{k}": v for k, v in inv.items()})
    row.update({
        "forecast_mae": fc.get("mae"),
        "forecast_rmse": fc.get("rmse"),
        "forecast_wape": fc.get("wape"),
        "forecast_bias": fc.get("bias"),
        "forecast_n_with_actuals": fc.get("n_with_actuals"),
        "forecast_interval_coverage": fc.get("interval_coverage"),
        "uci_h1_model": fc.get("uci_h1_model"),
        "synthetic_h1_model": fc.get("synthetic_h1_model"),
        "forecast_generated_at": fc.get("generated_at"),
    })
    row.update({f"ops_{k}": v for k, v in ops.items()})
    return row
