"""
Phase 20 — Dashboard Adapter
==============================
Provides dashboard-compatible data without modifying existing dashboard logic.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs")


def model_info_panel() -> dict[str, Any]:
    reg_path = os.path.join(DOCS, "phase20_production_registry.json")
    entry = json.load(open(reg_path))[0] if os.path.exists(reg_path) else {}
    return {
        "production_model": "Phase 20 Synthetic LightGBM",
        "model_id": entry.get("model_id", "phase20_synthetic_lightgbm"),
        "source_dataset": "SYNTHETIC",
        "validated_horizon": "6 Weeks",
        "extended_forecast": "Weeks 7-8 / Partial Accuracy",
        "overall_wape": entry.get("wape"),
        "supported_horizon_wape": entry.get("supported_horizon_wape"),
        "known_limitation": (
            "Holiday bias remains partially unresolved during Nov-Dec. "
            "Holiday-period forecasts should be interpreted with additional review."
        ),
        "forecast_status_default": "PRODUCTION",
        "extended_status": "EXTENDED_PARTIAL",
    }


def load_production_forecasts() -> pd.DataFrame:
    from src.phase20_api_adapter import batch_forecast_from_features_df

    feat_path = os.path.join(BASE, "data", "phase19", "features", "synthetic_weekly_features.parquet")
    feat = pd.read_parquet(feat_path)
    feat["week"] = pd.to_datetime(feat["week"])
    latest = feat[feat["week"] == feat["week"].max()].copy()
    return batch_forecast_from_features_df(latest)


def load_production_risk() -> pd.DataFrame:
    from src.phase20_risk_adapter import load_production_risk_matrix
    return load_production_risk_matrix()


def dashboard_bundle() -> dict[str, Any]:
    """Complete payload for Phase 20 production dashboard view."""
    forecasts = load_production_forecasts()
    risk = load_production_risk()
    info = model_info_panel()
    return {
        "model_info": info,
        "forecasts": forecasts,
        "risk": risk,
        "forecast_count": len(forecasts),
        "sku_count": forecasts["product_key"].nunique() if len(forecasts) > 0 else 0,
        "horizons": sorted(forecasts["horizon"].unique().tolist()) if len(forecasts) > 0 else [],
    }


def to_dashboard_records(bundle: dict) -> list[dict]:
    """Flatten forecast + risk for dashboard table display."""
    fc = bundle["forecasts"]
    risk = bundle["risk"]
    if len(fc) == 0 or len(risk) == 0:
        return []

    risk_map = risk.set_index("sku_id").to_dict(orient="index")
    records = []
    for _, row in fc.iterrows():
        sku_id = str(row["product_key"]).replace("SYN_", "")
        r = risk_map.get(sku_id, {})
        records.append({
            "product_key": row["product_key"],
            "horizon": int(row["horizon"]),
            "forecast_demand": round(float(row["forecast_demand"]), 2),
            "forecast_status": row["forecast_status"],
            "stockout_risk": r.get("stockout_risk_level", "N/A"),
            "overstock_risk": r.get("overstock_risk_level", "N/A"),
            "recommended_action": r.get("recommended_action", "N/A"),
            "sales_at_risk": r.get("sales_at_risk"),
            "model_version": row.get("model_version", "phase20"),
            "source_dataset": row.get("source_dataset", "SYNTHETIC"),
        })
    return records
