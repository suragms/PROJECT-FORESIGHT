"""
Phase 20 — API Adapter for Promoted Synthetic Weekly Forecast Model
=====================================================================
Additive layer; does not modify existing /forecast routes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "final", "phase20", "phase20_synthetic_lightgbm.joblib")
CONTRACT_PATH = os.path.join(BASE, "docs", "phase20_feature_contract.json")
REGISTRY_PATH = os.path.join(BASE, "docs", "phase20_production_registry.json")

SUPPORTED_HORIZON = 6
EXTENDED_HORIZONS = {7, 8}
MODEL_ID = "phase20_synthetic_lightgbm"
ALLOWED_SOURCE = "SYNTHETIC"


def load_contract() -> list[dict]:
    with open(CONTRACT_PATH) as f:
        data = json.load(f)
    return data["features"]


def required_feature_names() -> list[str]:
    return [f["feature_name"] for f in load_contract() if f.get("required", True)]


def validate_features(feature_row: dict) -> None:
    contract = load_contract()
    missing = []
    for feat in contract:
        if not feat.get("required", True):
            continue
        name = feat["feature_name"]
        if name not in feature_row or pd.isna(feature_row.get(name)):
            missing.append(name)
    if missing:
        raise ValueError(f"Missing required features: {missing[:10]}{'...' if len(missing) > 10 else ''}")


def validate_source(source_dataset: str) -> None:
    if source_dataset.upper() != ALLOWED_SOURCE:
        raise ValueError(
            f"Phase 20 model only supports source_dataset={ALLOWED_SOURCE}. "
            f"Received: {source_dataset}. UCI is RESEARCH CANDIDATE only."
        )


def load_model():
    import joblib
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Promoted model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def forecast_status(horizon: int) -> str:
    if horizon <= SUPPORTED_HORIZON:
        return "PRODUCTION"
    if horizon in EXTENDED_HORIZONS:
        return "EXTENDED_PARTIAL"
    raise ValueError(f"Horizon {horizon} not supported. Max extended: 8")


def generate_forecast(
    feature_rows: list[dict],
    source_dataset: str = "SYNTHETIC",
    forecast_origin: str | None = None,
    include_extended: bool = False,
) -> list[dict]:
    """
    Generate forecasts for horizons 1-6 (production) and optionally 7-8 (extended).
    Each feature_row must contain product_key and all required features for ONE origin week.
    For multi-horizon, caller provides one row per (product_key, target_week) or we predict
  single-step from same features for h=1 only.

    Production use: one row per SKU at forecast origin; returns h=1..6 using same features
    (direct multi-step approximation — same as Phase 17/19 backtest approach).
    """
    validate_source(source_dataset)
    model = load_model()
    req_cols = required_feature_names()
    origin = forecast_origin or datetime.now(timezone.utc).date().isoformat()
    results = []

    max_h = 8 if include_extended else SUPPORTED_HORIZON

    for row in feature_rows:
        validate_features(row)
        product_key = row.get("product_key", "UNKNOWN")
        X = np.array([[row[c] for c in req_cols]])
        pred = max(0.0, float(model.predict(X)[0]))

        for h in range(1, max_h + 1):
            week_offset = h  # weeks ahead from origin
            results.append({
                "model_id": MODEL_ID,
                "source_dataset": source_dataset.upper(),
                "forecast_origin": origin,
                "product_key": product_key,
                "forecast_week": row.get("forecast_week", origin),
                "horizon": h,
                "forecast_demand": pred if h == 1 else pred,  # point forecast per horizon step
                "forecast_status": forecast_status(h),
                "supported_horizon": SUPPORTED_HORIZON,
                "model_version": "phase20",
            })

    return results


def model_metadata() -> dict[str, Any]:
    reg = json.load(open(REGISTRY_PATH)) if os.path.exists(REGISTRY_PATH) else []
    entry = reg[0] if reg else {}
    return {
        "model_id": MODEL_ID,
        "source_dataset": ALLOWED_SOURCE,
        "forecast_grain": "weekly_sku",
        "supported_horizon_weeks": SUPPORTED_HORIZON,
        "extended_horizon": entry.get("extended_horizon", {"weeks": [7, 8], "accuracy_status": "PARTIAL"}),
        "wape": entry.get("wape"),
        "supported_horizon_wape": entry.get("supported_horizon_wape"),
        "status": "production",
        "limitations": entry.get("limitations", []),
    }


def batch_forecast_from_features_df(df: pd.DataFrame, source_dataset: str = "SYNTHETIC") -> pd.DataFrame:
    """Batch forecast from a features DataFrame (one row per SKU at origin)."""
    validate_source(source_dataset)
    model = load_model()
    req_cols = required_feature_names()
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    valid = df.dropna(subset=req_cols)
    if len(valid) == 0:
        raise ValueError("No rows with complete required features")

    X = valid[req_cols].values
    preds = np.maximum(0, model.predict(X))
    origin = str(valid["week"].iloc[0].date()) if "week" in valid.columns else datetime.now(timezone.utc).date().isoformat()

    rows = []
    for i, (_, row) in enumerate(valid.iterrows()):
        pk = row.get("product_key", f"SKU_{i}")
        for h in range(1, SUPPORTED_HORIZON + 1):
            rows.append({
                "model_id": MODEL_ID,
                "source_dataset": source_dataset.upper(),
                "forecast_origin": origin,
                "product_key": pk,
                "horizon": h,
                "forecast_demand": float(preds[i]),
                "forecast_status": "PRODUCTION",
                "supported_horizon": SUPPORTED_HORIZON,
                "model_version": "phase20",
            })
    return pd.DataFrame(rows)
