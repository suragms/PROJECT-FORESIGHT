"""
Phase 20 — Risk Adapter (Production)
=====================================
Consumes Phase 20 forecasts; does not overwrite Phase 17/19 risk engines.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P17_PROC = os.path.join(BASE, "data", "phase17", "processed")
SUPPORTED_HORIZON = 6


def compute_risk(row: dict) -> dict:
    """Deterministic risk scoring from forecast + inventory."""
    on_hand = float(row.get("on_hand_units", 0))
    on_order = float(row.get("on_order_units", 0))
    forecast_weekly = float(row.get("forecast_weekly_demand", 0.1))
    lead_time_weeks = int(row.get("lead_time_weeks", 2))
    safety_stock = float(row.get("safety_stock", 0))
    reorder_point = float(row.get("reorder_point", 0))
    base_price = row.get("base_price")
    cost_price = row.get("cost_price")

    lead_time_demand = forecast_weekly * lead_time_weeks
    inventory_position = on_hand + on_order
    weeks_of_supply = on_hand / max(forecast_weekly, 0.01)
    projected_balance = inventory_position - lead_time_demand

    stockout_score = 0.0
    if on_hand <= 0:
        stockout_score = 100.0
    elif on_hand < safety_stock:
        stockout_score = 80.0
    elif weeks_of_supply < lead_time_weeks:
        stockout_score = 60.0
    elif inventory_position <= reorder_point:
        stockout_score = 40.0

    stockout_level = "CRITICAL" if stockout_score >= 70 else ("MEDIUM" if stockout_score >= 35 else "LOW")

    forward_demand = forecast_weekly * SUPPORTED_HORIZON
    excess = max(0, on_hand - forward_demand)
    overstock_score = 80.0 if weeks_of_supply > SUPPORTED_HORIZON * 2 else (
        50.0 if weeks_of_supply > SUPPORTED_HORIZON else 0.0
    )
    if forecast_weekly < 0.5:
        overstock_score = min(100.0, overstock_score + 20.0)
    overstock_level = "SEVERE" if overstock_score >= 65 else ("MODERATE" if overstock_score >= 30 else "OPTIMAL")

    if stockout_level == "CRITICAL":
        action = "REORDER NOW"
    elif overstock_level == "SEVERE":
        action = "MARKDOWN / CLEAR"
    elif stockout_level == "MEDIUM" or overstock_level == "MODERATE":
        action = "WATCH / VOLATILE"
    else:
        action = "HEALTHY"

    sales_at_risk = None
    if base_price is not None and not pd.isna(base_price) and stockout_level in ("CRITICAL", "MEDIUM"):
        sales_at_risk = round(lead_time_demand * float(base_price), 2)

    locked_capital = None
    if cost_price is not None and not pd.isna(cost_price) and excess > 0:
        locked_capital = round(excess * float(cost_price), 2)

    return {
        "sku_id": row.get("sku_id"),
        "forecast_weekly_demand": round(forecast_weekly, 2),
        "lead_time_demand": round(lead_time_demand, 2),
        "on_hand_units": int(on_hand),
        "on_order_units": int(on_order),
        "safety_stock": int(safety_stock) if not pd.isna(safety_stock) else None,
        "inventory_position": int(inventory_position),
        "projected_balance": round(projected_balance, 2),
        "weeks_of_supply": round(weeks_of_supply, 2),
        "stockout_risk_score": stockout_score,
        "stockout_risk_level": stockout_level,
        "overstock_risk_score": overstock_score,
        "overstock_risk_level": overstock_level,
        "recommended_action": action,
        "sales_at_risk": sales_at_risk,
        "locked_capital": locked_capital,
        "demand_source": "PHASE20_PRODUCTION_FORECAST",
        "supported_horizon_weeks": SUPPORTED_HORIZON,
    }


def explain_risk(row: dict) -> dict:
    """Full explainability payload for one SKU."""
    r = compute_risk(row)
    return {
        "forecast_demand": r["forecast_weekly_demand"],
        "lead_time_demand": r["lead_time_demand"],
        "on_hand": r["on_hand_units"],
        "on_order": r["on_order_units"],
        "safety_threshold": r.get("safety_stock"),
        "projected_balance": r["projected_balance"],
        "stockout_risk": r["stockout_risk_level"],
        "overstock_risk": r["overstock_risk_level"],
        "recommended_action": r["recommended_action"],
        "financial_impact": {
            "sales_at_risk": r.get("sales_at_risk"),
            "locked_capital": r.get("locked_capital"),
        },
    }


def score_portfolio(forecast_df: pd.DataFrame, inventory_df: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """Join forecasts with inventory and SKU master; return risk matrix."""
    fc = forecast_df.groupby("product_key").agg(
        forecast_weekly_demand=("forecast_demand", "mean"),
    ).reset_index()
    fc["sku_id"] = fc["product_key"].str.replace("SYN_", "", regex=False)

    inv = inventory_df.groupby("sku_id").agg(
        on_hand_units=("ending_inventory", "sum"),
        on_order_units=("on_order_qty", "sum"),
    ).reset_index()

    merged = inv.merge(fc[["sku_id", "forecast_weekly_demand"]], on="sku_id", how="left")
    sku_cols = ["sku_id", "lead_time_days", "reorder_point", "safety_stock", "base_price", "cost_price"]
    merged = merged.merge(sku_master[[c for c in sku_cols if c in sku_master.columns]], on="sku_id", how="left")
    merged["forecast_weekly_demand"] = merged["forecast_weekly_demand"].fillna(0.1)
    merged["lead_time_weeks"] = np.ceil(merged["lead_time_days"].fillna(14) / 7).astype(int)

    rows = [compute_risk(r.to_dict()) for _, r in merged.iterrows()]
    return pd.DataFrame(rows)


def load_production_risk_matrix() -> pd.DataFrame:
    """Build risk matrix from Phase 20 forecasts + latest inventory."""
    from src.phase20_api_adapter import batch_forecast_from_features_df

    feat_path = os.path.join(BASE, "data", "phase19", "features", "synthetic_weekly_features.parquet")
    inv_path = os.path.join(P17_PROC, "synthetic_weekly_inventory.parquet")
    sku_path = os.path.join(P17_PROC, "sku_master.csv")

    feat = pd.read_parquet(feat_path)
    feat["week"] = pd.to_datetime(feat["week"])
    latest_week = feat["week"].max()
    latest_feat = feat[feat["week"] == latest_week].copy()

    fc = batch_forecast_from_features_df(latest_feat)
    inv = pd.read_parquet(inv_path)
    inv["week"] = pd.to_datetime(inv["week"])
    latest_inv = inv[inv["week"] == inv["week"].max()]
    skus = pd.read_csv(sku_path)

    return score_portfolio(fc, latest_inv, skus)
