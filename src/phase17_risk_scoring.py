"""
Phase 17 — Forecast-Driven Risk Scoring
=========================================
Uses forecast demand (not historical) for stockout and overstock risk.
Only applies to SYNTHETIC data where inventory is available.
UCI inventory = NOT_AVAILABLE.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P17_DIR = os.path.join(BASE_DIR, "data", "phase17")
P17_PROC = os.path.join(P17_DIR, "processed")
P17_RISK = os.path.join(P17_DIR, "risk")
P17_BT = os.path.join(P17_DIR, "backtests")
P17_FCST = os.path.join(P17_DIR, "forecasts")
os.makedirs(P17_RISK, exist_ok=True)

HORIZON_WEEKS = 8


def run_risk_scoring():
    print("=" * 60)
    print("PHASE 17 — FORECAST-DRIVEN RISK SCORING")
    print("=" * 60)

    # Load inventory (only SYNTHETIC has it)
    inv_path = os.path.join(P17_PROC, "synthetic_weekly_inventory.parquet")
    sku_path = os.path.join(P17_PROC, "sku_master.csv")

    if not os.path.exists(inv_path):
        print("WARNING: No inventory data available. Risk scoring limited.")
        return {"status": "PARTIAL", "reason": "No inventory data"}

    inv = pd.read_parquet(inv_path)
    inv["week"] = pd.to_datetime(inv["week"])

    skus = pd.read_csv(sku_path) if os.path.exists(sku_path) else None

    # Get latest inventory snapshot per store-SKU
    latest_week = inv["week"].max()
    latest_inv = inv[inv["week"] == latest_week].copy()
    print(f"Latest inventory week: {latest_week}")
    print(f"Latest inventory rows: {len(latest_inv):,}")

    # Load backtest results to get forecast demand
    bt_path = os.path.join(P17_BT, "backtest_results.parquet")
    if os.path.exists(bt_path):
        bt = pd.read_parquet(bt_path)
        bt_syn = bt[bt["source_dataset"] == "SYNTHETIC"].copy()
    else:
        bt_syn = pd.DataFrame()

    # Use selected forecast to estimate weekly demand per SKU
    # Aggregate forecast demand across folds (mean of selected_forecast)
    if len(bt_syn) > 0 and "selected_forecast" in bt_syn.columns:
        sku_forecast = bt_syn.groupby("product_key").agg(
            forecast_weekly_demand=("selected_forecast", "mean"),
        ).reset_index()
        sku_forecast["sku_id"] = sku_forecast["product_key"].str.replace("SYN_", "", regex=False)
        demand_source = "FORECAST"
    else:
        # Fallback: use last 8 weeks of actuals as demand estimate
        demand_path = os.path.join(P17_PROC, "synthetic_weekly_demand.parquet")
        if os.path.exists(demand_path):
            demand = pd.read_parquet(demand_path)
            demand["week"] = pd.to_datetime(demand["week"])
            recent = demand[demand["week"] >= demand["week"].max() - pd.Timedelta(weeks=8)]
            sku_forecast = recent.groupby("product_key").agg(
                forecast_weekly_demand=("units_sold", "mean"),
            ).reset_index()
            sku_forecast["sku_id"] = sku_forecast["product_key"].str.replace("SYN_", "", regex=False)
            demand_source = "HISTORICAL_FALLBACK"
        else:
            print("ERROR: No demand data available")
            return {"status": "FAIL"}
    print(f"Demand source: {demand_source}")

    # Aggregate inventory to SKU level (sum across stores)
    inv_sku = latest_inv.groupby("sku_id").agg(
        on_hand_units=("ending_inventory", "sum"),
        on_order_units=("on_order_qty", "sum"),
    ).reset_index()

    # Merge with SKU master for lead time, reorder point, safety stock, cost
    if skus is not None:
        sku_cols = ["sku_id", "lead_time_days", "reorder_point", "safety_stock",
                    "cost_price", "base_price", "category", "sub_category"]
        available_cols = [c for c in sku_cols if c in skus.columns]
        inv_sku = inv_sku.merge(skus[available_cols], on="sku_id", how="left")

    # Merge with forecast demand
    inv_sku = inv_sku.merge(sku_forecast[["sku_id", "forecast_weekly_demand"]], on="sku_id", how="left")
    inv_sku["forecast_weekly_demand"] = inv_sku["forecast_weekly_demand"].fillna(0.1)

    # Convert lead time to weeks
    if "lead_time_days" in inv_sku.columns:
        inv_sku["lead_time_weeks"] = np.ceil(inv_sku["lead_time_days"] / 7).astype(int)
    else:
        inv_sku["lead_time_weeks"] = 2  # default assumption, documented

    # --- STOCKOUT RISK ---
    inv_sku["lead_time_demand"] = inv_sku["forecast_weekly_demand"] * inv_sku["lead_time_weeks"]
    inv_sku["inventory_position"] = inv_sku["on_hand_units"] + inv_sku["on_order_units"]

    safety = inv_sku["safety_stock"] if "safety_stock" in inv_sku.columns else 0
    inv_sku["weeks_of_supply"] = inv_sku["on_hand_units"] / np.maximum(inv_sku["forecast_weekly_demand"], 0.01)

    stockout_score = np.zeros(len(inv_sku))
    # Zero stock
    stockout_score = np.where(inv_sku["on_hand_units"] <= 0, 100.0, stockout_score)
    # Below safety stock
    if "safety_stock" in inv_sku.columns:
        below_ss = (inv_sku["on_hand_units"] > 0) & (inv_sku["on_hand_units"] < inv_sku["safety_stock"])
        stockout_score = np.where(below_ss, 80.0, stockout_score)
    # Weeks of supply < lead time
    below_lt = (stockout_score == 0) & (inv_sku["weeks_of_supply"] < inv_sku["lead_time_weeks"])
    stockout_score = np.where(below_lt, 60.0, stockout_score)
    # Below reorder point
    if "reorder_point" in inv_sku.columns:
        below_rop = (stockout_score == 0) & (inv_sku["inventory_position"] <= inv_sku["reorder_point"])
        stockout_score = np.where(below_rop, 40.0, stockout_score)

    inv_sku["stockout_risk_score"] = np.clip(stockout_score, 0, 100)
    inv_sku["stockout_risk_level"] = np.select(
        [inv_sku["stockout_risk_score"] >= 70, inv_sku["stockout_risk_score"] >= 35],
        ["CRITICAL", "MEDIUM"],
        default="LOW"
    )

    # --- OVERSTOCK RISK ---
    forward_demand = inv_sku["forecast_weekly_demand"] * HORIZON_WEEKS
    excess = np.maximum(0, inv_sku["on_hand_units"] - forward_demand)
    inv_sku["excess_units"] = excess.astype(int)

    if "cost_price" in inv_sku.columns:
        inv_sku["locked_capital"] = inv_sku["excess_units"] * inv_sku["cost_price"]
    else:
        inv_sku["locked_capital"] = np.nan

    overstock_score = np.where(inv_sku["weeks_of_supply"] > HORIZON_WEEKS * 2, 80.0,
                     np.where(inv_sku["weeks_of_supply"] > HORIZON_WEEKS, 50.0, 0.0))
    overstock_score = np.where(inv_sku["forecast_weekly_demand"] < 0.5,
                               np.minimum(100.0, overstock_score + 20.0), overstock_score)
    inv_sku["overstock_risk_score"] = np.clip(overstock_score, 0, 100)
    inv_sku["overstock_risk_level"] = np.select(
        [inv_sku["overstock_risk_score"] >= 65, inv_sku["overstock_risk_score"] >= 30],
        ["SEVERE", "MODERATE"],
        default="OPTIMAL"
    )

    # --- DECISION GRID ---
    inv_sku["action"] = np.select(
        [inv_sku["stockout_risk_level"] == "CRITICAL",
         inv_sku["overstock_risk_level"] == "SEVERE",
         (inv_sku["stockout_risk_level"] == "MEDIUM") | (inv_sku["overstock_risk_level"] == "MODERATE"),
         ],
        ["REORDER NOW", "MARKDOWN / CLEAR", "WATCH / VOLATILE"],
        default="HEALTHY"
    )

    # --- RUPEE IMPACT ---
    if "base_price" in inv_sku.columns:
        inv_sku["sales_at_risk"] = np.where(
            inv_sku["stockout_risk_level"].isin(["CRITICAL", "MEDIUM"]),
            inv_sku["forecast_weekly_demand"] * inv_sku["lead_time_weeks"] * inv_sku["base_price"],
            0
        )
    else:
        inv_sku["sales_at_risk"] = np.nan

    # Save
    risk_path = os.path.join(P17_RISK, "forecast_driven_risk.parquet")
    inv_sku.to_parquet(risk_path, index=False)
    print(f"\nRisk matrix: {len(inv_sku):,} SKUs")

    summary = {
        "total_skus": len(inv_sku),
        "demand_source": demand_source,
        "stockout_critical": int((inv_sku["stockout_risk_level"] == "CRITICAL").sum()),
        "stockout_medium": int((inv_sku["stockout_risk_level"] == "MEDIUM").sum()),
        "stockout_low": int((inv_sku["stockout_risk_level"] == "LOW").sum()),
        "overstock_severe": int((inv_sku["overstock_risk_level"] == "SEVERE").sum()),
        "overstock_moderate": int((inv_sku["overstock_risk_level"] == "MODERATE").sum()),
        "overstock_optimal": int((inv_sku["overstock_risk_level"] == "OPTIMAL").sum()),
        "reorder_now": int((inv_sku["action"] == "REORDER NOW").sum()),
        "markdown_clear": int((inv_sku["action"] == "MARKDOWN / CLEAR").sum()),
        "watch_volatile": int((inv_sku["action"] == "WATCH / VOLATILE").sum()),
        "healthy": int((inv_sku["action"] == "HEALTHY").sum()),
    }

    if "locked_capital" in inv_sku.columns and inv_sku["locked_capital"].notna().any():
        summary["total_locked_capital"] = round(float(inv_sku["locked_capital"].sum()), 2)
    if "sales_at_risk" in inv_sku.columns and inv_sku["sales_at_risk"].notna().any():
        summary["total_sales_at_risk"] = round(float(inv_sku["sales_at_risk"].sum()), 2)

    for k, v in summary.items():
        print(f"  {k}: {v}")

    summary["uci_risk_status"] = "NOT_AVAILABLE — UCI has no inventory data"
    summary_path = os.path.join(P17_RISK, "risk_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    run_risk_scoring()
