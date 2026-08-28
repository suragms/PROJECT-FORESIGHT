"""
Phase 17 — Feature Engineering & Leakage Audit
================================================
Weekly SKU-level features for demand forecasting.
All rolling/lag features use .shift(1) minimum to prevent leakage.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P17_DIR = os.path.join(BASE_DIR, "data", "phase17")
P17_PROC = os.path.join(P17_DIR, "processed")
P17_FEAT = os.path.join(P17_DIR, "features")
os.makedirs(P17_FEAT, exist_ok=True)

GRAIN = ["source_dataset", "product_key"]
TARGET = "units_sold"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build weekly demand features per source_dataset + product_key series."""
    df = df.sort_values(GRAIN + ["week"]).copy()
    df["week"] = pd.to_datetime(df["week"])

    features = []
    for (src, pk), grp in df.groupby(GRAIN):
        g = grp.sort_values("week").copy()
        y = g[TARGET]

        # Lag features (shifted by 1 to avoid leakage at weekly grain)
        for lag in [1, 2, 4, 7, 13, 26, 52]:
            g[f"lag_{lag}"] = y.shift(lag)

        # Rolling features (shifted by 1)
        for win in [4, 8, 13, 26]:
            shifted = y.shift(1)
            g[f"rolling_mean_{win}"] = shifted.rolling(win, min_periods=1).mean()
            g[f"rolling_std_{win}"] = shifted.rolling(win, min_periods=1).std().fillna(0)
            g[f"rolling_min_{win}"] = shifted.rolling(win, min_periods=1).min()
            g[f"rolling_max_{win}"] = shifted.rolling(win, min_periods=1).max()

        # EWM (shifted by 1)
        for span in [4, 13, 26]:
            g[f"ewm_{span}"] = y.shift(1).ewm(span=span, min_periods=1).mean()

        # Calendar features from week date
        g["week_of_year"] = g["week"].dt.isocalendar().week.astype(int)
        g["month"] = g["week"].dt.month
        g["quarter"] = g["week"].dt.quarter
        g["year"] = g["week"].dt.year

        # Cyclical encoding
        g["sin_week"] = np.sin(2 * np.pi * g["week_of_year"] / 52)
        g["cos_week"] = np.cos(2 * np.pi * g["week_of_year"] / 52)
        g["sin_month"] = np.sin(2 * np.pi * g["month"] / 12)
        g["cos_month"] = np.cos(2 * np.pi * g["month"] / 12)

        # Price features (lagged)
        if "avg_unit_price" in g.columns:
            g["price_lag1"] = g["avg_unit_price"].shift(1)

        # Promotion features (lagged)
        if "promotion_flag" in g.columns:
            g["promo_lag1"] = g["promotion_flag"].shift(1).fillna(0).astype(int)

        features.append(g)

    result = pd.concat(features, ignore_index=True)
    return result


def run_leakage_audit(df: pd.DataFrame) -> list:
    """Audit every feature column for potential temporal leakage."""
    audit = []
    target_cols = {TARGET, "revenue"}
    base_cols = {"week", "source_dataset", "product_key", "units_sold", "revenue",
                 "avg_unit_price", "transaction_count", "unique_customers",
                 "promotion_flag", "store_count"}

    for col in df.columns:
        if col in base_cols:
            continue

        entry = {"feature": col, "source": "derived", "leakage_status": "PASS"}

        if col.startswith("lag_"):
            lag_val = int(col.split("_")[1])
            entry["source"] = f"units_sold shifted by {lag_val} weeks"
            entry["available_at_prediction_time"] = True
            entry["leakage_status"] = "PASS" if lag_val >= 1 else "FAIL"

        elif col.startswith("rolling_") or col.startswith("ewm_"):
            entry["source"] = "units_sold shifted by 1 then rolling/ewm"
            entry["available_at_prediction_time"] = True

        elif col in ("week_of_year", "month", "quarter", "year",
                     "sin_week", "cos_week", "sin_month", "cos_month"):
            entry["source"] = "calendar math on forecast week"
            entry["available_at_prediction_time"] = True

        elif col == "price_lag1":
            entry["source"] = "avg_unit_price shifted by 1 week"
            entry["available_at_prediction_time"] = True

        elif col == "promo_lag1":
            entry["source"] = "promotion_flag shifted by 1 week"
            entry["available_at_prediction_time"] = True

        else:
            entry["source"] = "unknown"
            entry["available_at_prediction_time"] = "UNKNOWN"
            entry["leakage_status"] = "REVIEW"

        audit.append(entry)

    return audit


def run_feature_engineering():
    print("=" * 60)
    print("PHASE 17 — FEATURE ENGINEERING")
    print("=" * 60)

    datasets = []

    # UCI
    uci_path = os.path.join(P17_PROC, "uci_weekly_demand.parquet")
    if os.path.exists(uci_path):
        uci = pd.read_parquet(uci_path)
        print(f"UCI weekly: {len(uci):,} rows")
        datasets.append(uci)

    # Synthetic
    syn_path = os.path.join(P17_PROC, "synthetic_weekly_demand.parquet")
    if os.path.exists(syn_path):
        syn = pd.read_parquet(syn_path)
        print(f"Synthetic weekly: {len(syn):,} rows")
        datasets.append(syn)

    if not datasets:
        print("ERROR: No datasets found")
        return None

    combined = pd.concat(datasets, ignore_index=True)
    print(f"Combined: {len(combined):,} rows")

    # Build features
    featured = build_features(combined)
    print(f"Features built: {len(featured):,} rows, {len(featured.columns)} columns")

    # Leakage audit
    audit = run_leakage_audit(featured)
    leakage_fails = [a for a in audit if a["leakage_status"] == "FAIL"]
    print(f"Leakage audit: {len(audit)} features checked, {len(leakage_fails)} FAIL")

    if leakage_fails:
        print("  LEAKAGE DETECTED:")
        for a in leakage_fails:
            print(f"    {a['feature']}: {a['source']}")

    # Save
    feat_path = os.path.join(P17_FEAT, "weekly_features.parquet")
    featured.to_parquet(feat_path, index=False)
    print(f"Saved: {feat_path}")

    audit_path = os.path.join(P17_FEAT, "leakage_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    return {"rows": len(featured), "columns": len(featured.columns),
            "leakage_fails": len(leakage_fails), "audit": audit}


if __name__ == "__main__":
    run_feature_engineering()
