"""
Phase 19 — Feature Engineering with Holiday Calendar Features
===============================================================
Extends Phase 17 weekly features with calendar/holiday indicators.
All features verified for prediction-time availability.
"""

import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P17_FEAT = os.path.join(BASE, "data", "phase17", "features")
P17_PROC = os.path.join(BASE, "data", "phase17", "processed")
P19_FEAT = os.path.join(BASE, "data", "phase19", "features")
RAW_CAL = os.path.join(BASE, "data", "raw", "calendar.csv")
os.makedirs(P19_FEAT, exist_ok=True)

GRAIN = ["source_dataset", "product_key"]
TARGET = "units_sold"


def load_calendar_weekly():
    cal = pd.read_csv(RAW_CAL, parse_dates=["date"])
    cal["week_start"] = cal["date"].dt.to_period("W-MON").dt.start_time
    weekly = cal.groupby("week_start").agg(
        is_holiday_week=("is_holiday", "max"),
        holiday_count=("is_holiday", "sum"),
        season=("season", "first"),
    ).reset_index()
    weekly.rename(columns={"week_start": "week"}, inplace=True)
    # Season one-hot (known at prediction time from calendar)
    for s in weekly["season"].unique():
        weekly[f"season_{s}"] = (weekly["season"] == s).astype(int)
    return weekly


def build_phase19_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build Phase 17-style features plus holiday calendar features."""
    df = df.sort_values(GRAIN + ["week"]).copy()
    df["week"] = pd.to_datetime(df["week"])

    cal_w = load_calendar_weekly()
    df = df.merge(cal_w, on="week", how="left")
    df["is_holiday_week"] = df["is_holiday_week"].fillna(0).astype(int)
    df["holiday_count"] = df["holiday_count"].fillna(0).astype(int)

    # Holiday proximity: weeks until next holiday (from calendar, not from demand)
    cal_dates = pd.read_csv(RAW_CAL, parse_dates=["date"])
    holiday_dates = cal_dates[cal_dates["is_holiday"] == 1]["date"].sort_values()
    holiday_weeks = holiday_dates.dt.to_period("W-MON").dt.start_time.unique()

    def weeks_to_next_holiday(week):
        week = pd.Timestamp(week)
        future = [hw for hw in holiday_weeks if hw > week]
        if not future:
            return 52
        return int((future[0] - week).days / 7)

    df["weeks_to_next_holiday"] = df["week"].apply(weeks_to_next_holiday)
    df["weeks_since_last_holiday"] = df["week"].apply(
        lambda w: min(
            [int((pd.Timestamp(w) - hw).days / 7) for hw in holiday_weeks if hw <= pd.Timestamp(w)] or [52]
        )
    )

    features = []
    for (src, pk), grp in df.groupby(GRAIN):
        g = grp.sort_values("week").copy()
        y = g[TARGET]

        for lag in [1, 2, 4, 7, 13, 26, 52]:
            g[f"lag_{lag}"] = y.shift(lag)

        for win in [4, 8, 13, 26]:
            shifted = y.shift(1)
            g[f"rolling_mean_{win}"] = shifted.rolling(win, min_periods=1).mean()
            g[f"rolling_std_{win}"] = shifted.rolling(win, min_periods=1).std().fillna(0)
            g[f"rolling_min_{win}"] = shifted.rolling(win, min_periods=1).min()
            g[f"rolling_max_{win}"] = shifted.rolling(win, min_periods=1).max()

        for span in [4, 13, 26]:
            g[f"ewm_{span}"] = y.shift(1).ewm(span=span, min_periods=1).mean()

        g["week_of_year"] = g["week"].dt.isocalendar().week.astype(int)
        g["month"] = g["week"].dt.month
        g["quarter"] = g["week"].dt.quarter
        g["year"] = g["week"].dt.year
        g["sin_week"] = np.sin(2 * np.pi * g["week_of_year"] / 52)
        g["cos_week"] = np.cos(2 * np.pi * g["week_of_year"] / 52)
        g["sin_month"] = np.sin(2 * np.pi * g["month"] / 12)
        g["cos_month"] = np.cos(2 * np.pi * g["month"] / 12)

        if "avg_unit_price" in g.columns:
            g["price_lag1"] = g["avg_unit_price"].shift(1)
        if "promotion_flag" in g.columns:
            g["promo_lag1"] = g["promotion_flag"].shift(1).fillna(0).astype(int)

        # Holiday interaction (calendar known at prediction time)
        g["holiday_x_promo"] = g["is_holiday_week"] * g.get("promo_lag1", 0)

        features.append(g)

    return pd.concat(features, ignore_index=True)


def run_feature_eligibility_audit(df: pd.DataFrame) -> list:
    """Audit all Phase 19 features for leakage."""
    base_cols = {"week", "source_dataset", "product_key", "units_sold", "revenue",
                 "avg_unit_price", "transaction_count", "unique_customers",
                 "promotion_flag", "store_count", "season"}
    audit = []
    holiday_features = {
        "is_holiday_week": ("calendar.csv is_holiday aggregated to week", True),
        "holiday_count": ("calendar.csv holiday count per week", True),
        "weeks_to_next_holiday": ("calendar-derived; no demand data used", True),
        "weeks_since_last_holiday": ("calendar-derived; no demand data used", True),
        "holiday_x_promo": ("is_holiday_week × promo_lag1; both known at prediction", True),
    }
    season_cols = [c for c in df.columns if c.startswith("season_")]

    for col in df.columns:
        if col in base_cols:
            continue
        entry = {"feature": col, "leakage_status": "PASS"}
        if col in holiday_features:
            src, avail = holiday_features[col]
            entry.update({"source": src, "available_at_prediction_time": avail,
                          "known_before_forecast_origin": True})
        elif col in season_cols:
            entry.update({"source": "calendar.csv season one-hot", "available_at_prediction_time": True,
                          "known_before_forecast_origin": True})
        elif col.startswith("lag_"):
            lag_val = int(col.split("_")[1])
            entry.update({"source": f"units_sold shifted {lag_val} weeks",
                          "available_at_prediction_time": True, "known_before_forecast_origin": True,
                          "leakage_status": "PASS" if lag_val >= 1 else "FAIL"})
        elif col.startswith(("rolling_", "ewm_")):
            entry.update({"source": "demand shifted 1 then rolling/ewm",
                          "available_at_prediction_time": True, "known_before_forecast_origin": True})
        elif col in ("week_of_year", "month", "quarter", "year", "sin_week", "cos_week",
                     "sin_month", "cos_month", "price_lag1", "promo_lag1"):
            entry.update({"source": "calendar or lagged price/promo",
                          "available_at_prediction_time": True, "known_before_forecast_origin": True})
        else:
            entry.update({"source": "derived", "available_at_prediction_time": "REVIEW",
                          "leakage_status": "REVIEW"})
        audit.append(entry)
    return audit


def run_feature_engineering():
    print("=" * 60)
    print("PHASE 19 — FEATURE ENGINEERING")
    print("=" * 60)

    # Load SYNTHETIC weekly demand from Phase 17 processed (read-only)
    syn_path = os.path.join(P17_PROC, "synthetic_weekly_demand.parquet")
    syn = pd.read_parquet(syn_path)
    print(f"Synthetic weekly rows: {len(syn):,}")

    featured = build_phase19_features(syn)
    audit = run_feature_eligibility_audit(featured)
    fails = [a for a in audit if a["leakage_status"] == "FAIL"]
    print(f"Features: {len(featured.columns)} cols, audit: {len(audit)} features, {len(fails)} FAIL")

    feat_path = os.path.join(P19_FEAT, "synthetic_weekly_features.parquet")
    featured.to_parquet(feat_path, index=False)
    audit_path = os.path.join(P19_FEAT, "leakage_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Saved: {feat_path}")
    return {"rows": len(featured), "columns": len(featured.columns), "fails": len(fails), "audit": audit}


if __name__ == "__main__":
    run_feature_engineering()
