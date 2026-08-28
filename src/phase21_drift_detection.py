"""Phase 21 — Data drift and prediction drift detection."""

from __future__ import annotations

import json
import os

import pandas as pd

from src.monitoring.forecast_monitor import psi
from src.phase21_common import P19_FEAT, P20_FCST, CONTRACT_PATH, now_iso

PSI_STABLE = 0.10
PSI_WATCH = 0.20
PSI_DRIFT = 0.25

P19_BT = os.path.join(os.path.dirname(P19_FEAT), "..", "backtests", "backtest_results.parquet")


def _contract_features():
    with open(CONTRACT_PATH) as f:
        return [feat["feature_name"] for feat in json.load(f)["features"]]


def _classify_psi(val: float) -> str:
    if val < PSI_STABLE:
        return "STABLE"
    if val < PSI_WATCH:
        return "WATCH"
    if val < PSI_DRIFT:
        return "DRIFT"
    return "CRITICAL_DRIFT"


def run_data_drift_monitoring(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_parquet(P19_FEAT)
    df["week"] = pd.to_datetime(df["week"])

    weeks = sorted(df["week"].unique())
    split = int(len(weeks) * 0.7)
    baseline_weeks = set(weeks[:split])
    current_weeks = set(weeks[-max(1, len(weeks) // 5):])

    baseline_df = df[df["week"].isin(baseline_weeks)]
    current_df = df[df["week"].isin(current_weeks)]

    feature_cols = _contract_features()
    numeric_feats = [f for f in feature_cols if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

    feature_drift = {}
    for f in numeric_feats[:20]:
        b = baseline_df[f].dropna().values
        c = current_df[f].dropna().values
        if len(b) < 10 or len(c) < 5:
            continue
        try:
            psi_val = psi(b, c)
            feature_drift[f] = {"psi": round(float(psi_val), 4), "status": _classify_psi(psi_val)}
        except Exception:
            feature_drift[f] = {"psi": None, "status": "WATCH"}

    b_dem = baseline_df.groupby("week")["units_sold"].sum()
    c_dem = current_df.groupby("week")["units_sold"].sum()
    dem_psi = psi(b_dem.values, c_dem.values) if len(b_dem) > 5 and len(c_dem) > 2 else 0.0

    statuses = [v["status"] for v in feature_drift.values()]
    n_critical = sum(1 for s in statuses if s == "CRITICAL_DRIFT")
    n_drift = sum(1 for s in statuses if s == "DRIFT")
    if n_critical >= 3:
        overall = "FAIL"
    elif n_critical >= 1 or n_drift >= 5:
        overall = "WARNING"
    else:
        overall = "PASS"

    return {
        "timestamp": now_iso(),
        "baseline_weeks": len(baseline_weeks),
        "current_weeks": len(current_weeks),
        "feature_drift": feature_drift,
        "demand_psi": round(float(dem_psi), 4),
        "demand_drift_status": _classify_psi(dem_psi),
        "sku_coverage_baseline": int(baseline_df["product_key"].nunique()),
        "sku_coverage_current": int(current_df["product_key"].nunique()),
        "thresholds_documented": {"PSI_STABLE": PSI_STABLE, "PSI_WATCH": PSI_WATCH, "PSI_DRIFT": PSI_DRIFT},
        "overall_status": overall,
    }


def run_prediction_drift_monitoring(fcst: pd.DataFrame | None = None) -> dict:
    if fcst is None and os.path.exists(P20_FCST):
        fcst = pd.read_parquet(P20_FCST)
    if fcst is None or len(fcst) == 0:
        return {"timestamp": now_iso(), "overall_status": "WARNING", "note": "No production forecasts available"}

    col = "forecast_demand" if "forecast_demand" in fcst.columns else "prediction"
    p = pd.to_numeric(fcst[col], errors="coerce")

    current_stats = {
        "mean": round(float(p.mean()), 4),
        "median": round(float(p.median()), 4),
        "std": round(float(p.std()), 4),
        "zero_rate_pct": round(100 * float((p == 0).mean()), 2),
        "p25": round(float(p.quantile(0.25)), 4),
        "p75": round(float(p.quantile(0.75)), 4),
    }

    baseline_stats = {}
    shift_detected = []
    if os.path.exists(P19_BT):
        bt = pd.read_parquet(P19_BT)
        bcol = next((c for c in ["phase19_forecast", "candidate_forecast", "forecast_demand"] if c in bt.columns), None)
        if bcol:
            bp = pd.to_numeric(bt[bcol], errors="coerce")
            baseline_stats = {"mean": round(float(bp.mean()), 4), "std": round(float(bp.std()), 4)}
            if baseline_stats["mean"] > 0:
                shift = (current_stats["mean"] - baseline_stats["mean"]) / baseline_stats["mean"]
                if abs(shift) > 0.3:
                    shift_detected.append(f"mean_shift_{shift:.1%}")
            if baseline_stats["std"] > 0 and current_stats["std"] / baseline_stats["std"] > 2:
                shift_detected.append("variance_explosion")

    if current_stats["zero_rate_pct"] > 50:
        shift_detected.append("excessive_zero_forecasts")

    return {
        "timestamp": now_iso(),
        "current_distribution": current_stats,
        "baseline_distribution": baseline_stats,
        "shifts_detected": shift_detected,
        "overall_status": "WARNING" if shift_detected else "PASS",
    }
