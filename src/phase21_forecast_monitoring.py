"""Phase 21 — Forecast performance and horizon monitoring."""

from __future__ import annotations

import os

import pandas as pd

from src.phase21_common import (
    SUPPORTED_HORIZON, OVERALL_WAPE_BASELINE, H16_WAPE_BASELINE,
    now_iso, wape, bias,
)

P19_BT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "phase19", "backtests", "backtest_results.parquet",
)


def run_forecast_performance_monitoring(bt: pd.DataFrame | None = None) -> dict:
    """
    Uses Phase 19 backtest where actuals exist as validation reference.
    Production actuals: PENDING_ACTUALS until live production data is collected.
    """
    if bt is None and os.path.exists(P19_BT_PATH):
        bt = pd.read_parquet(P19_BT_PATH)
        if "source_dataset" in bt.columns:
            bt = bt[bt["source_dataset"] == "SYNTHETIC"].copy()

    if bt is None or len(bt) == 0:
        return {
            "timestamp": now_iso(),
            "performance_status": "PENDING_ACTUALS",
            "note": "No actuals available for production performance measurement",
            "validation_baseline_wape": OVERALL_WAPE_BASELINE,
            "validation_baseline_h16_wape": H16_WAPE_BASELINE,
        }

    fcol = next((c for c in ["phase19_forecast", "candidate_forecast"] if c in bt.columns), None)
    if not fcol or "actual" not in bt.columns:
        return {"timestamp": now_iso(), "performance_status": "PENDING_ACTUALS"}

    # Overall (validation reference — NOT production measured)
    overall_wape = round(wape(bt["actual"], bt[fcol]) * 100, 4)
    overall_bias = round(bias(bt["actual"], bt[fcol]), 4)

    # Horizon breakdown (h1-h6 production; h7-h8 extended)
    horizon_rows = []
    for h in range(1, 9):
        if "horizon_step" in bt.columns:
            hdf = bt[bt["horizon_step"] == h]
        else:
            hdf = bt
        if len(hdf) == 0:
            continue
        hw = round(wape(hdf["actual"], hdf[fcol]) * 100, 4)
        hb = round(bias(hdf["actual"], hdf[fcol]), 4)
        horizon_rows.append({
            "horizon": h,
            "wape_pct": hw,
            "bias": hb,
            "n": len(hdf),
            "label": "PRODUCTION" if h <= SUPPORTED_HORIZON else "EXTENDED_PARTIAL",
            "status": "MEASURED",
        })

    prod_horizons = [r for r in horizon_rows if r["horizon"] <= SUPPORTED_HORIZON]
    h16_wape = round(
        wape(
            bt[bt["horizon_step"] <= SUPPORTED_HORIZON]["actual"],
            bt[bt["horizon_step"] <= SUPPORTED_HORIZON][fcol],
        ) * 100, 4
    ) if "horizon_step" in bt.columns else overall_wape

    perf_status = "MEASURED"
    if overall_wape > OVERALL_WAPE_BASELINE * 1.15:
        perf_status = "WARNING"
    if overall_wape > OVERALL_WAPE_BASELINE * 1.3:
        perf_status = "FAIL"

    return {
        "timestamp": now_iso(),
        "data_source": "VALIDATION_REFERENCE_NOT_LIVE_PRODUCTION",
        "performance_status": perf_status,
        "production_actuals_status": "PENDING_ACTUALS",
        "overall_wape_pct": overall_wape,
        "overall_bias": overall_bias,
        "h1_h6_wape_pct": h16_wape,
        "validation_baselines": {
            "overall_wape": OVERALL_WAPE_BASELINE,
            "h1_h6_wape": H16_WAPE_BASELINE,
        },
        "horizon_performance": horizon_rows,
        "overall_status": "PASS" if perf_status == "MEASURED" else perf_status,
    }
