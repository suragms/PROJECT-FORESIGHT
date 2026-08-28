"""Phase 21 — Holiday period monitoring."""

from __future__ import annotations

import os

import pandas as pd

from src.phase21_common import now_iso, wape, bias

P19_BT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "phase19", "backtests", "backtest_results.parquet",
)
CAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "calendar.csv",
)


def run_holiday_monitoring(bt: pd.DataFrame | None = None) -> dict:
    if not os.path.exists(CAL_PATH):
        return {"timestamp": now_iso(), "status": "NOT AVAILABLE", "note": "HOLIDAY MONITORING DATA NOT AVAILABLE"}

    cal = pd.read_csv(CAL_PATH, parse_dates=["date"])
    cal["week_start"] = cal["date"].dt.to_period("W-MON").dt.start_time
    holiday_weeks = set(cal.groupby("week_start")["is_holiday"].max().pipe(lambda s: s[s > 0].index))

    if bt is None and os.path.exists(P19_BT):
        bt = pd.read_parquet(P19_BT)
        if "source_dataset" in bt.columns:
            bt = bt[bt["source_dataset"] == "SYNTHETIC"].copy()
        bt["forecast_week"] = pd.to_datetime(bt["forecast_week"])

    if bt is None or len(bt) == 0:
        return {"timestamp": now_iso(), "status": "PARTIAL", "note": "No backtest data for holiday comparison"}

    fcol = next((c for c in ["phase19_forecast", "candidate_forecast"] if c in bt.columns), None)
    if not fcol:
        return {"timestamp": now_iso(), "status": "PARTIAL"}

    bt["is_holiday_week"] = bt["forecast_week"].isin(holiday_weeks)
    holiday = bt[bt["is_holiday_week"]]
    non_holiday = bt[~bt["is_holiday_week"]]

    result = {
        "timestamp": now_iso(),
        "calendar_source": "data/raw/calendar.csv",
        "holiday_weeks_in_data": int(bt["is_holiday_week"].sum()),
        "status": "PASS",
    }

    if len(holiday) > 0 and len(non_holiday) > 0:
        result["holiday_wape_pct"] = round(wape(holiday["actual"], holiday[fcol]) * 100, 4)
        result["non_holiday_wape_pct"] = round(wape(non_holiday["actual"], non_holiday[fcol]) * 100, 4)
        result["holiday_bias"] = round(bias(holiday["actual"], holiday[fcol]), 4)
        result["non_holiday_bias"] = round(bias(non_holiday["actual"], non_holiday[fcol]), 4)
        if result["holiday_wape_pct"] > result["non_holiday_wape_pct"] * 1.3:
            result["status"] = "WARNING"
            result["note"] = "Holiday-period WAPE elevated vs non-holiday (documented limitation)"

    return result
