"""Phase 21 — Risk engine consistency and distribution monitoring."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.phase21_common import P20_RISK, now_iso, classify_status


def run_risk_monitoring(risk: pd.DataFrame | None = None) -> dict:
    if risk is None and os.path.exists(P20_RISK):
        risk = pd.read_parquet(P20_RISK)
    if risk is None or len(risk) == 0:
        return {"timestamp": now_iso(), "overall_status": "WARNING", "note": "No risk data"}

    inconsistencies = []

    # WoS consistency (tolerate rounding when demand or supply is near zero)
    if "weeks_of_supply" in risk.columns:
        expected_wos = risk["on_hand_units"] / np.maximum(risk["forecast_weekly_demand"], 0.01)
        wos_ok = np.allclose(
            risk["weeks_of_supply"].fillna(0), expected_wos, rtol=0.05, atol=0.1,
        )
        if not wos_ok:
            inconsistencies.append("weeks_of_supply_mismatch")

    # REORDER NOW must be CRITICAL stockout
    reorder = risk[risk["recommended_action"] == "REORDER NOW"]
    if len(reorder) > 0 and not (reorder["stockout_risk_level"] == "CRITICAL").all():
        inconsistencies.append("REORDER_NOW_without_CRITICAL_stockout")

    # HEALTHY must be LOW/OPTIMAL
    healthy = risk[risk["recommended_action"] == "HEALTHY"]
    if len(healthy) > 0:
        if not healthy["stockout_risk_level"].isin(["LOW"]).all():
            inconsistencies.append("HEALTHY_with_elevated_stockout")
        if not healthy["overstock_risk_level"].isin(["OPTIMAL"]).all():
            inconsistencies.append("HEALTHY_with_elevated_overstock")

    # Negative projected balance with HEALTHY
    if "projected_balance" in risk.columns:
        bad = healthy[healthy["projected_balance"] < -100] if len(healthy) > 0 else pd.DataFrame()
        if len(bad) > 0:
            inconsistencies.append("HEALTHY_with_severe_negative_balance")

    consistency_status = "FAIL" if inconsistencies else "PASS"

    # Distribution
    action_counts = risk["recommended_action"].value_counts().to_dict()
    total = len(risk)
    reorder_pct = 100 * action_counts.get("REORDER NOW", 0) / total
    distribution_shift = "WARNING" if reorder_pct > 80 else "PASS"

    return {
        "timestamp": now_iso(),
        "total_skus": total,
        "action_distribution": {k: int(v) for k, v in action_counts.items()},
        "reorder_pct": round(reorder_pct, 2),
        "inconsistencies": inconsistencies,
        "consistency_status": consistency_status,
        "distribution_status": distribution_shift,
        "overall_status": classify_status([consistency_status, distribution_shift]),
    }
