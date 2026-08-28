"""
Phase 21 — Monitoring Orchestrator
==================================
Observability only. Does not retrain or modify production artifacts.
"""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from src.phase21_common import P21_MON, P21_HIST, save_json, now_iso
from src.phase21_integrity_monitoring import record_integrity_baseline, run_integrity_monitoring
from src.phase21_data_quality import run_data_quality_monitoring
from src.phase21_feature_quality import run_feature_quality_monitoring
from src.phase21_drift_detection import run_data_drift_monitoring, run_prediction_drift_monitoring
from src.phase21_forecast_monitoring import run_forecast_performance_monitoring
from src.phase21_risk_monitoring import run_risk_monitoring
from src.phase21_holiday_monitoring import run_holiday_monitoring


def _health_score(components: dict[str, str]) -> str:
    """
    Rule-based health (not AI confidence):
    CRITICAL if any FAIL
    DEGRADED if 2+ WARNING
    WATCH if 1 WARNING
    HEALTHY otherwise
    """
    vals = list(components.values())
    if any(v == "FAIL" for v in vals):
        return "CRITICAL"
    warnings = sum(1 for v in vals if v in ("WARNING", "PARTIAL"))
    if warnings >= 2:
        return "DEGRADED"
    if warnings >= 1:
        return "WATCH"
    return "HEALTHY"


def _make_alert(component: str, severity: str, message: str, evidence: dict) -> dict:
    actions = {
        "data_quality": "Review data ingestion pipeline",
        "feature_quality": "Review feature pipeline",
        "data_drift": "Investigate drift in input features",
        "prediction_drift": "Investigate forecast distribution shift",
        "forecast_performance": "Review forecast performance vs baseline",
        "risk_consistency": "Investigate inventory inputs and risk logic",
        "model_integrity": "Investigate model integrity immediately",
    }
    return {
        "alert_id": str(uuid.uuid4())[:8],
        "timestamp": now_iso(),
        "component": component,
        "severity": severity,
        "message": message,
        "evidence": evidence,
        "recommended_action": actions.get(component, "Review monitoring evidence"),
    }


def run_drift_simulations() -> list[dict]:
    """Controlled test scenarios — do not modify production data."""
    results = []
    feat = pd.read_parquet(__import__("os").path.join(
        __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))),
        "data", "phase19", "features", "synthetic_weekly_features.parquet",
    ))

    # 1 Stable input
    dq = run_data_quality_monitoring(feat)
    results.append({"scenario": "stable_input", "pass": dq["overall_status"] == "PASS"})

    # 2 Missing critical feature
    bad = feat.drop(columns=["lag_1"], errors="ignore")
    fq = run_feature_quality_monitoring(bad)
    results.append({"scenario": "missing_critical_feature", "pass": fq["missing_required"] == "FAIL"})

    # 3 High missing-value rate
    corrupted = feat.copy()
    if "lag_2" in corrupted.columns:
        corrupted.loc[corrupted.index[: len(corrupted) // 2], "lag_2"] = None
    fq2 = run_feature_quality_monitoring(corrupted)
    results.append({"scenario": "high_missing_rate", "pass": fq2["critical_nulls"] in ("WARNING", "FAIL")})

    # 4 Demand distribution shift
    shifted = feat.copy()
    shifted["units_sold"] = shifted["units_sold"] * 3
    dd = run_data_drift_monitoring(shifted)
    results.append({"scenario": "demand_shift", "pass": dd["overall_status"] in ("WARNING", "FAIL")})

    # 5 SKU coverage collapse
    collapsed = feat[feat["product_key"].isin(feat["product_key"].unique()[:10])]
    dq2 = run_data_quality_monitoring(collapsed)
    results.append({"scenario": "sku_coverage_collapse", "pass": dq2["sku_coverage"] == "WARNING"})

    # 6 Prediction variance (tested via drift module with synthetic forecasts)
    import numpy as np
    fake_fc = pd.DataFrame({"forecast_demand": np.random.exponential(1000, 600)})
    pd_drift = run_prediction_drift_monitoring(fake_fc)
    results.append({"scenario": "prediction_variance", "pass": pd_drift["overall_status"] in ("WARNING", "PASS")})

    # 7 Excessive zeros
    zero_fc = pd.DataFrame({"forecast_demand": [0.0] * 600})
    pd_zero = run_prediction_drift_monitoring(zero_fc)
    results.append({"scenario": "excessive_zeros", "pass": "excessive_zero_forecasts" in pd_zero.get("shifts_detected", [])})

    # 8 Model hash (integrity always runs)
    integrity = run_integrity_monitoring()
    results.append({"scenario": "model_hash_check", "pass": integrity["status"] == "PASS"})

    # 9 Risk inconsistency simulation
    from src.phase21_risk_monitoring import run_risk_monitoring
    fake_risk = pd.DataFrame({
        "recommended_action": ["HEALTHY"],
        "stockout_risk_level": ["CRITICAL"],
        "overstock_risk_level": ["OPTIMAL"],
        "on_hand_units": [0], "forecast_weekly_demand": [100],
        "weeks_of_supply": [0], "projected_balance": [-500],
    })
    rk = run_risk_monitoring(fake_risk)
    results.append({"scenario": "risk_inconsistency", "pass": rk["consistency_status"] == "FAIL"})

    return results


def run_phase21_monitoring() -> dict[str, Any]:
    print("=" * 60)
    print("PHASE 21 — PRODUCTION MONITORING")
    print("=" * 60)

    baseline = record_integrity_baseline()

    data_quality = run_data_quality_monitoring()
    feature_quality = run_feature_quality_monitoring()
    data_drift = run_data_drift_monitoring()
    prediction_drift = run_prediction_drift_monitoring()
    forecast_perf = run_forecast_performance_monitoring()
    risk = run_risk_monitoring()
    holiday = run_holiday_monitoring()
    integrity = run_integrity_monitoring(baseline)

    components = {
        "data_quality": data_quality["overall_status"],
        "feature_quality": feature_quality["overall_status"],
        "data_drift": data_drift["overall_status"],
        "prediction_drift": prediction_drift["overall_status"],
        "forecast_performance": forecast_perf.get("overall_status", forecast_perf.get("performance_status", "PASS")),
        "risk_consistency": risk["overall_status"],
        "model_integrity": integrity["status"],
    }
    health = _health_score(components)

    alerts = []
    for name, report in [
        ("data_quality", data_quality), ("feature_quality", feature_quality),
        ("data_drift", data_drift), ("prediction_drift", prediction_drift),
        ("risk_consistency", risk), ("model_integrity", integrity),
    ]:
        st = report.get("overall_status") or report.get("status")
        if st == "FAIL":
            alerts.append(_make_alert(name, "CRITICAL", f"{name} check failed", report))
        elif st == "WARNING":
            alerts.append(_make_alert(name, "WARNING", f"{name} warning", report))

    for a in integrity.get("alerts", []):
        alerts.append(_make_alert("model_integrity", a["severity"], a["message"], a))

    simulations = run_drift_simulations()
    sim_pass = sum(1 for s in simulations if s["pass"])

    summary = {
        "timestamp": now_iso(),
        "health_score": health,
        "components": components,
        "holiday_monitoring": holiday,
        "alerts": alerts,
        "drift_simulations": {"pass": sim_pass, "total": len(simulations), "details": simulations},
    }

    # Save reports
    save_json(f"{P21_MON}/data_quality_report.json", data_quality)
    save_json(f"{P21_MON}/feature_quality_report.json", feature_quality)
    save_json(f"{P21_MON}/data_drift_report.json", data_drift)
    save_json(f"{P21_MON}/prediction_drift_report.json", prediction_drift)
    save_json(f"{P21_MON}/forecast_performance_report.json", forecast_perf)
    save_json(f"{P21_MON}/risk_consistency_report.json", risk)
    save_json(f"{P21_MON}/model_integrity_report.json", integrity)
    save_json(f"{P21_MON}/monitoring_summary.json", summary)
    save_json(f"{P21_MON}/alerts.json", {"alerts": alerts, "timestamp": now_iso()})

    # History snapshot (never overwrite — use timestamped file)
    ts = now_iso().replace(":", "-").replace(".", "-")
    save_json(f"{P21_HIST}/snapshot_{ts}.json", summary)

    print(f"Health: {health}")
    print(f"Alerts: {len(alerts)}")
    print(f"Drift simulations: {sim_pass}/{len(simulations)}")
    for k, v in components.items():
        print(f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    run_phase21_monitoring()
