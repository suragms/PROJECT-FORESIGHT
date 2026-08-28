"""Phase 21 — Production Monitoring Tests."""

import hashlib
import json
import os
import pytest
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs")
P21_MON = os.path.join(BASE, "data", "phase21", "monitoring")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def feat_df():
    return pd.read_parquet(os.path.join(BASE, "data", "phase19", "features", "synthetic_weekly_features.parquet"))


class TestIntegrityBaseline:
    def test_baseline_exists(self):
        assert os.path.exists(os.path.join(DOCS, "phase21_production_integrity_baseline.json"))

    def test_frozen_12_match(self):
        baseline = json.load(open(os.path.join(DOCS, "phase21_production_integrity_baseline.json")))
        assert baseline["all_frozen_match"] is True
        assert baseline["frozen_models_count"] == 12

    def test_phase20_hash_matches_registry(self):
        baseline = json.load(open(os.path.join(DOCS, "phase21_production_integrity_baseline.json")))
        reg = json.load(open(os.path.join(DOCS, "phase20_production_registry.json")))
        assert baseline["phase20_production"]["match"] is True
        assert baseline["phase20_production"]["actual_hash"] == reg[0]["hash"]


class TestFeatureContract:
    def test_contract_45_features(self):
        contract = json.load(open(os.path.join(DOCS, "phase20_feature_contract.json")))
        assert contract["feature_count"] == 45

    def test_feature_quality_passes_on_reference(self, feat_df):
        from src.phase21_feature_quality import run_feature_quality_monitoring
        report = run_feature_quality_monitoring(feat_df)
        assert report["overall_status"] == "PASS"
        assert report["missing_required"] == "PASS"

    def test_missing_feature_fails(self, feat_df):
        from src.phase21_feature_quality import run_feature_quality_monitoring
        bad = feat_df.drop(columns=["lag_1"])
        report = run_feature_quality_monitoring(bad)
        assert report["missing_required"] == "FAIL"


class TestDataQuality:
    def test_stable_data_passes(self, feat_df):
        from src.phase21_data_quality import run_data_quality_monitoring
        report = run_data_quality_monitoring(feat_df)
        assert report["overall_status"] == "PASS"

    def test_negative_quantities_fail(self, feat_df):
        from src.phase21_data_quality import run_data_quality_monitoring
        bad = feat_df.copy()
        bad.loc[bad.index[0], "units_sold"] = -1
        report = run_data_quality_monitoring(bad)
        assert report["negative_quantities"] == "FAIL"


class TestDriftDetection:
    def test_data_drift_returns_status(self, feat_df):
        from src.phase21_drift_detection import run_data_drift_monitoring
        report = run_data_drift_monitoring(feat_df)
        assert "overall_status" in report
        assert "thresholds_documented" in report

    def test_demand_shift_detected(self, feat_df):
        from src.phase21_drift_detection import run_data_drift_monitoring
        shifted = feat_df.copy()
        shifted["units_sold"] = shifted["units_sold"] * 5
        report = run_data_drift_monitoring(shifted)
        assert report["overall_status"] in ("WARNING", "FAIL")

    def test_prediction_drift_excessive_zeros(self):
        from src.phase21_drift_detection import run_prediction_drift_monitoring
        fc = pd.DataFrame({"forecast_demand": [0.0] * 100})
        report = run_prediction_drift_monitoring(fc)
        assert "excessive_zero_forecasts" in report.get("shifts_detected", [])


class TestForecastPerformance:
    def test_horizon_monitoring(self):
        from src.phase21_forecast_monitoring import run_forecast_performance_monitoring
        report = run_forecast_performance_monitoring()
        assert "horizon_performance" in report
        prod = [h for h in report["horizon_performance"] if h["label"] == "PRODUCTION"]
        assert len(prod) == 6

    def test_pending_actuals_label(self):
        from src.phase21_forecast_monitoring import run_forecast_performance_monitoring
        report = run_forecast_performance_monitoring()
        assert report.get("production_actuals_status") == "PENDING_ACTUALS"


class TestHolidayMonitoring:
    def test_holiday_monitoring_runs(self):
        from src.phase21_holiday_monitoring import run_holiday_monitoring
        report = run_holiday_monitoring()
        assert report["status"] in ("PASS", "WARNING", "PARTIAL", "NOT AVAILABLE")


class TestRiskConsistency:
    def test_production_risk_passes(self):
        from src.phase21_risk_monitoring import run_risk_monitoring
        risk = pd.read_parquet(os.path.join(BASE, "data", "phase20", "production_risk.parquet"))
        report = run_risk_monitoring(risk)
        assert report["consistency_status"] == "PASS"

    def test_inconsistent_risk_fails(self):
        from src.phase21_risk_monitoring import run_risk_monitoring
        fake = pd.DataFrame({
            "recommended_action": ["HEALTHY"],
            "stockout_risk_level": ["CRITICAL"],
            "overstock_risk_level": ["OPTIMAL"],
            "on_hand_units": [0], "forecast_weekly_demand": [100],
            "weeks_of_supply": [0], "projected_balance": [-500],
        })
        report = run_risk_monitoring(fake)
        assert report["consistency_status"] == "FAIL"


class TestAlertsAndHealth:
    def test_alert_structure(self):
        from src.phase21_monitoring import _make_alert
        alert = _make_alert("data_quality", "WARNING", "test", {"x": 1})
        for key in ["alert_id", "timestamp", "component", "severity", "message", "evidence", "recommended_action"]:
            assert key in alert

    def test_health_score_critical_on_fail(self):
        from src.phase21_monitoring import _health_score
        assert _health_score({"a": "FAIL", "b": "PASS"}) == "CRITICAL"

    def test_health_score_healthy(self):
        from src.phase21_monitoring import _health_score
        assert _health_score({"a": "PASS", "b": "PASS"}) == "HEALTHY"


class TestMonitoringHistory:
    def test_monitoring_artifacts_exist(self):
        from src.phase21_monitoring import run_phase21_monitoring
        run_phase21_monitoring()
        assert os.path.exists(os.path.join(P21_MON, "monitoring_summary.json"))
        assert os.path.exists(os.path.join(P21_MON, "data_quality_report.json"))
        hist_dir = os.path.join(P21_MON, "history")
        assert len(os.listdir(hist_dir)) >= 1


class TestModelHashVerification:
    def test_integrity_monitoring_pass(self):
        from src.phase21_integrity_monitoring import run_integrity_monitoring
        report = run_integrity_monitoring()
        assert report["status"] == "PASS"
        assert report["frozen_12_unchanged"] is True
        assert report["phase20_unchanged"] is True


class TestDriftSimulations:
    def test_all_simulations_pass(self):
        from src.phase21_monitoring import run_drift_simulations
        results = run_drift_simulations()
        assert len(results) == 9
        assert all(r["pass"] for r in results)


class TestAPIRoutes:
    def test_phase21_routes_import(self):
        from src.api.phase21_routes import router
        paths = [r.path for r in router.routes]
        assert "/health" in paths
        assert "/monitoring/latest" in paths
        assert "/alerts" in paths
        assert "/integrity" in paths


class TestDocumentation:
    def test_docs_exist(self):
        docs = [
            "phase21_monitoring_architecture.md",
            "phase21_data_drift_report.md",
            "phase21_prediction_monitoring.md",
            "phase21_alerting_policy.md",
            "phase21_model_integrity_policy.md",
            "phase21_performance_baseline.md",
            "phase21_final_monitoring_report.md",
        ]
        for d in docs:
            assert os.path.exists(os.path.join(DOCS, d)), f"Missing {d}"
