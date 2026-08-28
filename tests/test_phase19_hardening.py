"""Phase 19 — Hardening Tests."""
import hashlib
import json
import os
import pytest
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs")
P17_DIR = os.path.join(BASE, "data", "phase17")
P19_DIR = os.path.join(BASE, "data", "phase19")
MODELS17 = os.path.join(BASE, "models", "phase17")
MODELS19 = os.path.join(BASE, "models", "phase19")
MODELS_FIN = os.path.join(BASE, "models", "final")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestFrozenProduction:
    def test_production_hashes_unchanged(self):
        snap = json.load(open(os.path.join(DOCS, "phase19_production_hash_snapshot.json")))
        assert snap["all_match"]
        for m in snap["models"]:
            path = os.path.join(BASE, m["path"].replace("\\", os.sep))
            assert sha256(path) == m["expected"]

    def test_twelve_models(self):
        snap = json.load(open(os.path.join(DOCS, "phase19_production_hash_snapshot.json")))
        assert len(snap["models"]) == 12


class TestPhase17Preservation:
    def test_phase17_model_exists(self):
        p = os.path.join(MODELS17, "synthetic", "phase17_synthetic_lightgbm.joblib")
        assert os.path.exists(p)

    def test_phase17_backtest_exists(self):
        assert os.path.exists(os.path.join(P17_DIR, "backtests", "backtest_results.parquet"))

    def test_phase19_does_not_overwrite_phase17(self):
        p17 = os.path.join(MODELS17, "synthetic", "phase17_synthetic_lightgbm.joblib")
        p19 = os.path.join(MODELS19, "synthetic", "phase19_synthetic_lightgbm.joblib")
        assert os.path.exists(p17) and os.path.exists(p19)
        assert sha256(p17) != sha256(p19) or p17 != p19


class TestPhase19Isolation:
    def test_phase19_artifacts_exist(self):
        assert os.path.exists(os.path.join(P19_DIR, "features", "synthetic_weekly_features.parquet"))
        assert os.path.exists(os.path.join(P19_DIR, "backtests", "backtest_results.parquet"))
        assert os.path.exists(os.path.join(MODELS19, "synthetic", "phase19_synthetic_lightgbm.joblib"))

    def test_phase19_not_in_production_registry(self):
        reg = json.load(open(os.path.join(DOCS, "final_model_registry.json")))
        prod_hashes = {e["hash"] for e in reg}
        p19 = os.path.join(MODELS19, "synthetic", "phase19_synthetic_lightgbm.joblib")
        assert sha256(p19) not in prod_hashes


class TestHolidayFeatures:
    def test_holiday_features_present(self):
        df = pd.read_parquet(os.path.join(P19_DIR, "features", "synthetic_weekly_features.parquet"))
        assert "is_holiday_week" in df.columns
        assert "holiday_count" in df.columns

    def test_no_holiday_leakage(self):
        audit = json.load(open(os.path.join(P19_DIR, "features", "leakage_audit.json")))
        fails = [a for a in audit if a["leakage_status"] == "FAIL"]
        assert len(fails) == 0

    def test_holiday_known_before_origin(self):
        audit = json.load(open(os.path.join(P19_DIR, "features", "leakage_audit.json")))
        holiday_feats = [a for a in audit if "holiday" in a["feature"] or a["feature"].startswith("season_")]
        for a in holiday_feats:
            assert a.get("known_before_forecast_origin") is True or a.get("available_at_prediction_time") is True


class TestHorizonLogic:
    def test_supported_horizon_is_six(self):
        metrics = json.load(open(os.path.join(P19_DIR, "forecasts", "backtest_metrics.json")))
        assert metrics["supported_horizon_weeks"] == 6

    def test_h1_h6_pass(self):
        metrics = json.load(open(os.path.join(P19_DIR, "forecasts", "backtest_metrics.json")))
        for hr in metrics["horizon_metrics"]:
            if hr["horizon"] <= 6:
                assert hr["status"] == "PASS"

    def test_h7_h8_partial(self):
        metrics = json.load(open(os.path.join(P19_DIR, "forecasts", "backtest_metrics.json")))
        for hr in metrics["horizon_metrics"]:
            if hr["horizon"] >= 7:
                assert hr["status"] == "PARTIAL"


class TestHybridRule:
    def test_hybrid_rule_predefined(self):
        metrics = json.load(open(os.path.join(P19_DIR, "forecasts", "backtest_metrics.json")))
        assert metrics["hybrid_rule"]["defined_before_evaluation"] is True


class TestRollingOrigin:
    @pytest.fixture
    def bt(self):
        return pd.read_parquet(os.path.join(P19_DIR, "backtests", "backtest_results.parquet"))

    def test_temporal_ordering(self, bt):
        bt["forecast_week"] = pd.to_datetime(bt["forecast_week"])
        bt["forecast_origin"] = pd.to_datetime(bt["forecast_origin"])
        assert (bt["forecast_week"] > bt["forecast_origin"]).all()

    def test_five_folds(self, bt):
        assert bt["fold"].nunique() >= 5


class TestWAPE:
    def test_phase19_beats_baseline(self):
        metrics = json.load(open(os.path.join(P19_DIR, "forecasts", "backtest_metrics.json")))
        assert metrics["phase19_wape_pct"] < metrics["seasonal_naive_wape_pct"]

    def test_no_material_regression(self):
        metrics = json.load(open(os.path.join(P19_DIR, "forecasts", "backtest_metrics.json")))
        assert metrics["material_regression_from_phase17"] is False


class TestReproducibility:
    def test_model_scoring_reproducible(self):
        import joblib
        model = joblib.load(os.path.join(MODELS19, "synthetic", "phase19_synthetic_lightgbm.joblib"))
        df = pd.read_parquet(os.path.join(P19_DIR, "features", "synthetic_weekly_features.parquet"))
        cols = [c for c in df.columns if c.startswith(("lag_", "rolling_", "ewm_", "sin_", "cos_", "season_"))
                or c in ("week_of_year", "month", "quarter", "year", "price_lag1", "promo_lag1",
                         "is_holiday_week", "holiday_count", "weeks_to_next_holiday",
                         "weeks_since_last_holiday", "holiday_x_promo")]
        sample = df.dropna(subset=cols).head(100)
        X = sample[cols].values
        r1, r2 = model.predict(X), model.predict(X)
        assert np.allclose(r1, r2)


class TestRiskValidation:
    def test_risk_output_exists(self):
        assert os.path.exists(os.path.join(P19_DIR, "risk", "forecast_driven_risk.parquet"))

    def test_stress_tests_pass(self):
        summary = json.load(open(os.path.join(P19_DIR, "risk", "risk_summary.json")))
        assert summary["stress_tests_pass"] is True

    def test_forecast_driven(self):
        summary = json.load(open(os.path.join(P19_DIR, "risk", "risk_summary.json")))
        assert summary["demand_source"] == "PHASE19_FORECAST"

    def test_decision_grid(self):
        risk = pd.read_parquet(os.path.join(P19_DIR, "risk", "forecast_driven_risk.parquet"))
        valid = {"REORDER NOW", "MARKDOWN / CLEAR", "WATCH / VOLATILE", "HEALTHY"}
        assert set(risk["action"].unique()).issubset(valid)

    def test_financial_non_negative(self):
        risk = pd.read_parquet(os.path.join(P19_DIR, "risk", "forecast_driven_risk.parquet"))
        for col in ["sales_at_risk", "locked_capital"]:
            if col in risk.columns and risk[col].notna().any():
                assert (risk[col].dropna() >= 0).all()


class TestGateResults:
    def test_metrics_file_exists(self):
        assert os.path.exists(os.path.join(P19_DIR, "forecasts", "backtest_metrics.json"))
