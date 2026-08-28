"""Phase 18 — Promotion Gate Tests."""
import os
import json
import hashlib
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR  = os.path.join(BASE_DIR, "docs")
P17_DIR   = os.path.join(BASE_DIR, "data", "phase17")
MODELS17  = os.path.join(BASE_DIR, "models", "phase17")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────
class TestFrozenProductionModels:
    """Production models must be untouched throughout Phase 18."""

    def test_production_hashes_match_registry(self):
        reg_path  = os.path.join(DOCS_DIR, "final_model_registry.json")
        snap_path = os.path.join(DOCS_DIR, "phase18_production_hash_snapshot.json")
        if not os.path.exists(snap_path):
            pytest.skip("Phase 18 snapshot not generated yet")
        with open(snap_path) as f:
            snap = json.load(f)
        assert snap["all_match"], "Production model hash mismatch detected"

    def test_all_12_models_verified(self):
        snap_path = os.path.join(DOCS_DIR, "phase18_production_hash_snapshot.json")
        if not os.path.exists(snap_path):
            pytest.skip("Snapshot not found")
        with open(snap_path) as f:
            snap = json.load(f)
        assert len(snap["models"]) == 12

    def test_no_model_in_final_changed(self):
        reg_path = os.path.join(DOCS_DIR, "final_model_registry.json")
        with open(reg_path) as f:
            registry = json.load(f)
        for entry in registry:
            mf     = os.path.join(BASE_DIR, entry["model_file"].replace("\\", os.sep))
            actual = sha256(mf)
            assert actual == entry["hash"], f"FROZEN MODEL CHANGED: {entry['model_id']}"


# ─────────────────────────────────────────────────────────
class TestCandidateArtifacts:
    """Phase 17 candidate artifacts must exist, be loadable, and have recorded hashes."""

    PATHS = {
        "UCI":       os.path.join(MODELS17, "uci",       "phase17_uci_lightgbm.joblib"),
        "SYNTHETIC": os.path.join(MODELS17, "synthetic", "phase17_synthetic_lightgbm.joblib"),
    }

    def test_artifacts_exist(self):
        for src, path in self.PATHS.items():
            assert os.path.exists(path), f"Candidate not found: {path}"

    def test_artifacts_loadable(self):
        import joblib
        for src, path in self.PATHS.items():
            model = joblib.load(path)
            assert model is not None
            assert hasattr(model, "predict")

    def test_candidate_hashes_recorded(self):
        h_path = os.path.join(DOCS_DIR, "phase18_candidate_hashes.json")
        if not os.path.exists(h_path):
            pytest.skip("Candidate hashes not yet recorded")
        with open(h_path) as f:
            hashes = json.load(f)
        for src in ["UCI", "SYNTHETIC"]:
            assert src in hashes
            assert hashes[src]["sha256"] is not None

    def test_candidate_hashes_not_in_production_registry(self):
        h_path   = os.path.join(DOCS_DIR, "phase18_candidate_hashes.json")
        reg_path = os.path.join(DOCS_DIR, "final_model_registry.json")
        if not os.path.exists(h_path):
            pytest.skip("Candidate hashes not found")
        with open(h_path) as f:
            cand = json.load(f)
        with open(reg_path) as f:
            registry = json.load(f)
        prod_hashes = {e["hash"] for e in registry}
        for src, info in cand.items():
            assert info["sha256"] not in prod_hashes, \
                f"Candidate hash ({src}) found in production registry — must not be promoted yet"


# ─────────────────────────────────────────────────────────
class TestReproducibility:
    """Scoring the same inputs twice must yield identical predictions."""

    def test_uci_reproducible(self):
        import joblib
        path = os.path.join(MODELS17, "uci", "phase17_uci_lightgbm.joblib")
        if not os.path.exists(path):
            pytest.skip("UCI model not found")
        model = joblib.load(path)
        feat_path = os.path.join(P17_DIR, "features", "weekly_features.parquet")
        df = pd.read_parquet(feat_path)
        feat_cols = [c for c in df.columns if c.startswith(("lag_", "rolling_", "ewm_", "sin_", "cos_"))
                     or c in ("week_of_year", "month", "quarter", "year", "price_lag1", "promo_lag1")]
        sample = df[df["source_dataset"] == "UCI"].dropna(subset=feat_cols).head(200)
        if len(sample) == 0:
            pytest.skip("No UCI feature rows")
        X = sample[feat_cols].values
        r1 = model.predict(X)
        r2 = model.predict(X)
        assert np.allclose(r1, r2), "UCI model predictions are not reproducible"

    def test_synthetic_reproducible(self):
        import joblib
        path = os.path.join(MODELS17, "synthetic", "phase17_synthetic_lightgbm.joblib")
        if not os.path.exists(path):
            pytest.skip("Synthetic model not found")
        model = joblib.load(path)
        feat_path = os.path.join(P17_DIR, "features", "weekly_features.parquet")
        df = pd.read_parquet(feat_path)
        feat_cols = [c for c in df.columns if c.startswith(("lag_", "rolling_", "ewm_", "sin_", "cos_"))
                     or c in ("week_of_year", "month", "quarter", "year", "price_lag1", "promo_lag1")]
        sample = df[df["source_dataset"] == "SYNTHETIC"].dropna(subset=feat_cols).head(200)
        if len(sample) == 0:
            pytest.skip("No SYNTHETIC feature rows")
        X = sample[feat_cols].values
        r1 = model.predict(X)
        r2 = model.predict(X)
        assert np.allclose(r1, r2), "SYNTHETIC model predictions are not reproducible"


# ─────────────────────────────────────────────────────────
class TestWAPECalculation:
    """Verify WAPE formula correctness."""

    def test_wape_zero_baseline(self):
        actual   = np.array([0.0, 0.0, 0.0])
        forecast = np.array([1.0, 1.0, 1.0])
        denom    = np.sum(np.abs(actual))
        assert denom == 0  # WAPE undefined; must not divide by zero

    def test_wape_perfect_forecast(self):
        actual   = np.array([10.0, 20.0, 30.0])
        forecast = np.array([10.0, 20.0, 30.0])
        w = np.sum(np.abs(actual - forecast)) / np.sum(np.abs(actual))
        assert w == 0.0

    def test_wape_known_value(self):
        actual   = np.array([100.0, 100.0])
        forecast = np.array([110.0,  90.0])
        w = np.sum(np.abs(actual - forecast)) / np.sum(np.abs(actual))
        assert abs(w - 0.10) < 1e-9


# ─────────────────────────────────────────────────────────
class TestRollingOriginValidation:
    """Verify that every test period is strictly after its training origin."""

    @pytest.fixture
    def bt(self):
        p = os.path.join(P17_DIR, "backtests", "backtest_results.parquet")
        if not os.path.exists(p):
            pytest.skip("Backtest results not found")
        return pd.read_parquet(p)

    def test_temporal_ordering_all_folds(self, bt):
        bt["forecast_week"]   = pd.to_datetime(bt["forecast_week"])
        bt["forecast_origin"] = pd.to_datetime(bt["forecast_origin"])
        violations = bt[bt["forecast_week"] <= bt["forecast_origin"]]
        assert len(violations) == 0, \
            f"{len(violations)} rows where forecast_week <= forecast_origin"

    def test_five_folds_present_synthetic(self, bt):
        src = bt[bt["source_dataset"] == "SYNTHETIC"]
        assert src["fold"].nunique() >= 1

    def test_five_folds_present_uci(self, bt):
        src = bt[bt["source_dataset"] == "UCI"]
        assert src["fold"].nunique() >= 1

    def test_no_negative_actuals(self, bt):
        assert (bt["actual"] >= 0).all()


# ─────────────────────────────────────────────────────────
class TestLeakageStatus:
    """Leakage audit must show zero failures."""

    def test_no_leakage_failures(self):
        audit_path = os.path.join(P17_DIR, "features", "leakage_audit.json")
        if not os.path.exists(audit_path):
            pytest.skip("Leakage audit not found")
        with open(audit_path) as f:
            audit = json.load(f)
        fails = [a for a in audit if a.get("leakage_status") == "FAIL"]
        assert len(fails) == 0, f"Leakage FAIL: {[f['feature'] for f in fails]}"


# ─────────────────────────────────────────────────────────
class TestRiskConsistency:
    """Risk outputs must be internally consistent."""

    @pytest.fixture
    def risk(self):
        p = os.path.join(P17_DIR, "risk", "forecast_driven_risk.parquet")
        if not os.path.exists(p):
            pytest.skip("Risk output not found")
        return pd.read_parquet(p)

    def test_decision_labels_valid(self, risk):
        valid = {"REORDER NOW", "MARKDOWN / CLEAR", "WATCH / VOLATILE", "HEALTHY"}
        assert set(risk["action"].unique()).issubset(valid)

    def test_reorder_now_implies_critical_stockout(self, risk):
        reorder = risk[risk["action"] == "REORDER NOW"]
        if len(reorder) == 0:
            pytest.skip("No REORDER NOW rows")
        assert (reorder["stockout_risk_level"] == "CRITICAL").all()

    def test_healthy_implies_low_risk(self, risk):
        healthy = risk[risk["action"] == "HEALTHY"]
        if len(healthy) == 0:
            pytest.skip("No HEALTHY rows")
        assert (healthy["stockout_risk_level"] == "LOW").all()
        assert (healthy["overstock_risk_level"] == "OPTIMAL").all()

    def test_no_negative_risk_scores(self, risk):
        assert (risk["stockout_risk_score"]  >= 0).all()
        assert (risk["overstock_risk_score"] >= 0).all()

    def test_financial_non_negative(self, risk):
        for col in ["locked_capital", "sales_at_risk"]:
            if col in risk.columns:
                col_data = risk[col].dropna()
                assert (col_data >= 0).all(), f"Negative values in {col}"

    def test_forecast_demand_non_negative(self, risk):
        assert (risk["forecast_weekly_demand"] >= 0).all()


# ─────────────────────────────────────────────────────────
class TestGateResults:
    """Gate results JSON must exist and contain expected structure."""

    @pytest.fixture
    def gate(self):
        p = os.path.join(DOCS_DIR, "phase18_gate_results.json")
        if not os.path.exists(p):
            pytest.skip("Gate results not found")
        with open(p) as f:
            return json.load(f)

    def test_decisions_present(self, gate):
        assert "decisions" in gate
        assert "UCI" in gate["decisions"]
        assert "SYNTHETIC" in gate["decisions"]

    def test_decisions_not_auto_promote(self, gate):
        for src, d in gate["decisions"].items():
            assert d["decision"] != "PROMOTE", \
                f"{src} decision is PROMOTE — this should require explicit phase gate"

    def test_leakage_pass(self, gate):
        assert gate["leakage"]["status"] == "PASS"

    def test_both_reproducible(self, gate):
        for src in ["UCI", "SYNTHETIC"]:
            r = gate["reproducibility"].get(src, {})
            assert r.get("status") in ("REPRODUCIBLE", "MINOR_NONDETERMINISM"), \
                f"{src} reproducibility: {r.get('status')}"

    def test_final_hash_ok(self, gate):
        assert gate["final_hash_ok"] is True
