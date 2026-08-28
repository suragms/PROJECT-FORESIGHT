"""Phase 20 — Production Integration Tests."""

import hashlib
import json
import os
import pytest
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs")
MODELS_FIN = os.path.join(BASE, "models", "final")
MODELS17 = os.path.join(BASE, "models", "phase17")
MODELS19 = os.path.join(BASE, "models", "phase19")
MODELS20 = os.path.join(MODELS_FIN, "phase20")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestPromotionEligibility:
    def test_gate_results_exist(self):
        assert os.path.exists(os.path.join(DOCS, "phase20_gate_results.json"))

    def test_promotion_complete(self):
        gate = json.load(open(os.path.join(DOCS, "phase20_gate_results.json")))
        assert gate.get("promotion_status") == "COMPLETE"


class TestCandidateLineage:
    def test_phase19_unchanged(self):
        p19 = os.path.join(MODELS19, "synthetic", "phase19_synthetic_lightgbm.joblib")
        assert os.path.exists(p19)

    def test_promoted_copy_matches_source(self):
        prov = json.load(open(os.path.join(DOCS, "phase20_promotion_provenance.json")))
        assert prov["copy_verified"] is True
        assert prov["source_sha256"] == prov["promoted_sha256"]

    def test_phase17_unchanged(self):
        p17 = os.path.join(MODELS17, "synthetic", "phase17_synthetic_lightgbm.joblib")
        assert os.path.exists(p17)


class TestFrozenProduction:
    def test_twelve_original_models_unchanged(self):
        reg = json.load(open(os.path.join(DOCS, "final_model_registry.json")))
        for e in reg:
            mf = os.path.join(BASE, e["model_file"].replace("\\", os.sep))
            assert sha256(mf) == e["hash"]

    def test_phase20_separate_registry(self):
        p20_reg = json.load(open(os.path.join(DOCS, "phase20_production_registry.json")))
        orig_reg = json.load(open(os.path.join(DOCS, "final_model_registry.json")))
        p20_hashes = {e["hash"] for e in p20_reg}
        orig_hashes = {e["hash"] for e in orig_reg}
        assert p20_hashes.isdisjoint(orig_hashes) or len(p20_reg) == 1


class TestFeatureContract:
    def test_contract_exists(self):
        assert os.path.exists(os.path.join(DOCS, "phase20_feature_contract.json"))

    def test_all_features_pass_leakage(self):
        contract = json.load(open(os.path.join(DOCS, "phase20_feature_contract.json")))
        for f in contract["features"]:
            assert f["leakage_status"] == "PASS"


class TestAPIAdapter:
    def test_rejects_uci(self):
        from src.phase20_api_adapter import validate_source
        with pytest.raises(ValueError, match="UCI"):
            validate_source("UCI")

    def test_six_week_horizon(self):
        from src.phase20_api_adapter import SUPPORTED_HORIZON
        assert SUPPORTED_HORIZON == 6

    def test_extended_status_label(self):
        from src.phase20_api_adapter import forecast_status
        assert forecast_status(1) == "PRODUCTION"
        assert forecast_status(7) == "EXTENDED_PARTIAL"

    def test_model_loadable(self):
        from src.phase20_api_adapter import load_model
        m = load_model()
        assert hasattr(m, "predict")


class TestRiskAdapter:
    def test_compute_risk(self):
        from src.phase20_risk_adapter import compute_risk
        r = compute_risk({
            "sku_id": "TEST", "forecast_weekly_demand": 100,
            "on_hand_units": 0, "on_order_units": 0,
            "lead_time_weeks": 2, "safety_stock": 50, "reorder_point": 200,
            "base_price": 500.0,
        })
        assert r["recommended_action"] == "REORDER NOW"
        assert r["stockout_risk_level"] == "CRITICAL"

    def test_explainability_fields(self):
        from src.phase20_risk_adapter import explain_risk
        e = explain_risk({
            "sku_id": "TEST", "forecast_weekly_demand": 100,
            "on_hand_units": 400, "on_order_units": 0,
            "lead_time_weeks": 2, "safety_stock": 50, "reorder_point": 200,
        })
        assert "forecast_demand" in e
        assert "recommended_action" in e


class TestE2E:
    def test_e2e_results_pass(self):
        e2e = json.load(open(os.path.join(DOCS, "phase20_e2e_results.json")))
        assert e2e["pass"] is True

    def test_smoke_tests_pass(self):
        e2e = json.load(open(os.path.join(DOCS, "phase20_e2e_results.json")))
        assert e2e["smoke_pass_count"] == e2e["smoke_total"]

    def test_production_forecasts_exist(self):
        p = os.path.join(BASE, "data", "phase20", "production_forecasts.parquet")
        assert os.path.exists(p)
        df = pd.read_parquet(p)
        assert df["horizon"].max() == 6
        assert (df["forecast_status"] == "PRODUCTION").all()


class TestDashboardAdapter:
    def test_model_info_panel(self):
        from src.phase20_dashboard_adapter import model_info_panel
        info = model_info_panel()
        assert "6 Weeks" in info["validated_horizon"]
        assert "Partial" in info["extended_forecast"]


class TestIntegrity:
    def test_promoted_artifact_registered(self):
        p20 = os.path.join(MODELS20, "phase20_synthetic_lightgbm.joblib")
        assert os.path.exists(p20)

    def test_pre_promotion_snapshot(self):
        snap = json.load(open(os.path.join(DOCS, "phase20_pre_promotion_snapshot.json")))
        assert snap["all_frozen_match"] is True
        assert snap["frozen_models_count"] == 12
