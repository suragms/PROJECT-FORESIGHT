"""Phase 17 — Forecasting Tests."""
import os
import json
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P17_DIR = os.path.join(BASE_DIR, "data", "phase17")
DOCS_DIR = os.path.join(BASE_DIR, "docs")


class TestBacktestResults:
    @pytest.fixture
    def results(self):
        p = os.path.join(P17_DIR, "backtests", "backtest_results.parquet")
        if not os.path.exists(p):
            pytest.skip("Backtest results not found")
        return pd.read_parquet(p)

    def test_results_not_empty(self, results):
        assert len(results) > 0

    def test_has_required_columns(self, results):
        required = {"source_dataset", "product_key", "actual", "seasonal_naive_forecast"}
        assert required.issubset(set(results.columns))

    def test_temporal_ordering(self, results):
        for _, grp in results.groupby(["source_dataset", "product_key", "fold"]):
            weeks = pd.to_datetime(grp["forecast_week"])
            origins = pd.to_datetime(grp["forecast_origin"])
            assert (weeks > origins).all(), "Forecast week must be after origin"

    def test_no_negative_actuals(self, results):
        assert (results["actual"] >= 0).all()


class TestBaselineWAPE:
    @pytest.fixture
    def metrics(self):
        p = os.path.join(P17_DIR, "forecasts", "backtest_metrics.json")
        if not os.path.exists(p):
            pytest.skip("Metrics not found")
        with open(p) as f:
            return json.load(f)

    def test_wape_calculated(self, metrics):
        for source, m in metrics.items():
            if isinstance(m, dict) and "seasonal_naive_wape" in m:
                assert m["seasonal_naive_wape"] is not None
                assert 0 <= m["seasonal_naive_wape"] <= 200


class TestCandidateRegistry:
    @pytest.fixture
    def registry(self):
        p = os.path.join(DOCS_DIR, "phase17_candidate_model_registry.json")
        if not os.path.exists(p):
            pytest.skip("Candidate registry not found")
        with open(p) as f:
            return json.load(f)

    def test_registry_not_empty(self, registry):
        assert len(registry) > 0

    def test_all_candidates_not_production(self, registry):
        for entry in registry:
            assert entry["status"] == "candidate", \
                f"{entry['model_id']} has status '{entry['status']}', expected 'candidate'"

    def test_all_have_wape(self, registry):
        for entry in registry:
            assert "wape" in entry
            assert "baseline_wape" in entry
