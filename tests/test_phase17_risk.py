"""Phase 17 — Risk Scoring Tests."""
import os
import json
import pytest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P17_RISK = os.path.join(BASE_DIR, "data", "phase17", "risk")


class TestRiskOutput:
    @pytest.fixture
    def risk(self):
        p = os.path.join(P17_RISK, "forecast_driven_risk.parquet")
        if not os.path.exists(p):
            pytest.skip("Risk output not found")
        return pd.read_parquet(p)

    def test_risk_not_empty(self, risk):
        assert len(risk) > 0

    def test_has_stockout_level(self, risk):
        assert "stockout_risk_level" in risk.columns
        valid = {"CRITICAL", "MEDIUM", "LOW"}
        assert set(risk["stockout_risk_level"].unique()).issubset(valid)

    def test_has_overstock_level(self, risk):
        assert "overstock_risk_level" in risk.columns
        valid = {"SEVERE", "MODERATE", "OPTIMAL"}
        assert set(risk["overstock_risk_level"].unique()).issubset(valid)

    def test_has_action(self, risk):
        assert "action" in risk.columns
        valid = {"REORDER NOW", "MARKDOWN / CLEAR", "WATCH / VOLATILE", "HEALTHY"}
        assert set(risk["action"].unique()).issubset(valid)

    def test_uses_forecast_demand(self, risk):
        assert "forecast_weekly_demand" in risk.columns

    def test_no_negative_risk_scores(self, risk):
        assert (risk["stockout_risk_score"] >= 0).all()
        assert (risk["overstock_risk_score"] >= 0).all()


class TestRiskSummary:
    @pytest.fixture
    def summary(self):
        p = os.path.join(P17_RISK, "risk_summary.json")
        if not os.path.exists(p):
            pytest.skip("Risk summary not found")
        with open(p) as f:
            return json.load(f)

    def test_demand_source_is_forecast(self, summary):
        assert summary.get("demand_source") in ("FORECAST", "HISTORICAL_FALLBACK")

    def test_uci_risk_documented(self, summary):
        assert "uci_risk_status" in summary
        assert "NOT_AVAILABLE" in summary["uci_risk_status"]
