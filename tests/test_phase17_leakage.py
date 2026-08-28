"""Phase 17 — Leakage Tests."""
import os
import json
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P17_FEAT = os.path.join(BASE_DIR, "data", "phase17", "features")


class TestLeakageAudit:
    @pytest.fixture
    def audit(self):
        p = os.path.join(P17_FEAT, "leakage_audit.json")
        if not os.path.exists(p):
            pytest.skip("Leakage audit not found")
        with open(p) as f:
            return json.load(f)

    def test_no_leakage_failures(self, audit):
        fails = [a for a in audit if a.get("leakage_status") == "FAIL"]
        assert len(fails) == 0, f"Leakage detected: {[f['feature'] for f in fails]}"

    def test_all_features_audited(self, audit):
        assert len(audit) > 0

    def test_lag_features_shifted(self, audit):
        for a in audit:
            if a["feature"].startswith("lag_"):
                lag_val = int(a["feature"].split("_")[1])
                assert lag_val >= 1, f"lag_{lag_val} is not shifted"
