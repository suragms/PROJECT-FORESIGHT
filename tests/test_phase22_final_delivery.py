"""Phase 22 — Final delivery tests."""

import hashlib
import json
import os
import importlib.util

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


REQUIRED_DOCS = [
    "phase22_system_architecture.md",
    "phase22_api_documentation.md",
    "phase22_deployment_guide.md",
    "phase22_quick_start.md",
    "phase22_user_guide.md",
    "phase22_technical_documentation.md",
    "phase22_business_value.md",
    "phase22_model_card.md",
    "phase22_dataset_documentation.md",
    "PROJECT_FORESIGHT_FINAL_REPORT.md",
    "phase22_future_roadmap.md",
    "phase22_demo_script.md",
    "phase22_project_structure.md",
    "phase22_zidio_submission_checklist.md",
]


class TestDocumentation:
    @pytest.mark.parametrize("doc", REQUIRED_DOCS)
    def test_doc_exists(self, doc):
        assert os.path.exists(os.path.join(DOCS, doc)), f"Missing {doc}"

    def test_architecture_has_mermaid(self):
        with open(os.path.join(DOCS, "phase22_system_architecture.md"), encoding="utf-8") as f:
            content = f.read()
        assert "```mermaid" in content

    def test_api_docs_cover_phase20_and_21(self):
        with open(os.path.join(DOCS, "phase22_api_documentation.md"), encoding="utf-8") as f:
            content = f.read()
        assert "/phase20/forecast" in content
        assert "/phase21/health" in content


class TestExecutiveDashboard:
    def test_dashboard_imports(self):
        path = os.path.join(BASE, "dashboard", "phase22_executive_dashboard.py")
        spec = importlib.util.spec_from_file_location("phase22_exec", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    def test_executive_adapter(self):
        from src.phase22_executive_adapter import executive_bundle
        bundle = executive_bundle()
        assert bundle["project_name"] == "PROJECT FORESIGHT"
        assert bundle["production_performance"] == "PENDING ACTUALS"
        assert bundle["validation_label"] == "VALIDATION / BACKTEST"


class TestIntegrity:
    def test_integrity_snapshot(self):
        from src.phase22_final_audit import record_integrity_snapshot
        snap = record_integrity_snapshot()
        assert snap["frozen_models_12_unchanged"] is True
        assert snap["phase20_production_unchanged"] is True
        assert os.path.exists(os.path.join(DOCS, "phase22_integrity_snapshot.json"))

    def test_frozen_models_unchanged(self):
        reg = json.load(open(os.path.join(DOCS, "final_model_registry.json")))
        for e in reg:
            mf = os.path.join(BASE, e["model_file"].replace("\\", os.sep))
            assert sha256(mf) == e["hash"]

    def test_phase20_model_unchanged(self):
        reg = json.load(open(os.path.join(DOCS, "phase20_production_registry.json")))
        p20 = os.path.join(BASE, "models", "final", "phase20", "phase20_synthetic_lightgbm.joblib")
        assert sha256(p20) == reg[0]["hash"]


class TestFinalAudit:
    def test_audit_passes(self):
        from src.phase22_final_audit import run_final_audit
        audit = run_final_audit()
        assert audit["status"] == "PASS"
        assert audit["delivery_status"] == "PROJECT DELIVERY READY"
        assert os.path.exists(os.path.join(DOCS, "phase22_final_audit.json"))

    def test_phase17_artifact_exists(self):
        p17 = os.path.join(BASE, "models", "phase17", "synthetic", "phase17_synthetic_lightgbm.joblib")
        assert os.path.exists(p17)

    def test_phase19_artifact_exists(self):
        p19 = os.path.join(BASE, "models", "phase19", "synthetic", "phase19_synthetic_lightgbm.joblib")
        assert os.path.exists(p19)


class TestProductionArtifacts:
    def test_phase20_contract(self):
        contract = json.load(open(os.path.join(DOCS, "phase20_feature_contract.json")))
        assert contract["feature_count"] == 45

    def test_phase21_monitoring_exists(self):
        assert os.path.exists(
            os.path.join(BASE, "data", "phase21", "monitoring", "monitoring_summary.json")
        )

    def test_dashboards_exist(self):
        for name in ["phase20_production.py", "phase21_monitoring.py", "phase22_executive_dashboard.py"]:
            assert os.path.exists(os.path.join(BASE, "dashboard", name))
