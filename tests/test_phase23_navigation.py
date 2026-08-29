"""Phase 23 — Unified navigation tests."""

import hashlib
import importlib
import json
import os

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestNavigationConfig:
    def test_nav_groups_exist(self):
        from dashboard.navigation import NAV_GROUPS, all_nav_items
        assert len(NAV_GROUPS) >= 7
        assert len(all_nav_items()) >= 25

    def test_all_page_keys_unique(self):
        from dashboard.navigation import all_nav_items
        keys = [item.key for item in all_nav_items()]
        assert len(keys) == len(set(keys))


class TestComponents:
    def test_sidebar_import(self):
        import dashboard.components.sidebar as sidebar
        assert hasattr(sidebar, "render_navigation")

    def test_status_badge(self):
        from dashboard.components.status_cards import status_badge
        assert "status-pass" in status_badge("PASS")
        assert "status-pending" in status_badge("PENDING ACTUALS")

    def test_data_loader_import(self):
        from dashboard.components.data_loader import load_feature_contract
        contract = load_feature_contract()
        assert contract["feature_count"] == 45


class TestPages:
    @pytest.mark.parametrize("module_name", [
        "dashboard.pages.home",
        "dashboard.pages.executive",
        "dashboard.pages.forecasting",
        "dashboard.pages.inventory",
        "dashboard.pages.analytics",
        "dashboard.pages.ml",
        "dashboard.pages.monitoring",
        "dashboard.pages.system",
    ])
    def test_page_module_imports(self, module_name):
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "render")

    def test_app_entry_import(self):
        assert os.path.exists(os.path.join(BASE, "app.py"))


class TestIntegrations:
    def test_phase20_adapter(self):
        from src.phase20_dashboard_adapter import dashboard_bundle
        bundle = dashboard_bundle()
        assert "forecasts" in bundle
        assert "risk" in bundle

    def test_phase21_monitoring_artifacts(self):
        path = os.path.join(BASE, "data", "phase21", "monitoring", "monitoring_summary.json")
        assert os.path.exists(path)

    def test_phase22_adapter(self):
        from src.phase22_executive_adapter import executive_bundle
        data = executive_bundle()
        assert data["project_name"] == "PROJECT FORESIGHT"
        assert data["production_performance"] == "PENDING ACTUALS"


class TestBackwardCompatibility:
    def test_legacy_dashboards_exist(self):
        for name in ["phase20_production.py", "phase21_monitoring.py", "phase22_executive_dashboard.py"]:
            assert os.path.exists(os.path.join(BASE, "dashboard", name))

    def test_api_routes_exist(self):
        assert os.path.exists(os.path.join(BASE, "src", "api", "phase20_routes.py"))
        assert os.path.exists(os.path.join(BASE, "src", "api", "phase21_routes.py"))


class TestIntegrity:
    def test_frozen_models_unchanged(self):
        reg = json.load(open(os.path.join(DOCS, "final_model_registry.json")))
        for e in reg:
            mf = os.path.join(BASE, e["model_file"].replace("\\", os.sep))
            assert sha256(mf) == e["hash"]

    def test_phase20_unchanged(self):
        reg = json.load(open(os.path.join(DOCS, "phase20_production_registry.json")))
        p20 = os.path.join(BASE, "models", "final", "phase20", "phase20_synthetic_lightgbm.joblib")
        assert sha256(p20) == reg[0]["hash"]


class TestDocumentation:
    def test_phase23_doc_exists(self):
        assert os.path.exists(os.path.join(DOCS, "phase23_unified_navigation.md"))

    def test_public_frontend_has_no_demo_credentials(self):
        app_js_path = os.path.join(BASE, "public", "js", "app.js")
        app_js = open(app_js_path, encoding="utf-8").read()
        forbidden = [
            "Foresight2026",
            "foresight.ai",
            "signInWithPreset",
            "Quick Demo Roles",
            "demo_token_",
            "local demo credentials",
        ]
        for token in forbidden:
            assert token not in app_js, f"public/js/app.js must not expose {token!r}"
