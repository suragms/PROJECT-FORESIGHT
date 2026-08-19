"""Phase 14 end-to-end simulation tests. Frozen models are not modified."""

from __future__ import annotations

from src.production.simulation import (
    EXPECTED_UCI_HASH,
    attach_inventory,
    business_scenarios,
    dashboard_e2e,
    load_uci_record,
    run_simulation,
)
from src.forecasting.registry import load_registry, resolve_model_file, verify_hash


def test_frozen_hashes_unchanged():
    recs = load_registry()
    hashes = {r["model_id"]: verify_hash(r) for r in recs}
    assert hashes["uci_h1_phase8_lightgbm"] == EXPECTED_UCI_HASH
    rec = dict(recs[0])
    rec["model_file"] = rec["model_file"].replace("/", "\\")
    assert resolve_model_file(rec).exists()


def test_dashboard_e2e_labels():
    result = dashboard_e2e()
    assert result["passed"] is True
    assert result["monitoring_snapshot_labelled"] is True
    assert result["does_not_claim_realtime_monitoring"] is True


def test_inventory_does_not_invent_uci_position():
    rec = load_uci_record()
    joined = attach_inventory({"product_key": rec["product_key"], "prediction": 1.0})
    assert joined["join"] == "NOT AVAILABLE"
    assert joined["ending_inventory"] == "NOT AVAILABLE"


def test_stockout_scenario_exists():
    scenarios = business_scenarios()
    names = {s["name"]: s for s in scenarios}
    assert names["A_high_stockout"]["status"] == "PASS"
    assert names["A_high_stockout"]["autonomous_decision"] is False


def test_phase14_simulation_core():
    payload = run_simulation()
    by_name = {row["name"]: row for row in payload["results"]}
    for key in (
        "API Health",
        "API Readiness",
        "Authentication",
        "Single Forecast",
        "Batch Forecast",
        "Model Integrity",
        "Data Contract",
        "Reproducibility",
        "Audit Logging",
        "Performance",
        "Business Workflow",
    ):
        assert by_name[key]["passed"] is True, key
    assert payload["evidence"]["single_forecast"]["hash"] == EXPECTED_UCI_HASH
    assert payload["evidence"]["single_forecast"]["request_id"]
    blob = str(payload).lower()
    assert "phase14-simulation-key" not in blob
    assert payload["evidence"]["data_contract"]["malformed_json"] is True


def test_dockerfile_static_non_root():
    from src.production.docker_check import inspect_dockerfile

    spec = inspect_dockerfile()
    assert spec["passed"] is True
    assert spec["checks"]["non_root_user"] is True
    assert spec["checks"]["no_api_key_in_image"] is True
    assert spec["checks"]["dockerignore_env"] is True
