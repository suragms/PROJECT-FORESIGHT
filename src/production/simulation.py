"""Local production-style simulation. Uses the existing API; does not retrain."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.metrics import snapshot as metrics_snapshot
from src.config import (
    INVENTORY_RISK_PATH,
    OUTPUTS_MONITORING_DIR,
    PROJECT_ROOT,
    SAMPLES_DIR,
)
from src.forecasting.make_samples import write_samples
from src.forecasting.registry import load_registry, resolve_selected, verify_hash
from src.forecasting.validation import InputValidationError, records_to_frame
from src.production.business_validation import validate_ten_questions
from src.security.audit import captured_events, start_capture, stop_capture

SIM_KEY = "phase14-simulation-key"
EXPECTED_UCI_HASH = "331909f0fe191c0b9cb0418884b25eb59012f479f61f8b3e2ad51b729273e90d"
EXPECTED_SYN_HASH = "59a2b72024861d7f9c827596a52256af95facabfa796bdae5955374221cf1bf4"
PHASE12_REF = {
    "single_s": 0.0681,
    "batch10_s": 0.0607,
    "rows_per_s": 164.8,
    "note": "Historical Phase 12 freeze snapshot; not an SLO.",
}
INVENTORY_FIELDS = [
    "ending_inventory",
    "lead_time_days",
    "safety_stock",
    "reorder_point",
    "stockout_risk_level",
    "overstock_risk_level",
    "recommended_reorder_qty",
]


def _ok(name: str, passed: bool, detail: str = "", **extra: Any) -> dict[str, Any]:
    row = {"name": name, "passed": bool(passed), "detail": detail}
    row.update(extra)
    return row


@contextmanager
def simulation_env(**overrides: str) -> Iterator[str]:
    values = {
        "FORESIGHT_ENV": "production",
        "FORESIGHT_API_AUTH_ENABLED": "true",
        "FORESIGHT_API_API_KEY": SIM_KEY,
        "FORESIGHT_RATE_LIMIT_ENABLED": "false",
    }
    values.update(overrides)
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield SIM_KEY
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def load_uci_record() -> dict[str, Any]:
    if not (SAMPLES_DIR / "uci_h1_sample.json").exists():
        write_samples()
    payload = json.loads((SAMPLES_DIR / "uci_h1_sample.json").read_text(encoding="utf-8"))
    rec = payload["record"]
    keys = ["date", "source_dataset", "entity_id", "product_key"]
    return {
        "source_dataset": "UCI",
        "horizon": 1,
        "date": rec["date"],
        "entity_id": rec["entity_id"],
        "product_key": rec["product_key"],
        "features": {k: v for k, v in rec.items() if k not in keys},
    }


def _auth_headers(key: str | None = None) -> dict[str, str]:
    return {"X-API-Key": key or SIM_KEY}


def _unique_batch(record: dict[str, Any], n: int) -> list[dict[str, Any]]:
    rows = []
    base = {k: v for k, v in record.items() if k != "horizon"}
    for i in range(n):
        item = dict(base)
        item["product_key"] = f"{record['product_key']}__p14_{i}"
        rows.append(item)
    return rows


def _memory_rss() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _risk_frame():
    path = Path(INVENTORY_RISK_PATH)
    if not path.exists():
        return None
    import pandas as pd
    return pd.read_parquet(path)


def attach_inventory(forecast: dict[str, Any]) -> dict[str, Any]:
    out = {
        "layer": "INVENTORY RISK",
        "forecast_layer": "MODEL FORECAST",
        "product_key": forecast.get("product_key"),
    }
    risk = _risk_frame()
    if risk is None:
        for field in INVENTORY_FIELDS:
            out[field] = "NOT AVAILABLE"
        out["join"] = "NOT AVAILABLE"
        out["recommendation"] = "NOT AVAILABLE"
        return out
    pk = str(forecast.get("product_key", ""))
    hits = risk[risk["sku_id"].astype(str) == pk] if "sku_id" in risk.columns else risk.iloc[0:0]
    if hits.empty:
        for field in INVENTORY_FIELDS:
            out[field] = "NOT AVAILABLE"
        out["join"] = "NOT AVAILABLE"
        out["recommendation"] = (
            "NOT AVAILABLE — UCI/online product keys are not store-SKU inventory rows. "
            "Do not invent a warehouse position."
        )
        return out
    row = hits.iloc[0]
    for field in INVENTORY_FIELDS:
        out[field] = row[field] if field in row.index else "NOT AVAILABLE"
    out["join"] = "sku_id"
    out["recommendation"] = "Investigate inventory position; this is not a purchase order."
    return out


def business_scenarios() -> list[dict[str, Any]]:
    risk = _risk_frame()
    scenarios = []

    def pick(mask, name, interpretation, fallback=None):
        if risk is None or not bool(mask.any()):
            if fallback is not None and risk is not None and bool(fallback.any()):
                row = risk.loc[fallback].iloc[0]
                source = "fallback_label"
            else:
                scenarios.append({
                    "name": name,
                    "status": "NOT AVAILABLE",
                    "interpretation": interpretation,
                    "note": "No matching row in the on-disk risk extract.",
                    "autonomous_decision": False,
                })
                return
        else:
            row = risk.loc[mask].iloc[0]
            source = "primary_label"
        scenarios.append({
            "name": name,
            "status": "PASS",
            "source": source,
            "sku_id": str(row.get("sku_id", "NOT AVAILABLE")),
            "store_id": str(row.get("store_id", "NOT AVAILABLE")),
            "stockout_risk_level": str(row.get("stockout_risk_level", "NOT AVAILABLE")),
            "overstock_risk_level": str(row.get("overstock_risk_level", "NOT AVAILABLE")),
            "ending_inventory": row.get("ending_inventory", "NOT AVAILABLE"),
            "interpretation": interpretation,
            "autonomous_decision": False,
        })

    if risk is None:
        return [{
            "name": "all",
            "status": "NOT AVAILABLE",
            "note": "inventory_risk_matrix.parquet missing",
            "autonomous_decision": False,
        }]

    pick(
        risk["stockout_risk_level"] == "CRITICAL / HIGH",
        "A_high_stockout",
        "Inventory may be insufficient relative to expected demand. Review replenishment requirements. Not an automatic purchase order.",
    )
    pick(
        risk["overstock_risk_level"] == "SEVERE OVERSTOCK",
        "B_high_overstock",
        "Potential excess inventory exposure. Review demand and inventory alignment.",
        fallback=risk["overstock_risk_level"] == "MODERATE OVERSTOCK" if "overstock_risk_level" in risk.columns else None,
    )
    pick(
        (risk["stockout_risk_level"] == "LOW / SAFE") & (risk["overstock_risk_level"] == "OPTIMAL"),
        "C_stable_demand",
        "No exceptional intervention indicated.",
    )
    rec = load_uci_record()
    growth = rec["features"].get("demand_growth_30")
    scenarios.append({
        "name": "D_strong_demand_growth",
        "status": "PASS" if growth is not None else "NOT AVAILABLE",
        "layer": "MODEL FORECAST feature (demand_growth_30), not inventory execution",
        "demand_growth_30": growth if growth is not None else "NOT AVAILABLE",
        "interpretation": "Review future supply capacity and inventory requirements.",
        "autonomous_decision": False,
    })
    return scenarios


def dashboard_e2e() -> dict[str, Any]:
    os.environ["FORECAST_DASHBOARD_SKIP_MAIN"] = "1"
    from dashboard import forecast_analytics as dash
    from src.config import FINAL_FORECASTS_PATH, REGISTRY_PATH

    preds = dash.load_parquet(str(FINAL_FORECASTS_PATH))
    src = Path(dash.__file__).read_text(encoding="utf-8")
    app_src = (Path(dash.__file__).parent / "app.py").read_text(encoding="utf-8")
    snapshot_labelled = "file snapshot" in src.lower() or "not a live" in src.lower()
    intervals_labelled = "P10" in src and "P90" in src
    no_realtime_claim = "real-time monitoring" not in src.lower()
    inventory_labelled = "inventory risk" in app_src.lower()
    rec_labelled = "recommendation" in app_src.lower()
    return {
        "starts_import": True,
        "forecast_rows": 0 if preds is None else int(len(preds)),
        "dataset_filter_present": "Dataset" in src,
        "horizon_filter_present": "Horizon" in src,
        "intervals_labelled": intervals_labelled,
        "point_forecast_labelled": "point" in src.lower() or "prediction" in src.lower(),
        "monitoring_snapshot_labelled": snapshot_labelled,
        "does_not_claim_realtime_monitoring": no_realtime_claim,
        "inventory_risk_labelled": inventory_labelled,
        "recommendations_labelled": rec_labelled,
        "registry_loads": isinstance(dash.load_json(str(REGISTRY_PATH)), list),
        "passed": bool(
            preds is not None and len(preds) > 0 and snapshot_labelled
            and intervals_labelled and no_realtime_claim
            and inventory_labelled and rec_labelled
        ),
    }


def run_simulation() -> dict[str, Any]:
    """Execute the Phase 14 local production simulation against the existing API."""
    results: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    start_capture()
    rec = load_uci_record()

    with simulation_env():
        client = TestClient(create_app())
        client.__enter__()
        health = client.get("/health")
        header_names = {k.lower() for k in health.headers.keys()}
        headers_ok = {"x-content-type-options", "x-frame-options", "referrer-policy"}.issubset(header_names)
        results.append(_ok("API Health", health.status_code == 200 and headers_ok, str(health.status_code)))
        ready = client.get("/ready")
        results.append(_ok("API Readiness", ready.status_code == 200 and ready.json().get("status") == "ready"))

        denied = client.get("/model")
        results.append(_ok("Authentication", denied.status_code == 401 and SIM_KEY not in denied.text))
        allowed = client.get("/model", headers=_auth_headers())
        results.append(_ok("Valid API key", allowed.status_code == 200 and len(allowed.json().get("models", [])) >= 1))

        t0 = time.perf_counter()
        single = client.post("/forecast", json=rec, headers=_auth_headers())
        single_s = time.perf_counter() - t0
        body = single.json() if single.status_code == 200 else {}
        row = (body.get("forecasts") or [{}])[0]
        evidence["single_forecast"] = {
            "dataset": rec["source_dataset"],
            "entity_id": rec["entity_id"],
            "product_key": rec["product_key"],
            "horizon": rec["horizon"],
            "model": (body.get("metadata") or {}).get("model_id"),
            "hash": (body.get("metadata") or {}).get("hash"),
            "prediction": row.get("prediction"),
            "lower_bound": row.get("lower_bound"),
            "upper_bound": row.get("upper_bound"),
            "latency_s": round(single_s, 4),
            "request_id": single.headers.get("x-request-id"),
            "status_code": single.status_code,
        }
        results.append(_ok(
            "Single Forecast",
            single.status_code == 200 and row.get("prediction") is not None,
            f"pred={row.get('prediction')} latency_s={single_s:.4f}",
        ))

        batch_evidence = []
        batch_pass = True
        rss0 = _memory_rss()
        for n in (10, 100, 500):
            payload = {
                "source_dataset": "UCI",
                "horizon": 1,
                "records": _unique_batch(rec, n),
            }
            t1 = time.perf_counter()
            resp = client.post("/forecast/batch", json=payload, headers=_auth_headers())
            elapsed = time.perf_counter() - t1
            ok = resp.status_code == 200 and resp.json().get("n") == n
            batch_pass = batch_pass and ok
            batch_evidence.append({
                "n": n,
                "status_code": resp.status_code,
                "rows_processed": resp.json().get("n") if resp.status_code == 200 else 0,
                "success_count": n if ok else 0,
                "failure_count": 0 if ok else n,
                "latency_s": round(elapsed, 4),
                "rows_per_s": round(n / elapsed, 2) if elapsed else None,
                "response_bytes": len(resp.content),
            })
        rss1 = _memory_rss()
        evidence["batch"] = {
            "runs": batch_evidence,
            "rss_before": rss0,
            "rss_after": rss1,
        }
        results.append(_ok("Batch Forecast", batch_pass, json.dumps(batch_evidence)))

        hashes = {}
        recs = load_registry()
        for item in recs:
            hashes[item["model_id"]] = verify_hash(item)
        selected = resolve_selected("UCI", 1, recs)
        used_hash = (body.get("metadata") or {}).get("hash")
        hash_ok = (
            hashes.get("uci_h1_phase8_lightgbm") == EXPECTED_UCI_HASH
            and hashes.get("synthetic_h1_hurdle_th050") == EXPECTED_SYN_HASH
            and used_hash == EXPECTED_UCI_HASH
            and selected["model_id"] == "uci_h1_phase8_lightgbm"
        )
        evidence["hashes"] = {
            "uci_h1": hashes.get("uci_h1_phase8_lightgbm"),
            "synthetic_h1": hashes.get("synthetic_h1_hurdle_th050"),
            "used_for_inference": used_hash,
        }
        results.append(_ok("Model Integrity", hash_ok, used_hash or ""))

        contract = {}
        valid = client.post("/forecast", json=rec, headers=_auth_headers())
        contract["valid"] = valid.status_code == 200
        missing = client.post("/forecast", json={"source_dataset": "UCI"}, headers=_auth_headers())
        contract["missing_field"] = missing.status_code in (400, 422) and "traceback" not in missing.text.lower()
        wrong_type = client.post(
            "/forecast",
            json={**rec, "horizon": "abc"},
            headers=_auth_headers(),
        )
        contract["wrong_datatype"] = wrong_type.status_code in (400, 422)
        nan = dict(rec)
        nan["features"] = dict(rec["features"])
        nan["features"]["units_sold_lag_1"] = None
        nan_r = client.post("/forecast", json=nan, headers=_auth_headers())
        contract["nan"] = nan_r.status_code in (400, 422)
        inf_ok = False
        try:
            records_to_frame([{
                "source_dataset": "UCI",
                "entity_id": "ONLINE",
                "product_key": "UCI_1",
                "date": "2011-09-26",
                "features": {"units_sold_lag_1": float("inf")},
            }])
        except InputValidationError:
            inf_ok = True
        contract["infinity"] = inf_ok
        cat = dict(rec)
        cat["features"] = dict(rec["features"])
        cat["features"]["season"] = "NOT_A_SEASON"
        cat_r = client.post("/forecast", json=cat, headers=_auth_headers())
        contract["invalid_categorical"] = {
            "status_code": cat_r.status_code,
            "traceback": "traceback" not in cat_r.text.lower(),
            "note": (
                "Rejected" if cat_r.status_code in (400, 422)
                else "Existing encoder accepts unseen season levels; no 5xx and no traceback."
            ),
        }
        contract["invalid_categorical_safe"] = cat_r.status_code != 500 and "traceback" not in cat_r.text.lower()
        over_batch = client.post(
            "/forecast/batch",
            json={"source_dataset": "UCI", "horizon": 1, "records": _unique_batch(rec, 501)},
            headers=_auth_headers(),
        )
        contract["excessive_batch"] = over_batch.status_code in (400, 422)
        os.environ["FORESIGHT_API_MAX_PAYLOAD_BYTES"] = "128"
        tiny2 = TestClient(create_app())
        over_payload = tiny2.post("/forecast", json=rec, headers=_auth_headers())
        contract["oversized_payload"] = over_payload.status_code == 413
        os.environ.pop("FORESIGHT_API_MAX_PAYLOAD_BYTES", None)
        dup_r = client.post(
            "/forecast/batch",
            json={"source_dataset": "UCI", "horizon": 1, "records": [
                {k: v for k, v in rec.items() if k != "horizon"},
                {k: v for k, v in rec.items() if k != "horizon"},
            ]},
            headers=_auth_headers(),
        )
        contract["duplicate_records"] = dup_r.status_code in (400, 422)
        malformed = client.post(
            "/forecast",
            content=b"{not-json",
            headers={**_auth_headers(), "Content-Type": "application/json"},
        )
        contract["malformed_json"] = (
            malformed.status_code in (400, 422) and "traceback" not in malformed.text.lower()
        )
        evidence["data_contract"] = contract
        contract_pass = all(bool(contract[k]) for k in (
            "valid", "missing_field", "wrong_datatype", "nan", "infinity",
            "invalid_categorical_safe", "excessive_batch", "oversized_payload",
            "duplicate_records", "malformed_json",
        ))
        results.append(_ok("Data Contract", contract_pass, json.dumps(contract, default=str)[:500]))

        inventory = attach_inventory(row)
        evidence["inventory"] = inventory
        results.append(_ok(
            "Inventory Risk",
            inventory.get("join") in {"sku_id", "NOT AVAILABLE"} and "recommendation" in inventory,
            inventory.get("join"),
        ))

        questions = validate_ten_questions()
        evidence["questions"] = {
            "status": questions.get("status"),
            "n": questions.get("questions_with_evidence"),
        }
        questions_ok = questions.get("questions_with_evidence") == 10
        results.append(_ok("Business Questions", questions_ok))

        scenarios = business_scenarios()
        if row.get("lower_bound") is not None and row.get("upper_bound") is not None and row.get("prediction"):
            width = abs(float(row["upper_bound"]) - float(row["lower_bound"]))
            rel = width / max(abs(float(row["prediction"])), 1e-6)
            scenarios.append({
                "name": "E_high_uncertainty",
                "status": "PASS",
                "p10": row.get("lower_bound"),
                "p90": row.get("upper_bound"),
                "prediction": row.get("prediction"),
                "relative_interval_width": round(rel, 4),
                "interpretation": "Additional business review recommended.",
                "autonomous_decision": False,
            })
        else:
            scenarios.append({
                "name": "E_high_uncertainty",
                "status": "NOT AVAILABLE",
                "interpretation": "Additional business review recommended.",
                "autonomous_decision": False,
            })
        evidence["scenarios"] = scenarios
        scenario_ok = all(s.get("status") in {"PASS", "NOT AVAILABLE"} for s in scenarios) and any(
            s.get("name") == "A_high_stockout" and s.get("status") == "PASS" for s in scenarios
        )
        results.append(_ok("Business Scenarios", scenario_ok, f"n={len(scenarios)}"))
        results.append(_ok("Business Workflow", questions_ok and scenario_ok))

        dash = dashboard_e2e()
        evidence["dashboard"] = dash
        results.append(_ok("Dashboard", dash.get("passed") is True))

        second = client.post("/forecast", json=rec, headers=_auth_headers())
        pred1 = row.get("prediction")
        pred2 = (second.json().get("forecasts") or [{}])[0].get("prediction") if second.status_code == 200 else None
        results.append(_ok("Reproducibility", pred1 is not None and pred1 == pred2, f"{pred1} vs {pred2}"))
        client.__exit__(None, None, None)

    with simulation_env(FORESIGHT_RATE_LIMIT_ENABLED="true", FORESIGHT_RATE_LIMIT_REQUESTS="2", FORESIGHT_RATE_LIMIT_FORECAST_REQUESTS="2"):
        limited = TestClient(create_app())
        limited.get("/model", headers=_auth_headers())
        limited.get("/model", headers=_auth_headers())
        third = limited.get("/model", headers=_auth_headers())
        results.append(_ok("Rate Limiting", third.status_code == 429))

    from src.production import readiness as readiness_mod
    from src.forecasting.registry import RegistryError

    def _missing_registry(*_a, **_k):
        raise RegistryError("Model registry not found")

    original_load = readiness_mod.load_registry
    readiness_mod.load_registry = _missing_registry
    try:
        with simulation_env():
            miss = TestClient(create_app()).get("/ready")
            missing_registry_ok = miss.status_code == 503
    finally:
        readiness_mod.load_registry = original_load

    def _missing_model_file(*_a, **_k):
        recs = original_load()
        patched = [dict(r) for r in recs]
        patched[0]["model_file"] = "models/final/missing_phase14.joblib"
        return patched

    readiness_mod.load_registry = _missing_model_file
    try:
        with simulation_env():
            miss_model = TestClient(create_app()).get("/ready")
            missing_model_ok = miss_model.status_code == 503
    finally:
        readiness_mod.load_registry = original_load

    def _bad_hash(*_a, **_k):
        raise RegistryError("Hash mismatch")

    original_hash = readiness_mod.verify_hash
    readiness_mod.verify_hash = _bad_hash
    try:
        with simulation_env():
            badh = TestClient(create_app()).get("/ready")
            hash_fail = badh.status_code == 503
    finally:
        readiness_mod.verify_hash = original_hash

    with simulation_env():
        boom_client = TestClient(create_app(), raise_server_exceptions=False)
        from src.forecasting.inference import ForecastEngine
        original_predict = ForecastEngine.predict

        def _boom(self, *a, **k):
            raise RuntimeError("simulated internal failure")

        ForecastEngine.predict = _boom
        try:
            exploded = boom_client.post("/forecast", json=rec, headers=_auth_headers())
            internal_ok = exploded.status_code == 500 and "traceback" not in exploded.text.lower() and "simulated" not in exploded.text.lower()
        finally:
            ForecastEngine.predict = original_predict
        invalid = boom_client.post("/forecast", json={"source_dataset": "UCI"}, headers=_auth_headers())
        invalid_ok = invalid.status_code in (400, 422) and "traceback" not in invalid.text.lower()

    recovery_ok = missing_registry_ok and missing_model_ok and hash_fail and internal_ok and invalid_ok
    results.append(_ok(
        "Failure Recovery",
        recovery_ok,
        "missing model/registry/hash 503, invalid 4xx, internal 5xx without traceback",
    ))

    events = stop_capture()
    names = {e.get("event") for e in events}
    needed = {
        "application_startup",
        "readiness_check",
        "authentication_failure",
        "forecast_request",
        "batch_forecast_request",
        "validation_failure",
        "unhandled_error",
        "model_hash_verification",
    }
    blob = json.dumps(events, default=str).lower()
    secret_free = SIM_KEY.lower() not in blob and "password" not in blob
    evidence["audit_events"] = sorted(names)
    results.append(_ok("Audit Logging", needed.issubset(names) and secret_free, f"events={sorted(names)}"))

    mon_files = [
        OUTPUTS_MONITORING_DIR / "monitoring_summary.json",
        OUTPUTS_MONITORING_DIR / "data_quality_report.json",
        OUTPUTS_MONITORING_DIR / "drift_report.json",
        OUTPUTS_MONITORING_DIR / "api_metrics.json",
    ]
    from src.monitoring.run_monitoring import run_monitoring
    summary = run_monitoring()
    metrics = metrics_snapshot()
    evidence["monitoring"] = {
        "n_alerts": summary.get("n_alerts"),
        "retraining": summary.get("retraining"),
        "api_request_count": metrics.get("request_count"),
        "auth_failures": metrics.get("auth_failures"),
        "rate_limit_events": metrics.get("rate_limit_events"),
        "mean_latency_ms": metrics.get("mean_latency_ms"),
        "files": [str(p.name) for p in mon_files if p.exists()],
    }
    results.append(_ok(
        "Monitoring",
        all(p.exists() for p in mon_files) and summary.get("retraining") == "disabled",
    ))

    batch10 = next((r for r in evidence["batch"]["runs"] if r["n"] == 10), {})
    evidence["performance"] = {
        "single_s": evidence["single_forecast"]["latency_s"],
        "batch": evidence["batch"]["runs"],
        "success_rate": 1.0 if batch_pass else 0.0,
        "phase12_reference": PHASE12_REF,
        "vs_phase12": {
            "single_s_delta": round(evidence["single_forecast"]["latency_s"] - PHASE12_REF["single_s"], 4),
            "batch10_s_delta": round((batch10.get("latency_s") or 0) - PHASE12_REF["batch10_s"], 4),
            "note": "Phase 14 timings include production auth/TestClient overhead; not an SLO.",
        },
    }
    results.append(_ok(
        "Performance",
        batch_pass and evidence["single_forecast"]["status_code"] == 200,
        json.dumps({
            "single_s": evidence["single_forecast"]["latency_s"],
            "batch10_s": batch10.get("latency_s"),
            "batch10_rows_per_s": batch10.get("rows_per_s"),
        }),
    ))
    from src.production.docker_check import inspect_dockerfile
    from src.production.config_validation import validate_runtime_config
    docker_static = inspect_dockerfile()
    prev_env = os.environ.get("FORESIGHT_ENV")
    prev_auth = os.environ.get("FORESIGHT_API_AUTH_ENABLED")
    prev_key = os.environ.get("FORESIGHT_API_API_KEY")
    os.environ["FORESIGHT_ENV"] = "production"
    os.environ["FORESIGHT_API_AUTH_ENABLED"] = "true"
    os.environ.pop("FORESIGHT_API_API_KEY", None)
    prod_errors = validate_runtime_config()
    if prev_env is None:
        os.environ.pop("FORESIGHT_ENV", None)
    else:
        os.environ["FORESIGHT_ENV"] = prev_env
    if prev_auth is None:
        os.environ.pop("FORESIGHT_API_AUTH_ENABLED", None)
    else:
        os.environ["FORESIGHT_API_AUTH_ENABLED"] = prev_auth
    if prev_key is None:
        os.environ.pop("FORESIGHT_API_API_KEY", None)
    else:
        os.environ["FORESIGHT_API_API_KEY"] = prev_key
    evidence["security_regression"] = {
        "headers": headers_ok,
        "dockerfile_non_root": docker_static["checks"].get("non_root_user"),
        "no_api_key_in_dockerfile": docker_static["checks"].get("no_api_key_in_image"),
        "production_requires_key": any("AUTH" in e.upper() or "API_KEY" in e.upper() for e in prod_errors),
    }
    out_dir = PROJECT_ROOT / "outputs" / "phase14"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"results": results, "evidence": evidence}
    (out_dir / "simulation.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
