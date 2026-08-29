"""API tests using FastAPI TestClient. No arbitrary model paths."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import API_MAX_BATCH, SAMPLES_DIR
from src.forecasting.make_samples import write_samples


@pytest.fixture(scope="module")
def client():
    if not (SAMPLES_DIR / "uci_h1_sample.parquet").exists():
        write_samples()
    return TestClient(create_app())


@pytest.fixture(scope="module")
def uci_record():
    if not (SAMPLES_DIR / "uci_h1_sample.json").exists():
        write_samples()
    payload = json.loads((SAMPLES_DIR / "uci_h1_sample.json").read_text(encoding="utf-8"))
    rec = payload["record"]
    keys = ["date", "source_dataset", "entity_id", "product_key"]
    features = {k: v for k, v in rec.items() if k not in keys}
    return {
        "source_dataset": "UCI",
        "horizon": 1,
        "date": rec["date"],
        "entity_id": rec["entity_id"],
        "product_key": rec["product_key"],
        "features": features,
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "timestamp" in body


def test_docs_swagger_accessible(client):
    """Swagger /docs must not use default-src 'none' CSP (breaks blank UI)."""
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger-ui" in r.text.lower()
    csp = r.headers.get("content-security-policy", "")
    assert "default-src 'none'" not in csp
    assert "cdn.jsdelivr.net" in csp


def test_openapi_json(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    body = r.json()
    assert body.get("openapi", "").startswith("3.")
    assert "/health" in body.get("paths", {})


def test_model_registry(client):
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    ids = [m["model_id"] for m in body["models"]]
    assert "uci_h1_phase8_lightgbm" in ids
    assert all(len(m["hash"]) == 64 for m in body["models"])


def test_model_filter(client):
    r = client.get("/model", params={"dataset": "UCI", "horizon": 1})
    assert r.status_code == 200
    assert r.json()["models"][0]["model_id"] == "uci_h1_phase8_lightgbm"


def test_forecast_valid(client, uci_record):
    r = client.post("/forecast", json=uci_record)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 1
    assert body["forecasts"][0]["prediction"] >= 0
    assert body["metadata"]["model_id"] == "uci_h1_phase8_lightgbm"
    assert "actual" not in body["forecasts"][0] or body["forecasts"][0]["actual"] is None


def test_forecast_invalid_schema(client):
    r = client.post("/forecast", json={"source_dataset": "UCI"})
    assert r.status_code in (400, 422)


def test_forecast_missing_feature(client, uci_record):
    bad = dict(uci_record)
    feats = dict(bad["features"])
    feats.pop("units_sold_lag_1", None)
    bad["features"] = feats
    r = client.post("/forecast", json=bad)
    assert r.status_code == 400


def test_forecast_invalid_horizon(client, uci_record):
    bad = dict(uci_record)
    bad["horizon"] = 2
    r = client.post("/forecast", json=bad)
    assert r.status_code in (400, 422)


def test_forecast_malformed_date(client, uci_record):
    bad = dict(uci_record)
    bad["date"] = "32/13/not-a-date"
    r = client.post("/forecast", json=bad)
    assert r.status_code == 400


def test_batch_valid(client, uci_record):
    rec = {k: v for k, v in uci_record.items() if k not in ("horizon",)}
    r = client.post("/forecast/batch", json={
        "source_dataset": "UCI",
        "horizon": 1,
        "records": [rec],
    })
    assert r.status_code == 200, r.text
    assert r.json()["n"] == 1


def test_batch_oversized(client, uci_record):
    rec = {k: v for k, v in uci_record.items() if k not in ("horizon",)}
    r = client.post("/forecast/batch", json={
        "source_dataset": "UCI",
        "horizon": 1,
        "records": [rec] * (API_MAX_BATCH + 1),
    })
    assert r.status_code == 400


def test_deterministic_api(client, uci_record):
    a = client.post("/forecast", json=uci_record).json()
    b = client.post("/forecast", json=uci_record).json()
    assert a["forecasts"][0]["prediction"] == b["forecasts"][0]["prediction"]


def test_no_stacktrace_on_error(client):
    r = client.post("/forecast", json={"source_dataset": "UCI", "horizon": 1,
                                      "entity_id": "x", "product_key": "y", "date": "bad"})
    assert r.status_code in (400, 422)
    text = r.text.lower()
    assert "traceback" not in text
    assert "file \"" not in text
