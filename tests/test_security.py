"""Phase 13 security, readiness, rate-limit, and validation tests."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import SAMPLES_DIR
from src.forecasting.make_samples import write_samples
from src.forecasting.registry import RegistryError
from src.forecasting.validation import InputValidationError, records_to_frame
from src.production.config_validation import ConfigValidationError, assert_runtime_config, validate_runtime_config


@pytest.fixture()
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


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_health_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    for header in ("x-content-type-options", "x-frame-options", "referrer-policy", "content-security-policy"):
        assert header in r.headers


def test_ready_when_valid(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["models_verified"] is True
    assert body["registry_verified"] is True


def test_unauthenticated_rejected(monkeypatch):
    monkeypatch.setenv("FORESIGHT_ENV", "production")
    monkeypatch.setenv("FORESIGHT_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("FORESIGHT_API_API_KEY", "phase13-test-key")
    client = TestClient(create_app())
    r = client.get("/model")
    assert r.status_code == 401
    assert r.json()["detail"] == "Unauthorized"
    assert "phase13-test-key" not in r.text
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_valid_authentication_accepted(monkeypatch):
    monkeypatch.setenv("FORESIGHT_ENV", "production")
    monkeypatch.setenv("FORESIGHT_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("FORESIGHT_API_API_KEY", "phase13-test-key")
    client = TestClient(create_app())
    r = client.get("/model", headers={"X-API-Key": "phase13-test-key"})
    assert r.status_code == 200
    assert len(r.json()["models"]) >= 1


def test_bearer_authentication_accepted(monkeypatch, uci_record):
    monkeypatch.setenv("FORESIGHT_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("FORESIGHT_API_API_KEY", "phase13-test-key")
    client = TestClient(create_app())
    r = client.post(
        "/forecast",
        json=uci_record,
        headers={"Authorization": "Bearer phase13-test-key"},
    )
    assert r.status_code == 200, r.text


def test_invalid_authentication_rejected(monkeypatch):
    monkeypatch.setenv("FORESIGHT_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("FORESIGHT_API_API_KEY", "phase13-test-key")
    client = TestClient(create_app())
    r = client.get("/model", headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401
    assert "phase13-test-key" not in r.text.lower()
    assert "traceback" not in r.text.lower()


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setenv("FORESIGHT_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("FORESIGHT_RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("FORESIGHT_RATE_LIMIT_FORECAST_REQUESTS", "3")
    monkeypatch.setenv("FORESIGHT_RATE_LIMIT_WINDOW_SECONDS", "60")
    client = TestClient(create_app())
    codes = [client.get("/model").status_code for _ in range(3)]
    assert codes == [200, 200, 200]
    limited = client.get("/model")
    assert limited.status_code == 429
    assert limited.json()["detail"] == "Rate limit exceeded"
    assert client.get("/health").status_code == 200


def test_ready_not_ready_when_registry_unavailable(monkeypatch):
    from src.production import readiness as readiness_mod

    def _boom(*_a, **_k):
        raise RegistryError("Model registry not found")

    monkeypatch.setattr(readiness_mod, "load_registry", _boom)
    client = TestClient(create_app())
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"
    assert r.json()["registry_verified"] is False


def test_ready_not_ready_when_hash_fails(monkeypatch):
    from src.production import readiness as readiness_mod

    def _boom(*_a, **_k):
        raise RegistryError("Hash mismatch")

    monkeypatch.setattr(readiness_mod, "verify_hash", _boom)
    client = TestClient(create_app())
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["models_verified"] is False


def test_production_rejects_missing_api_key(monkeypatch):
    monkeypatch.setenv("FORESIGHT_ENV", "production")
    monkeypatch.setenv("FORESIGHT_API_AUTH_ENABLED", "true")
    monkeypatch.delenv("FORESIGHT_API_API_KEY", raising=False)
    errors = validate_runtime_config()
    assert any("FORESIGHT_API_API_KEY" in e for e in errors)
    with pytest.raises(ConfigValidationError):
        assert_runtime_config()


def test_no_traceback_or_secret_leak(client, monkeypatch):
    monkeypatch.setenv("FORESIGHT_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("FORESIGHT_API_API_KEY", "super-secret-key")
    authed = TestClient(create_app())
    r = authed.post(
        "/forecast",
        json={"source_dataset": "UCI", "horizon": 1, "entity_id": "x", "product_key": "y", "date": "bad"},
        headers={"X-API-Key": "super-secret-key"},
    )
    assert r.status_code in (400, 422)
    text = r.text.lower()
    assert "traceback" not in text
    assert 'file "' not in text
    assert "super-secret-key" not in r.text


def test_path_traversal_rejected(client):
    r = client.post("/forecast", json={
        "source_dataset": "UCI",
        "horizon": 1,
        "date": "2011-09-26",
        "entity_id": "../models/final/uci_h1_phase8_lightgbm.joblib",
        "product_key": "x",
        "features": {"units_sold_lag_1": 1},
    })
    assert r.status_code in (400, 422)


def test_model_path_field_rejected(client):
    r = client.post("/forecast", json={
        "source_dataset": "UCI",
        "horizon": 1,
        "date": "2011-09-26",
        "entity_id": "ONLINE",
        "product_key": "UCI_1",
        "model_path": "models/final/secret.joblib",
        "features": {"units_sold_lag_1": 1},
    })
    assert r.status_code in (400, 422)


def test_non_finite_feature_rejected():
    with pytest.raises(InputValidationError):
        records_to_frame([{
            "source_dataset": "UCI",
            "entity_id": "ONLINE",
            "product_key": "UCI_1",
            "date": "2011-09-26",
            "features": {"units_sold_lag_1": float("inf")},
        }])


def test_negative_lag_rejected():
    with pytest.raises(InputValidationError):
        records_to_frame([{
            "source_dataset": "UCI",
            "entity_id": "ONLINE",
            "product_key": "UCI_1",
            "date": "2011-09-26",
            "features": {"units_sold_lag_1": -1},
        }])


def test_duplicate_records_rejected():
    rec = {
        "source_dataset": "UCI",
        "entity_id": "ONLINE",
        "product_key": "UCI_1",
        "date": "2011-09-26",
        "features": {"units_sold_lag_1": 1},
    }
    with pytest.raises(InputValidationError):
        records_to_frame([rec, rec])


def test_oversized_payload_rejected(monkeypatch, uci_record):
    monkeypatch.setenv("FORESIGHT_API_MAX_PAYLOAD_BYTES", "128")
    client = TestClient(create_app())
    r = client.post("/forecast", json=uci_record)
    assert r.status_code == 413


def test_oversized_batch_rejected(uci_record, monkeypatch):
    monkeypatch.setenv("FORESIGHT_API_MAX_BATCH", "2")
    rec = {k: v for k, v in uci_record.items() if k != "horizon"}
    records = []
    for i in range(3):
        item = dict(rec)
        item["product_key"] = f"{rec['product_key']}_{i}"
        records.append(item)
    client = TestClient(create_app())
    r = client.post("/forecast/batch", json={
        "source_dataset": "UCI",
        "horizon": 1,
        "records": records,
    })
    assert r.status_code in (400, 422)
