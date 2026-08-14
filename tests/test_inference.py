"""Phase 12 inference tests."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from src.config import SAMPLES_DIR
from src.forecasting.inference import ForecastEngine
from src.forecasting.make_samples import write_samples
from src.forecasting.registry import RegistryError, load_registry, resolve_selected, verify_hash
from src.forecasting.validation import InputValidationError, records_to_frame


@pytest.fixture(scope="session")
def samples():
    uci = SAMPLES_DIR / "uci_h1_sample.parquet"
    if not uci.exists():
        write_samples()
    return {
        "UCI": pd.read_parquet(SAMPLES_DIR / "uci_h1_sample.parquet"),
        "SYNTHETIC": pd.read_parquet(SAMPLES_DIR / "synthetic_h1_sample.parquet"),
    }


def test_registry_loads_and_hashes():
    recs = load_registry()
    assert recs
    rec = resolve_selected("UCI", 1, recs)
    assert rec["model_id"] == "uci_h1_phase8_lightgbm"
    h = verify_hash(rec)
    assert len(h) == 64


def test_invalid_horizon_rejected():
    with pytest.raises(InputValidationError):
        ForecastEngine("UCI", 2)


def test_unregistered_dataset_rejected():
    with pytest.raises(Exception):
        ForecastEngine("OTHER", 1)


def test_valid_uci_forecast(samples):
    eng = ForecastEngine("UCI", 1)
    out = eng.predict(samples["UCI"].head(5), include_actual=False)
    assert len(out) == 5
    assert (out["prediction"] >= 0).all()
    assert "actual" not in out.columns
    assert "generated_at" in out.columns
    assert out["model_name"].iloc[0] == "uci_h1_phase8_lightgbm"


def test_deterministic_uci(samples):
    eng = ForecastEngine("UCI", 1)
    df = samples["UCI"].head(8)
    a = eng.predict(df, include_actual=False)
    b = eng.predict(df, include_actual=False)
    assert (a["prediction"].to_numpy() == b["prediction"].to_numpy()).all()


def test_missing_lag_rejected(samples):
    eng = ForecastEngine("UCI", 1)
    bad = samples["UCI"].head(3).drop(columns=["units_sold_lag_1"])
    with pytest.raises(InputValidationError):
        eng.predict(bad)


def test_duplicate_keys_rejected(samples):
    eng = ForecastEngine("UCI", 1)
    df = samples["UCI"].head(2)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(InputValidationError):
        eng.predict(dup)


def test_malformed_date_rejected(samples):
    eng = ForecastEngine("UCI", 1)
    df = samples["UCI"].head(2).copy()
    df["date"] = df["date"].astype(str)
    df.loc[df.index[0], "date"] = "not-a-date"
    with pytest.raises(InputValidationError):
        eng.predict(df)


def test_records_to_frame_batch_limit():
    from src.config import API_MAX_BATCH
    recs = [{"source_dataset": "UCI", "entity_id": "a", "product_key": "b", "date": "2011-01-01"}] * (API_MAX_BATCH + 1)
    with pytest.raises(InputValidationError):
        records_to_frame(recs)
