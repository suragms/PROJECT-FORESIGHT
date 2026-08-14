"""Dashboard smoke tests: load data without Streamlit rerun / retraining."""

from __future__ import annotations

import os

os.environ["FORECAST_DASHBOARD_SKIP_MAIN"] = "1"

from dashboard import forecast_analytics as dash  # noqa: E402
from src.config import FINAL_FORECASTS_PATH, REGISTRY_PATH


def test_dashboard_data_loads():
    preds = dash.load_parquet(str(FINAL_FORECASTS_PATH))
    assert preds is not None and len(preds) > 0
    reg = dash.load_json(str(REGISTRY_PATH))
    assert isinstance(reg, list) and len(reg) >= 10


def test_dashboard_handles_missing_file():
    assert dash.load_parquet("does/not/exist.parquet") is None
    assert dash.load_json("does/not/exist.json") is None
