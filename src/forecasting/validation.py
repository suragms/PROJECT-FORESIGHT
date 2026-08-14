"""Input validation helpers for the production inference layer."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import API_MAX_BATCH, SUPPORTED_DATASETS, SUPPORTED_HORIZONS
from src.forecasting.schemas import DATE_COLUMNS, KEY_COLUMNS, LEAKAGE_FORBIDDEN


class InputValidationError(ValueError):
    """Caller-facing validation error (safe to return to API clients)."""


def validate_dataset_horizon(dataset: str, horizon: int) -> None:
    if dataset not in SUPPORTED_DATASETS:
        raise InputValidationError(
            f"Unsupported dataset '{dataset}'. Supported: {list(SUPPORTED_DATASETS)}"
        )
    try:
        h = int(horizon)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"Invalid horizon: {horizon}") from exc
    if h not in SUPPORTED_HORIZONS:
        raise InputValidationError(
            f"Unsupported horizon {h}. Supported: {list(SUPPORTED_HORIZONS)}"
        )


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        raise InputValidationError("No forecast records provided")
    if len(records) > API_MAX_BATCH:
        raise InputValidationError(
            f"Batch size {len(records)} exceeds maximum {API_MAX_BATCH}"
        )
    rows = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise InputValidationError(f"Record {i} is not an object")
        row = {k: v for k, v in rec.items() if k != "features"}
        feats = rec.get("features") or {}
        if feats and not isinstance(feats, dict):
            raise InputValidationError(f"Record {i}: features must be an object")
        if isinstance(feats, dict):
            overlap = set(feats) & set(row)
            if overlap:
                raise InputValidationError(
                    f"Record {i}: features overlap top-level keys {sorted(overlap)}"
                )
            row.update(feats)
        leak = [c for c in LEAKAGE_FORBIDDEN if c in row and c != "units_sold"]
        # units_sold is allowed only as optional actual, never as a predictor later
        rows.append(row)
    df = pd.DataFrame(rows)
    missing_keys = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing_keys:
        raise InputValidationError(f"Missing key columns: {missing_keys}")
    if not any(c in df.columns for c in DATE_COLUMNS):
        raise InputValidationError("Each record needs date, forecast_date, or origin_date")
    return df
