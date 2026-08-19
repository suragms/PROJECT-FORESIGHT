"""Input validation helpers for the production inference layer."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.config import SUPPORTED_DATASETS, SUPPORTED_HORIZONS, api_max_batch
from src.forecasting.schemas import DATE_COLUMNS, KEY_COLUMNS, LEAKAGE_FORBIDDEN
from src.phase11_common import PROHIBITED_NEGATIVE

FORBIDDEN_CLIENT_KEYS = {
    "model_file",
    "model_path",
    "model_dir",
    "joblib",
    "filepath",
    "file_path",
    "__file__",
    "registry_path",
}


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


def _reject_path_like(value: str, field: str) -> None:
    if not value or not str(value).strip():
        raise InputValidationError(f"{field} must not be empty")
    text = str(value)
    if "\x00" in text:
        raise InputValidationError(f"{field} contains invalid characters")
    if ".." in text.split("/") or ".." in text.split("\\") or ".." in text:
        raise InputValidationError(f"{field} must not contain path traversal")
    lowered = text.lower()
    if lowered.endswith(".joblib") or lowered.endswith(".pkl") or "models/final" in lowered.replace("\\", "/"):
        raise InputValidationError(f"{field} must not reference model files")


def _is_non_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, str):
        stripped = value.strip().lower()
        return stripped in {"nan", "inf", "+inf", "-inf", "infinity", "-infinity"}
    return False


def _validate_feature_value(key: str, value: Any, record_index: int) -> None:
    if key in FORBIDDEN_CLIENT_KEYS:
        raise InputValidationError(f"Record {record_index}: '{key}' is not an allowed field")
    if _is_non_finite_number(value):
        raise InputValidationError(
            f"Record {record_index}: non-finite numeric value in '{key}'"
        )
    if key in PROHIBITED_NEGATIVE and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < 0:
            raise InputValidationError(
                f"Record {record_index}: negative values are prohibited for '{key}'"
            )
    if key == "units_sold_lag_1" and value is None:
        raise InputValidationError(
            f"Record {record_index}: units_sold_lag_1 is required and must not be null"
        )
    if str(key).startswith("hcal_") and value is None:
        raise InputValidationError(
            f"Record {record_index}: required direct-horizon calendar '{key}' is null"
        )


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        raise InputValidationError("No forecast records provided")
    try:
        max_batch = api_max_batch()
    except (TypeError, ValueError) as exc:
        raise InputValidationError("Invalid FORESIGHT_API_MAX_BATCH") from exc
    if len(records) > max_batch:
        raise InputValidationError(
            f"Batch size {len(records)} exceeds maximum {max_batch}"
        )
    rows = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise InputValidationError(f"Record {i} is not an object")
        illegal = sorted(set(rec) & FORBIDDEN_CLIENT_KEYS)
        if illegal:
            raise InputValidationError(f"Record {i}: forbidden fields {illegal}")
        row = {k: v for k, v in rec.items() if k != "features"}
        for field in ("entity_id", "product_key"):
            if field in row and row[field] is not None:
                _reject_path_like(str(row[field]), field)
        feats = rec.get("features") or {}
        if feats and not isinstance(feats, dict):
            raise InputValidationError(f"Record {i}: features must be an object")
        if isinstance(feats, dict):
            overlap = set(feats) & set(row)
            if overlap:
                raise InputValidationError(
                    f"Record {i}: features overlap top-level keys {sorted(overlap)}"
                )
            forbidden_feat = sorted(set(feats) & FORBIDDEN_CLIENT_KEYS)
            if forbidden_feat:
                raise InputValidationError(
                    f"Record {i}: forbidden feature keys {forbidden_feat}"
                )
            nested = [k for k, v in feats.items() if isinstance(v, (dict, list))]
            if nested:
                raise InputValidationError(
                    f"Record {i}: nested feature values are not allowed: {nested}"
                )
            for k, v in feats.items():
                _validate_feature_value(str(k), v, i)
            row.update(feats)
        leak = [c for c in LEAKAGE_FORBIDDEN if c in row and c != "units_sold"]
        if leak:
            raise InputValidationError(f"Record {i}: leakage columns are not allowed: {leak}")
        rows.append(row)
    df = pd.DataFrame(rows)
    missing_keys = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing_keys:
        raise InputValidationError(f"Missing key columns: {missing_keys}")
    if not any(c in df.columns for c in DATE_COLUMNS):
        raise InputValidationError("Each record needs date, forecast_date, or origin_date")
    date_col = next(c for c in DATE_COLUMNS if c in df.columns)
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    if parsed.isna().any() and df[date_col].notna().any():
        bad = int(((df[date_col].notna()) & parsed.isna()).sum())
        if bad:
            raise InputValidationError(f"Invalid dates: {bad} values could not be parsed")
    grain = [c for c in ("source_dataset", "entity_id", "product_key", date_col) if c in df.columns]
    n_dup = int(df.duplicated(grain).sum())
    if n_dup:
        raise InputValidationError(f"Duplicate records: {n_dup} duplicate forecasting keys")
    return df
