"""Preprocessing contract for production inference.

Phase 11 models pickle a fitted `FeaturePreprocessor` (frequency encoding,
no median imputation). This module does not refit or impute. It only
documents required columns and builds a frame the engine can validate.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.forecasting.schemas import (
    CATEGORICAL_FEATURES_BOTH,
    CATEGORICAL_FEATURES_SYNTHETIC,
    NUMERIC_FEATURES_BOTH,
    NUMERIC_FEATURES_SYNTHETIC_EXTRA,
)
from src.forecasting.validation import InputValidationError


def expected_features(dataset: str, *, include_hcal: bool = False) -> dict[str, list[str]]:
    numeric = list(NUMERIC_FEATURES_BOTH)
    cats = list(CATEGORICAL_FEATURES_BOTH)
    if dataset == "SYNTHETIC":
        numeric = numeric + list(NUMERIC_FEATURES_SYNTHETIC_EXTRA)
        cats = cats + list(CATEGORICAL_FEATURES_SYNTHETIC)
    return {"numeric": numeric, "categorical": cats}


def required_columns(dataset: str, feature_cols: list[str] | None = None) -> list[str]:
    keys = ["date", "source_dataset", "entity_id", "product_key"]
    feats = feature_cols if feature_cols is not None else (
        expected_features(dataset)["numeric"] + expected_features(dataset)["categorical"]
    )
    return keys + list(feats)


def assert_feature_availability(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise InputValidationError(f"Missing required features/columns: {missing}")
