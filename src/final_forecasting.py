"""
Final forecasting inference layer (Phase 11).

Loads selected models from models/final/. Does not modify Phase 8 inference.
Rejects invalid input instead of silently repairing it.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml_forecasting import FeaturePreprocessor  # noqa: E402
from src.phase10_common import GRAIN, TARGET  # noqa: E402
from src.phase10_direct_horizon import HCAL_PREFIX  # noqa: E402
from src.phase10_hurdle_forecasting import _combine  # noqa: E402
from src.phase10_prediction_intervals import _enforce_nonneg_and_order  # noqa: E402
from src.phase11_common import (  # noqa: E402
    LEAKAGE_FORBIDDEN,
    MODELS_FINAL_DIR,
    OUTPUT_SCHEMA,
    PROHIBITED_NEGATIVE,
    REGISTRY_PATH,
    file_sha256,
)

logger = logging.getLogger("final_forecasting")


class FinalForecastError(ValueError):
    """Invalid inference input or model payload."""


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise FinalForecastError(f"Missing {label}: {missing}")


def _as_datetime(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().any():
        n = int(out.isna().sum())
        raise FinalForecastError(f"Invalid dates: {n} values could not be parsed")
    return out


class FinalForecaster:
    """Load one selected final model and generate schema-stable forecasts."""

    def __init__(self, payload: dict[str, Any], *, model_file: str, expected_hash: Optional[str] = None):
        self.payload = payload
        self.model_file = model_file
        self.model_id = str(payload["model_id"])
        self.dataset = str(payload["dataset"])
        self.horizon = int(payload["horizon"])
        self.model_type = str(payload["model_type"])
        self.model_version = str(payload.get("code_version") or payload.get("model_id"))
        self.numeric = list(payload["numeric_features"])
        self.categorical = list(payload["categorical_features"])
        self.feature_cols = self.numeric + self.categorical
        leak = [c for c in self.feature_cols if c in LEAKAGE_FORBIDDEN]
        if leak:
            raise FinalForecastError(f"Payload features include leakage columns: {leak}")
        if expected_hash:
            got = file_sha256(model_file)
            if got != expected_hash:
                raise FinalForecastError(
                    f"Model hash mismatch for {self.model_id}: expected {expected_hash}, got {got}"
                )
        logger.info(
            "Loaded final model %s type=%s dataset=%s horizon=%s file=%s",
            self.model_id, self.model_type, self.dataset, self.horizon, model_file,
        )

    @classmethod
    def from_file(cls, path: str, expected_hash: Optional[str] = None) -> "FinalForecaster":
        if not os.path.exists(path):
            raise FinalForecastError(f"Model file not found: {path}")
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "model_id" not in payload:
            raise FinalForecastError(f"Unrecognized model payload: {path}")
        return cls(payload, model_file=path, expected_hash=expected_hash)

    @classmethod
    def from_registry(cls, model_id: str, registry_path: str = REGISTRY_PATH) -> "FinalForecaster":
        rec = load_registry_record(model_id, registry_path)
        path = rec["model_file"]
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        return cls.from_file(path, expected_hash=rec.get("hash"))

    def validate_input(self, df: pd.DataFrame, *, allow_actual: bool = True) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise FinalForecastError("Input must be a pandas DataFrame")
        if df.empty:
            raise FinalForecastError("Input is empty")
        out = df.copy()
        _require_columns(out, ["source_dataset", "entity_id", "product_key"], "key columns")
        if "date" not in out.columns and "forecast_date" not in out.columns and "origin_date" not in out.columns:
            raise FinalForecastError("Input needs date, forecast_date, or origin_date")
        if "date" not in out.columns:
            if "origin_date" in out.columns:
                out["date"] = out["origin_date"]
            else:
                out["date"] = out["forecast_date"]
        out["date"] = _as_datetime(out["date"])
        srcs = set(out["source_dataset"].astype(str).unique())
        if srcs != {self.dataset}:
            raise FinalForecastError(
                f"source_dataset {sorted(srcs)} does not match model dataset {self.dataset}"
            )
        if "horizon" in out.columns:
            hz = set(pd.to_numeric(out["horizon"], errors="coerce").dropna().astype(int).unique())
            if hz and hz != {self.horizon}:
                raise FinalForecastError(
                    f"horizon values {sorted(hz)} do not match model horizon {self.horizon}"
                )

        key = ["date", "source_dataset", "entity_id", "product_key"]
        n_dup = int(out.duplicated(key).sum())
        if n_dup:
            raise FinalForecastError(f"Duplicate forecasting keys: {n_dup} rows")
        # Within each series, dates must be unique (already implied by grain)
        # and not reverse-sorted relative to the source file without being sortable.
        out = out.sort_values(GRAIN)

        _require_columns(out, self.feature_cols, "required model features")

        for c in self.numeric:
            coerced = pd.to_numeric(out[c], errors="coerce")
            invalid = out[c].notna() & coerced.isna()
            if int(invalid.sum()):
                raise FinalForecastError(
                    f"Non-numeric values in feature {c}: {int(invalid.sum())} rows"
                )
            out[c] = coerced

        # lag_1 is the history gate used in Phases 8-10. Longer lags / rolling
        # windows and SYNTHETIC historical_doi may be LightGBM-native NaN;
        # they are not imputed. Known-in-advance hcal_* must be present.
        if "units_sold_lag_1" in self.numeric:
            n_miss = int(out["units_sold_lag_1"].isna().sum())
            if n_miss:
                raise FinalForecastError(
                    f"Missing units_sold_lag_1 (insufficient history): {n_miss} rows"
                )
        hcal_cols = [c for c in self.numeric + self.categorical if str(c).startswith("hcal_")]
        for c in hcal_cols:
            n_miss = int(out[c].isna().sum())
            if n_miss:
                raise FinalForecastError(
                    f"Missing known-in-advance target calendar {c}: {n_miss} rows"
                )

        for c in self.categorical:
            if c not in out.columns:
                raise FinalForecastError(f"Missing categorical column {c}")

        for c in PROHIBITED_NEGATIVE:
            if c in out.columns and c in self.numeric:
                n_neg = int((out[c] < 0).sum())
                if n_neg:
                    raise FinalForecastError(
                        f"Impossible negative values in {c}: {n_neg} rows (rejected, not clipped)"
                    )

        lag_cols = [c for c in self.numeric if c.startswith("units_sold_lag_")]
        for c in lag_cols:
            if c not in out.columns:
                raise FinalForecastError(f"Missing lag feature {c}")

        if TARGET in out.columns and not allow_actual:
            raise FinalForecastError(
                "units_sold is present but this call forbids actuals (future forecast)"
            )
        return out.sort_values(GRAIN).reset_index(drop=True)

    def _transform_point(self, df: pd.DataFrame, preprocessor: FeaturePreprocessor) -> pd.DataFrame:
        X = df[preprocessor.numeric + preprocessor.categorical]
        arr = preprocessor.transform(X)
        return pd.DataFrame(arr, columns=preprocessor.feature_names_)

    def _predict_raw(self, df: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        mt = self.model_type
        if mt in ("lightgbm_point", "direct_lightgbm"):
            pre: FeaturePreprocessor = self.payload["preprocessor"]
            model = self.payload["model"]
            X = self._transform_point(df, pre)
            pred = np.asarray(model.predict(X), dtype=float)
            pred = np.nan_to_num(pred, nan=np.nan, posinf=np.nan, neginf=np.nan)
            if not np.isfinite(pred).all():
                raise FinalForecastError("Model produced non-finite predictions")
            pred = np.maximum(0.0, pred)
            return pred, None, None
        if mt == "hurdle":
            pre_clf: FeaturePreprocessor = self.payload["preprocessor_clf"]
            pre_reg: FeaturePreprocessor = self.payload["preprocessor_reg"]
            clf = self.payload["classifier"]
            reg = self.payload["regressor"]
            th = float(self.payload["threshold"])
            Xc = self._transform_point(df, pre_clf)
            Xr = self._transform_point(df, pre_reg)
            proba = np.asarray(clf.predict_proba(Xc)[:, 1], dtype=float)
            qty = np.asarray(reg.predict(Xr), dtype=float)
            if not np.isfinite(proba).all() or not np.isfinite(qty).all():
                raise FinalForecastError("Hurdle model produced non-finite predictions")
            pred = _combine(proba, qty, th)
            return pred, None, None
        if mt == "quantile_intervals":
            pre: FeaturePreprocessor = self.payload["preprocessor"]
            models = self.payload["models"]
            X = self._transform_point(df, pre)
            raw = {tau: np.asarray(models[tau].predict(X), dtype=float) for tau in (0.10, 0.50, 0.90)}
            p10, p50, p90, _ = _enforce_nonneg_and_order(raw[0.10], raw[0.50], raw[0.90])
            if not (np.isfinite(p10).all() and np.isfinite(p50).all() and np.isfinite(p90).all()):
                raise FinalForecastError("Quantile model produced non-finite predictions")
            return p50, p10, p90
        raise FinalForecastError(f"Unsupported model_type: {mt}")

    def predict(
        self,
        df: pd.DataFrame,
        *,
        include_actual: bool = True,
        intervals: Optional["FinalForecaster"] = None,
    ) -> pd.DataFrame:
        frame = df.copy()
        if self.model_type == "direct_lightgbm":
            frame = self._prepare_direct(frame)
        validated = self.validate_input(frame, allow_actual=True)
        pred, lo, hi = self._predict_raw(validated)
        if intervals is not None:
            iv = intervals.predict(df, include_actual=include_actual)
            if len(iv) != len(pred):
                raise FinalForecastError(
                    f"Interval companion row count {len(iv)} != point forecast {len(pred)}"
                )
            lo = iv["lower_bound"].to_numpy()
            hi = iv["upper_bound"].to_numpy()
        if self.horizon == 1 and self.model_type != "direct_lightgbm":
            forecast_date = validated["date"]
            origin_date = validated["date"]
        else:
            forecast_date = validated.get("target_date", validated["date"])
            origin_date = validated["date"]
        actual = np.full(len(validated), np.nan)
        if include_actual:
            if self.model_type == "direct_lightgbm" and "target" in validated.columns:
                actual = pd.to_numeric(validated["target"], errors="coerce").to_numpy(dtype=float)
            elif TARGET in validated.columns:
                actual = pd.to_numeric(validated[TARGET], errors="coerce").to_numpy(dtype=float)
            elif "actual" in validated.columns:
                actual = pd.to_numeric(validated["actual"], errors="coerce").to_numpy(dtype=float)

        out = pd.DataFrame({
            "forecast_date": pd.to_datetime(forecast_date),
            "origin_date": pd.to_datetime(origin_date),
            "source_dataset": validated["source_dataset"].astype(str).to_numpy(),
            "entity_id": validated["entity_id"].to_numpy(),
            "product_key": validated["product_key"].to_numpy(),
            "horizon": np.full(len(validated), self.horizon, dtype=int),
            "actual": actual,
            "prediction": pred,
            "lower_bound": lo if lo is not None else np.full(len(validated), np.nan),
            "upper_bound": hi if hi is not None else np.full(len(validated), np.nan),
            "model_name": np.full(len(validated), self.model_id),
            "model_version": np.full(len(validated), self.model_version),
        })
        extra = [c for c in OUTPUT_SCHEMA if c not in out.columns]
        if extra:
            raise FinalForecastError(f"Internal schema missing {extra}")
        logger.info(
            "Forecast %s n=%s horizon=%s",
            self.model_id, len(out), self.horizon,
        )
        return out

    def _prepare_direct(self, df: pd.DataFrame) -> pd.DataFrame:
        hcal_needed = [c for c in self.numeric + self.categorical if str(c).startswith(HCAL_PREFIX)]
        missing = [c for c in hcal_needed if c not in df.columns]
        if missing:
            raise FinalForecastError(
                "Direct-horizon inference requires known-in-advance target calendar "
                f"columns at the origin. Missing: {missing}. "
                "Do not pass a truncated panel and expect an internal shift; "
                "supply hcal_* (holiday/season/calendar of the target date)."
            )
        return df


def load_registry(path: str = REGISTRY_PATH) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise FinalForecastError(f"Registry not found: {path}")
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)
    if not isinstance(recs, list):
        raise FinalForecastError("Registry must be a list")
    return recs


def load_registry_record(model_id: str, path: str = REGISTRY_PATH) -> dict[str, Any]:
    recs = load_registry(path)
    hits = [r for r in recs if r.get("model_id") == model_id]
    if not hits:
        raise FinalForecastError(f"model_id not in registry: {model_id}")
    return hits[0]


def list_selected_models(path: str = REGISTRY_PATH) -> list[dict[str, Any]]:
    return [r for r in load_registry(path) if r.get("status") == "selected"]


def model_path_for(model_id: str) -> str:
    return os.path.join(MODELS_FINAL_DIR, f"{model_id}.joblib")
