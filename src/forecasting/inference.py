"""
Production inference engine.

Loads Phase 11 selected models from the registry, validates hashes, applies
the fitted training preprocessor, and returns the Phase 11 output schema plus
generated_at. Does not retrain.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from src.final_forecasting import FinalForecastError, FinalForecaster
from src.forecasting.registry import (
    RegistryError,
    get_record,
    interval_companion,
    load_registry,
    resolve_model_file,
    resolve_selected,
    verify_hash,
)
from src.forecasting.schemas import OUTPUT_COLUMNS
from src.forecasting.validation import InputValidationError, validate_dataset_horizon

logger = logging.getLogger("forecast_service.inference")


class ForecastEngine:
    """Resolve a registered model and run Phase 11 inference."""

    def __init__(self, dataset: str, horizon: int, *, attach_intervals: bool = True):
        validate_dataset_horizon(dataset, horizon)
        self.dataset = dataset
        self.horizon = int(horizon)
        self.registry = load_registry()
        self.record = resolve_selected(dataset, self.horizon, self.registry)
        self.hash = verify_hash(self.record)
        path = str(resolve_model_file(self.record))
        t0 = time.perf_counter()
        self.forecaster = FinalForecaster.from_file(path, expected_hash=self.hash)
        self.load_seconds = time.perf_counter() - t0
        self.interval_forecaster: Optional[FinalForecaster] = None
        self.interval_record = None
        if attach_intervals:
            iv = interval_companion(dataset, self.horizon, self.registry)
            if iv is not None:
                verify_hash(iv)
                self.interval_record = iv
                self.interval_forecaster = FinalForecaster.from_file(
                    str(resolve_model_file(iv)), expected_hash=iv.get("hash")
                )
        logger.info(
            "engine_ready model_id=%s dataset=%s horizon=%s hash=%s load_s=%.3f intervals=%s",
            self.record["model_id"], dataset, self.horizon, self.hash,
            self.load_seconds, self.interval_forecaster is not None,
        )

    @classmethod
    def from_model_id(cls, model_id: str, *, attach_intervals: bool = True) -> "ForecastEngine":
        rec = get_record(model_id)
        eng = object.__new__(cls)
        validate_dataset_horizon(rec["dataset"], int(rec["horizon"]))
        eng.dataset = rec["dataset"]
        eng.horizon = int(rec["horizon"])
        eng.registry = load_registry()
        eng.record = rec
        if rec.get("status") not in ("selected", "interval_companion"):
            raise RegistryError(f"Model {model_id} is not a registered production model")
        eng.hash = verify_hash(rec)
        eng.forecaster = FinalForecaster.from_file(
            str(resolve_model_file(rec)), expected_hash=eng.hash
        )
        eng.load_seconds = 0.0
        eng.interval_forecaster = None
        eng.interval_record = None
        if attach_intervals and rec.get("status") == "selected":
            iv = interval_companion(eng.dataset, eng.horizon, eng.registry)
            if iv is not None:
                verify_hash(iv)
                eng.interval_record = iv
                eng.interval_forecaster = FinalForecaster.from_file(
                    str(resolve_model_file(iv)), expected_hash=iv.get("hash")
                )
        return eng

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.record["model_id"],
            "model_name": self.record["model_id"],
            "model_type": self.record.get("model_type"),
            "dataset": self.dataset,
            "horizon": self.horizon,
            "model_version": self.record.get("code_version"),
            "hash": self.hash,
            "training_timestamp": self.record.get("training_timestamp"),
            "status": self.record.get("status"),
            "interval_model_id": None if not self.interval_record else self.interval_record["model_id"],
        }

    def predict(self, df: pd.DataFrame, *, include_actual: bool = False) -> pd.DataFrame:
        t0 = time.perf_counter()
        n_in = int(len(df))
        logger.info(
            "infer_start model_id=%s n=%s horizon=%s",
            self.record["model_id"], n_in, self.horizon,
        )
        try:
            out = self.forecaster.predict(
                df,
                include_actual=include_actual,
                intervals=self.interval_forecaster,
            )
        except FinalForecastError as exc:
            logger.info("infer_validation_error model_id=%s err=%s", self.record["model_id"], exc)
            raise InputValidationError(str(exc)) from exc
        generated = datetime.now(timezone.utc).isoformat()
        out["generated_at"] = generated
        if not include_actual and "actual" in out.columns:
            out = out.drop(columns=["actual"])
        elif include_actual and "actual" in out.columns and out["actual"].notna().sum() == 0:
            out = out.drop(columns=["actual"])
        elapsed = time.perf_counter() - t0
        logger.info(
            "infer_done model_id=%s n_out=%s duration_s=%.4f validation=pass",
            self.record["model_id"], len(out), elapsed,
        )
        cols = [c for c in OUTPUT_COLUMNS if c in out.columns]
        extra = [c for c in out.columns if c not in cols]
        return out[cols + extra]
