"""Forecast API routes. Registered models only; no filesystem paths from clients."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.metrics import incr, observe_batch, snapshot
from src.api.schemas import (
    BatchForecastRequest,
    ForecastRequest,
    ForecastResponse,
    ForecastRow,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
)
from src.config import (
    APP_VERSION,
    SUPPORTED_DATASETS,
    SUPPORTED_HORIZONS,
)
from src.forecasting.inference import ForecastEngine
from src.forecasting.registry import RegistryError, load_registry, resolve_selected, verify_hash
from src.forecasting.validation import InputValidationError, records_to_frame, validate_dataset_horizon
from src.production.health import liveness
from src.production.readiness import check_readiness
from src.security.audit import audit

router = APIRouter()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _frame_to_rows(df: pd.DataFrame) -> list[ForecastRow]:
    rows = []
    for rec in df.to_dict(orient="records"):
        def _f(key, default=None):
            val = rec.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return val
        fd = rec.get("forecast_date")
        if hasattr(fd, "isoformat"):
            fd = fd.isoformat()
        rows.append(ForecastRow(
            forecast_date=str(fd),
            source_dataset=str(rec["source_dataset"]),
            entity_id=str(rec["entity_id"]),
            product_key=str(rec["product_key"]),
            horizon=int(rec["horizon"]),
            prediction=float(rec["prediction"]),
            lower_bound=_f("lower_bound"),
            upper_bound=_f("upper_bound"),
            model_name=str(rec["model_name"]),
            model_version=_f("model_version"),
            generated_at=str(rec.get("generated_at") or _now()),
            actual=_f("actual"),
        ))
    return rows


def _engine(request: Request, dataset: str, horizon: int) -> ForecastEngine:
    cache: dict = request.app.state.engines
    key = (dataset, int(horizon))
    if key not in cache:
        cache[key] = ForecastEngine(dataset, int(horizon))
        meta = cache[key].metadata()
        audit(
            "model_loaded",
            model_id=meta.get("model_id"),
            dataset=dataset,
            horizon=int(horizon),
            hash=meta.get("hash"),
        )
        audit("model_hash_verification", model_id=meta.get("model_id"), hash=meta.get("hash"))
    return cache[key]


@router.get("/")
def root() -> dict:
    return {
        "status": "online",
        "service": "Project FORESIGHT — Demand & Inventory Intelligence API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "supported_datasets": list(SUPPORTED_DATASETS),
        "supported_horizons": list(SUPPORTED_HORIZONS),
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    body = liveness()
    return HealthResponse(**body)


@router.get("/ready")
def ready() -> JSONResponse:
    status_code, body = check_readiness()
    audit(
        "readiness_check",
        status=body.get("status"),
        models_verified=body.get("models_verified"),
        registry_verified=body.get("registry_verified"),
    )
    return JSONResponse(status_code=status_code, content=body)


@router.get("/metrics")
def metrics() -> dict:
    return snapshot()


@router.get("/model", response_model=ModelListResponse)
def model_info(dataset: str | None = None, horizon: int | None = None) -> ModelListResponse:
    try:
        recs = load_registry()
        selected = [r for r in recs if r.get("status") == "selected"]
        if dataset is not None and horizon is not None:
            validate_dataset_horizon(dataset, horizon)
            rec = resolve_selected(dataset, int(horizon), recs)
            selected = [rec]
        elif dataset is not None:
            selected = [r for r in selected if r.get("dataset") == dataset]
        models = []
        for r in selected:
            h = verify_hash(r)
            models.append(ModelInfo(
                model_id=r["model_id"],
                model_name=r["model_id"],
                model_type=r.get("model_type"),
                dataset=r["dataset"],
                horizon=int(r["horizon"]),
                model_version=r.get("code_version"),
                hash=h,
                training_timestamp=r.get("training_timestamp"),
                status=r.get("status"),
                supported_horizons=list(SUPPORTED_HORIZONS),
            ))
        return ModelListResponse(
            models=models,
            datasets=list(SUPPORTED_DATASETS),
            horizons=list(SUPPORTED_HORIZONS),
        )
    except (RegistryError, InputValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/forecast", response_model=ForecastResponse)
def forecast(body: ForecastRequest, request: Request) -> ForecastResponse:
    try:
        validate_dataset_horizon(body.source_dataset, body.horizon)
        rec = body.model_dump()
        rec.pop("include_actual", None)
        rec["features"] = body.features
        df = records_to_frame([rec])
        engine = _engine(request, body.source_dataset, body.horizon)
        out = engine.predict(df, include_actual=body.include_actual)
        observe_batch(1)
        rid = getattr(request.state, "request_id", None)
        audit(
            "forecast_request",
            dataset=body.source_dataset,
            horizon=body.horizon,
            n=1,
            model_id=engine.metadata().get("model_id"),
            request_id=rid,
        )
        return ForecastResponse(
            forecasts=_frame_to_rows(out),
            metadata=engine.metadata(),
            n=len(out),
        )
    except (InputValidationError, RegistryError) as exc:
        incr("forecast_failures")
        audit("validation_failure", path="/forecast", reason=str(exc)[:200])
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/forecast/batch", response_model=ForecastResponse)
def forecast_batch(body: BatchForecastRequest, request: Request) -> ForecastResponse:
    try:
        validate_dataset_horizon(body.source_dataset, body.horizon)
        payload = []
        for rec in body.records:
            d = rec.model_dump()
            d["source_dataset"] = body.source_dataset
            d["horizon"] = body.horizon
            payload.append(d)
        df = records_to_frame(payload)
        engine = _engine(request, body.source_dataset, body.horizon)
        out = engine.predict(df, include_actual=body.include_actual)
        observe_batch(len(out))
        rid = getattr(request.state, "request_id", None)
        audit(
            "batch_forecast_request",
            dataset=body.source_dataset,
            horizon=body.horizon,
            n=len(out),
            model_id=engine.metadata().get("model_id"),
            request_id=rid,
        )
        return ForecastResponse(
            forecasts=_frame_to_rows(out),
            metadata=engine.metadata(),
            n=len(out),
        )
    except (InputValidationError, RegistryError) as exc:
        incr("forecast_failures")
        audit("validation_failure", path="/forecast/batch", reason=str(exc)[:200])
        raise HTTPException(status_code=400, detail=str(exc)) from exc
