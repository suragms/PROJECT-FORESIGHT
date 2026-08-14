"""Pydantic request/response schemas for the forecast API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    model_type: Optional[str] = None
    dataset: str
    horizon: int
    model_version: Optional[str] = None
    hash: str
    training_timestamp: Optional[str] = None
    status: Optional[str] = None
    supported_horizons: Optional[list[int]] = None


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
    datasets: list[str]
    horizons: list[int]


class ForecastRecord(BaseModel):
    source_dataset: str
    entity_id: str
    product_key: str
    date: Optional[str] = None
    forecast_date: Optional[str] = None
    origin_date: Optional[str] = None
    horizon: Optional[int] = None
    features: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_dataset")
    @classmethod
    def _ds(cls, v: str) -> str:
        return str(v).strip()


class ForecastRequest(ForecastRecord):
    source_dataset: str
    horizon: int
    include_actual: bool = False


class BatchForecastRequest(BaseModel):
    source_dataset: str
    horizon: int
    records: list[ForecastRecord]
    include_actual: bool = False


class ForecastRow(BaseModel):
    forecast_date: str
    source_dataset: str
    entity_id: str
    product_key: str
    horizon: int
    prediction: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    model_name: str
    model_version: Optional[str] = None
    generated_at: str
    actual: Optional[float] = None


class ForecastResponse(BaseModel):
    forecasts: list[ForecastRow]
    metadata: dict[str, Any]
    n: int
