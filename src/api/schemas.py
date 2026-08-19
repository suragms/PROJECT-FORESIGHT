"""Pydantic request/response schemas for the forecast API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import SUPPORTED_DATASETS, SUPPORTED_HORIZONS


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class ReadyResponse(BaseModel):
    status: str
    version: str
    models_verified: bool
    registry_verified: bool
    config_valid: Optional[bool] = None
    reason: Optional[str] = None
    config_errors: Optional[list[str]] = None


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
    model_config = ConfigDict(extra="forbid")

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
        value = str(v).strip()
        if value not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset '{value}'")
        return value

    @field_validator("entity_id", "product_key")
    @classmethod
    def _ids(cls, v: str) -> str:
        value = str(v).strip()
        if not value:
            raise ValueError("must not be empty")
        if ".." in value or "\x00" in value:
            raise ValueError("path traversal is not allowed")
        lowered = value.lower()
        if lowered.endswith(".joblib") or "models/final" in lowered.replace("\\", "/"):
            raise ValueError("model filesystem paths are not allowed")
        return value

    @field_validator("horizon")
    @classmethod
    def _horizon(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if int(v) not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported horizon {v}")
        return int(v)

    @field_validator("features")
    @classmethod
    def _features(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("features must be an object")
        for key in v:
            lowered = str(key).lower()
            if lowered in {"model_file", "model_path", "filepath", "file_path"}:
                raise ValueError("model filesystem paths are not allowed")
        return v


class ForecastRequest(ForecastRecord):
    source_dataset: str
    horizon: int
    include_actual: bool = False

    @field_validator("horizon")
    @classmethod
    def _required_horizon(cls, v: int) -> int:
        if int(v) not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported horizon {v}")
        return int(v)


class BatchForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_dataset: str
    horizon: int
    records: list[ForecastRecord]
    include_actual: bool = False

    @field_validator("source_dataset")
    @classmethod
    def _ds(cls, v: str) -> str:
        value = str(v).strip()
        if value not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset '{value}'")
        return value

    @field_validator("horizon")
    @classmethod
    def _horizon(cls, v: int) -> int:
        if int(v) not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported horizon {v}")
        return int(v)


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
