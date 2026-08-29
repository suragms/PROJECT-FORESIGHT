"""Phase 20 production API routes — additive; does not modify /forecast."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.auth.dependencies import require_user_auth
from src.phase20_api_adapter import (
    SUPPORTED_HORIZON,
    batch_forecast_from_features_df,
    generate_forecast,
    model_metadata,
    validate_source,
)
from src.phase20_risk_adapter import explain_risk, compute_risk

router = APIRouter(dependencies=[Depends(require_user_auth)])


class Phase20FeatureRecord(BaseModel):
    product_key: str
    features: dict = Field(..., description="Feature name -> value map")


class Phase20ForecastRequest(BaseModel):
    source_dataset: str = "SYNTHETIC"
    forecast_origin: str | None = None
    records: list[Phase20FeatureRecord]
    include_extended: bool = False


class Phase20RiskRequest(BaseModel):
    sku_id: str
    forecast_weekly_demand: float
    on_hand_units: float = 0
    on_order_units: float = 0
    lead_time_weeks: int = 2
    safety_stock: float = 0
    reorder_point: float = 0
    base_price: float | None = None
    cost_price: float | None = None


@router.get("/model")
def phase20_model_info():
    return model_metadata()


@router.post("/forecast")
def phase20_forecast(body: Phase20ForecastRequest):
    try:
        validate_source(body.source_dataset)
        rows = []
        for rec in body.records:
            row = {"product_key": rec.product_key, **rec.features}
            rows.append(row)
        results = generate_forecast(
            rows,
            source_dataset=body.source_dataset,
            forecast_origin=body.forecast_origin,
            include_extended=body.include_extended,
        )
        return {
            "forecasts": results,
            "metadata": model_metadata(),
            "n": len(results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/risk/explain")
def phase20_risk_explain(body: Phase20RiskRequest):
    row = body.model_dump()
    row["sku_id"] = body.sku_id
    return explain_risk(row)


@router.get("/contract")
def phase20_contract():
    import json
    import os
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "docs", "phase20_feature_contract.json"
    )
    path = os.path.normpath(path)
    with open(path) as f:
        return json.load(f)
