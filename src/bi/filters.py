"""Filter helpers for BI tables. Filters never change model selection."""

from __future__ import annotations

from typing import Any

import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    *,
    dataset: str | None = None,
    entity: str | None = None,
    product: str | None = None,
    date_start: Any = None,
    date_end: Any = None,
    horizon: int | None = None,
    risk_level: str | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df
    if dataset and "source_dataset" in out.columns:
        out = out[out["source_dataset"].astype(str) == str(dataset)]
    if entity:
        col = "entity_id" if "entity_id" in out.columns else ("store_id" if "store_id" in out.columns else None)
        if col:
            out = out[out[col].astype(str) == str(entity)]
    if product:
        col = "product_key" if "product_key" in out.columns else ("sku_id" if "sku_id" in out.columns else None)
        if col:
            out = out[out[col].astype(str) == str(product)]
    date_col = next((c for c in ("forecast_date", "date") if c in out.columns), None)
    if date_col and date_start is not None:
        out = out[pd.to_datetime(out[date_col]) >= pd.to_datetime(date_start)]
    if date_col and date_end is not None:
        out = out[pd.to_datetime(out[date_col]) <= pd.to_datetime(date_end)]
    if horizon is not None and "horizon" in out.columns:
        out = out[out["horizon"].astype(int) == int(horizon)]
    if risk_level and "stockout_risk_level" in out.columns:
        out = out[out["stockout_risk_level"].astype(str) == str(risk_level)]
    return out.reset_index(drop=True)
