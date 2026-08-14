"""
Phase 6 — ML Engine Feature Compatibility Adapter
=================================================
Project FORESIGHT: Demand & Inventory Intelligence

Preserves the **legacy ML forecasting feature API** used by:
  - ``src/forecasting.py``
  - ``src/validate_ml_stack.py``
  - ``dashboard/app.py``

without rebuilding those engines.

Two roles
---------
1. **Legacy pipeline** — ``aggregate_daily_sales`` /
   ``build_forecasting_feature_matrix`` and related helpers remain available
   for SKU-level ML training on legacy-shaped sales frames (via
   ``cam_adapter.forecast_base_to_legacy_sales``).

2. **Phase 6 → legacy column bridge** — maps
   ``data/processed/features/forecast_features.parquet`` column names onto
   the legacy ``FEATURE_COLS`` expected by the ML engine.

Phase 6 is the authoritative feature store for Phase 7+ baselines.
The legacy path remains for the existing LightGBM / XGBoost / RF stack
until Phase 8 optionally migrates fully to Phase 6 names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column rename map: Phase 6 names → legacy ML FEATURE_COLS names
# ---------------------------------------------------------------------------
PHASE6_TO_LEGACY_RENAMES: dict[str, str] = {
    "dow_sin": "sin_day_of_week",
    "dow_cos": "cos_day_of_week",
    "month_sin": "sin_month",
    "month_cos": "cos_month",
    "rolling_mean_7": "units_sold_rolling_mean_7",
    "rolling_std_7": "units_sold_rolling_std_7",
    "rolling_mean_14": "units_sold_rolling_mean_14",
    "rolling_mean_30": "units_sold_rolling_mean_30",
    "rolling_std_14": "units_sold_rolling_std_14",
    "rolling_std_30": "units_sold_rolling_std_30",
}

# Features the legacy ML engine expects that Phase 6 does not emit.
LEGACY_ONLY_FEATURES: list[str] = [
    "units_sold_ewm_7",
    "units_sold_ewm_28",
]


def adapt_phase6_to_legacy_ml(df: pd.DataFrame, add_ewm: bool = True) -> pd.DataFrame:
    """
    Map a Phase 6 ``forecast_features`` frame toward legacy ML column names.

    - Renames cyclical / rolling columns to legacy names.
    - Optionally adds leakage-safe EWM features (shift(1) then ewm) so the
      existing ``FEATURE_COLS`` list can be satisfied without rebuilding
      ``src/forecasting.py``.
    """
    out = df.copy()
    rename = {k: v for k, v in PHASE6_TO_LEGACY_RENAMES.items() if k in out.columns}
    out = out.rename(columns=rename)

    if add_ewm and "units_sold" in out.columns:
        grain = [c for c in ["source_dataset", "entity_id", "product_key"] if c in out.columns]
        if not grain and "sku_id" in out.columns:
            grain = ["sku_id"]
        if "store_id" in out.columns and "sku_id" in out.columns:
            grain = ["store_id", "sku_id"]

        sort_cols = grain + ["date"] if "date" in out.columns else grain
        out = out.sort_values(sort_cols)
        shifted = out.groupby(grain, observed=True)["units_sold"].shift(1)
        out["units_sold_ewm_7"] = (
            shifted.groupby([out[c] for c in grain], observed=True)
            .transform(lambda s: s.ewm(span=7, min_periods=1).mean())
        )
        out["units_sold_ewm_28"] = (
            shifted.groupby([out[c] for c in grain], observed=True)
            .transform(lambda s: s.ewm(span=28, min_periods=1).mean())
        )

    return out


def get_compatibility_summary() -> dict:
    """Human-readable compatibility contract for reports / notebooks."""
    return {
        "phase6_output": "data/processed/features/forecast_features.parquet",
        "legacy_consumers": [
            "src/forecasting.py",
            "src/validate_ml_stack.py",
            "dashboard/app.py",
        ],
        "renames": PHASE6_TO_LEGACY_RENAMES,
        "legacy_only_features": LEGACY_ONLY_FEATURES,
        "adapter_required": True,
        "note": (
            "Phase 6 does not duplicate the legacy SKU-level feature matrix. "
            "Use adapt_phase6_to_legacy_ml() to bridge Phase 6 columns, or "
            "build_forecasting_feature_matrix() on a legacy sales frame for "
            "the existing ML training path. Phase 7 baselines should consume "
            "forecast_features.parquet directly."
        ),
    }


# ===========================================================================
# Legacy ML feature pipeline (preserved — do not delete)
# ===========================================================================

def aggregate_daily_sales(
    sales_df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("sku_id",),
) -> pd.DataFrame:
    """
    Aggregate store-SKU daily sales records to a coarser entity grain while
    keeping the columns the forecasting feature pipeline expects.
    """
    df = sales_df.copy()
    agg = {
        "units_sold": "sum",
        "total_revenue": "sum",
        "transaction_count": "sum",
        "unique_customers": "sum",
        "promotion_flag": "max",
    }
    group_cols = list(group_cols)
    out = df.groupby(["date", *group_cols], as_index=False).agg(agg)
    out["avg_unit_price"] = out["total_revenue"] / out["units_sold"].replace(0, np.nan)
    out["avg_unit_price"] = out["avg_unit_price"].fillna(
        out.groupby(group_cols)["avg_unit_price"].transform("median")
    )
    return out


def create_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Legacy calendar + cyclical features (sin_/cos_ naming)."""
    df = df.copy()
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])

    dt = df[date_col].dt
    df["year"] = dt.year
    df["month"] = dt.month
    df["quarter"] = dt.quarter
    df["day_of_month"] = dt.day
    df["day_of_week"] = dt.dayofweek
    df["day_of_year"] = dt.dayofyear
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)
    df["is_month_start"] = dt.is_month_start.astype(int)
    df["is_month_end"] = dt.is_month_end.astype(int)

    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def create_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "units_sold",
    lags: list[int] = (1, 2, 3, 7, 14, 21, 28, 30),
) -> pd.DataFrame:
    """Generate lag features grouped by time-series entities."""
    df = df.copy().sort_values(by=group_cols + ["date"])
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "units_sold",
    windows: list[int] = (7, 14, 30),
) -> pd.DataFrame:
    """
    Rolling statistics shifted by 1 to avoid lookahead bias.
    Legacy naming: ``{target}_rolling_mean_{w}``, plus EWM.
    """
    df = df.copy().sort_values(by=group_cols + ["date"])
    grouped = df.groupby(group_cols)[target_col]
    shifted = grouped.shift(1)

    for w in windows:
        roll = shifted.groupby(
            df[group_cols[0]] if len(group_cols) == 1 else [df[c] for c in group_cols]
        ).rolling(w)
        df[f"{target_col}_rolling_mean_{w}"] = roll.mean().reset_index(drop=True)
        df[f"{target_col}_rolling_std_{w}"] = roll.std().reset_index(drop=True)
        df[f"{target_col}_rolling_min_{w}"] = roll.min().reset_index(drop=True)
        df[f"{target_col}_rolling_max_{w}"] = roll.max().reset_index(drop=True)

    df[f"{target_col}_ewm_7"] = (
        shifted.groupby(
            df[group_cols[0]] if len(group_cols) == 1 else [df[c] for c in group_cols]
        ).transform(lambda s: s.ewm(span=7, min_periods=1).mean())
    )
    df[f"{target_col}_ewm_28"] = (
        shifted.groupby(
            df[group_cols[0]] if len(group_cols) == 1 else [df[c] for c in group_cols]
        ).transform(lambda s: s.ewm(span=28, min_periods=1).mean())
    )

    return df


def create_pricing_promotional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy pricing discount ratios and promotional indicators."""
    df = df.copy()
    if "base_price" in df.columns and "avg_unit_price" in df.columns:
        df["price_discount"] = np.maximum(0, df["base_price"] - df["avg_unit_price"])
        df["discount_pct"] = np.where(
            df["base_price"] > 0,
            df["price_discount"] / df["base_price"],
            0.0,
        )
    else:
        df["discount_pct"] = 0.0

    if "promotion_flag" not in df.columns:
        df["promotion_flag"] = 0

    return df


def build_forecasting_feature_matrix(
    sales_df: pd.DataFrame,
    sku_df: pd.DataFrame = None,
    store_df: pd.DataFrame = None,
    calendar_df: pd.DataFrame = None,
    group_cols: list[str] = None,
    target_col: str = "units_sold",
    drop_na_rows: bool = True,
) -> pd.DataFrame:
    """
    End-to-end legacy feature matrix for ML training / multi-step inference.

    Preserved for the existing forecasting engine. New Phase 7+ work should
    prefer ``forecast_features.parquet`` from the Phase 6 pipeline.
    """
    if group_cols is None:
        group_cols = ["store_id", "sku_id"]

    df = sales_df.copy()
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    if sku_df is not None:
        sku_cols = [
            c for c in [
                "sku_id", "category", "sub_category", "brand", "cost_price",
                "base_price", "lead_time_days", "reorder_point", "safety_stock",
            ]
            if c in sku_df.columns
        ]
        if "sku_id" in df.columns and "sku_id" in sku_cols:
            df = pd.merge(df, sku_df[sku_cols], on="sku_id", how="left")

    if store_df is not None:
        store_cols = [
            c for c in ["store_id", "region", "store_type", "store_size_sqft"]
            if c in store_df.columns
        ]
        if "store_id" in df.columns and "store_id" in store_cols:
            df = pd.merge(df, store_df[store_cols], on="store_id", how="left")

    if calendar_df is not None:
        cal_cols = [
            c for c in ["date", "is_holiday", "holiday_name", "season"]
            if c in calendar_df.columns
        ]
        if "date" in cal_cols:
            df = pd.merge(df, calendar_df[cal_cols], on="date", how="left")

    df = create_calendar_features(df, date_col="date")
    df = create_pricing_promotional_features(df)
    df = create_lag_features(
        df, group_cols=group_cols, target_col=target_col,
        lags=[1, 2, 3, 7, 14, 21, 28, 30],
    )
    df = create_rolling_features(
        df, group_cols=group_cols, target_col=target_col, windows=[7, 14, 30],
    )

    if drop_na_rows:
        df = df.dropna().reset_index(drop=True)

    return df
