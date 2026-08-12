"""
Phase 6 — Feature Engineering Pipeline
========================================
Project FORESIGHT: Demand & Inventory Intelligence

Generates temporal, calendar, lag, rolling window statistics, and promotional
features for time series demand forecasting and inventory risk models.
"""

import numpy as np
import pandas as pd


def aggregate_daily_sales(
    sales_df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("sku_id",),
) -> pd.DataFrame:
    """
    Aggregate store-SKU daily sales records to a coarser entity grain while
    keeping the columns the forecasting feature pipeline expects.

    The forecasting engine is trained and served at a single entity grain
    (default: SKU level, i.e. demand summed across stores). Aggregating the
    training data to the SAME grain as inference avoids the store-SKU vs
    SKU-total scale mismatch that otherwise produces systematically
    under-scaled forecasts.

    Derived fields:
      - `units_sold` / `total_revenue` / `transaction_count` /
        `unique_customers`: summed across the grouped entities.
      - `promotion_flag`: 1 if any member store ran a promotion that day.
      - `avg_unit_price`: volume-weighted price (revenue / units); when a day
        has zero units the price is imputed with the entity median.
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
    """Extract temporal, day-of-week, seasonal, and cyclical calendar features."""
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

    # Cyclical representations
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
    """Generate lag features grouped by time-series entities (e.g. store_id, sku_id)."""
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
    Generate rolling statistics (mean, std, min, max) strictly shifted by 1
    to avoid lookahead bias and data leakage.
    """
    df = df.copy().sort_values(by=group_cols + ["date"])
    grouped = df.groupby(group_cols)[target_col]

    # Shifted series to prevent current-step target leakage
    shifted = grouped.shift(1)

    for w in windows:
        roll = shifted.groupby(df[group_cols[0]] if len(group_cols) == 1 else [df[c] for c in group_cols]).rolling(w)
        df[f"{target_col}_rolling_mean_{w}"] = roll.mean().reset_index(drop=True)
        df[f"{target_col}_rolling_std_{w}"] = roll.std().fillna(0).reset_index(drop=True)
        df[f"{target_col}_rolling_min_{w}"] = roll.min().reset_index(drop=True)
        df[f"{target_col}_rolling_max_{w}"] = roll.max().reset_index(drop=True)

    # Exponential weighted moving averages
    df[f"{target_col}_ewm_7"] = (
        shifted.groupby(df[group_cols[0]] if len(group_cols) == 1 else [df[c] for c in group_cols])
        .transform(lambda s: s.ewm(span=7, min_periods=1).mean())
    )
    df[f"{target_col}_ewm_28"] = (
        shifted.groupby(df[group_cols[0]] if len(group_cols) == 1 else [df[c] for c in group_cols])
        .transform(lambda s: s.ewm(span=28, min_periods=1).mean())
    )

    return df


def create_pricing_promotional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate pricing discount ratios and promotional indicators."""
    df = df.copy()
    if "base_price" in df.columns and "avg_unit_price" in df.columns:
        df["price_discount"] = np.maximum(0, df["base_price"] - df["avg_unit_price"])
        df["discount_pct"] = np.where(
            df["base_price"] > 0,
            df["price_discount"] / df["base_price"],
            0.0
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
    group_cols: list[str] = ["store_id", "sku_id"],
    target_col: str = "units_sold",
    drop_na_rows: bool = True,
) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline to generate complete feature matrix
    for ML model training and multi-step inference.
    """
    df = sales_df.copy()
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    # 1. Merge metadata
    if sku_df is not None:
        sku_cols = [c for c in ["sku_id", "category", "sub_category", "brand", "cost_price", "base_price", "lead_time_days", "reorder_point", "safety_stock"] if c in sku_df.columns]
        if "sku_id" in df.columns and "sku_id" in sku_cols:
            df = pd.merge(df, sku_df[sku_cols], on="sku_id", how="left")

    if store_df is not None:
        store_cols = [c for c in ["store_id", "region", "store_type", "store_size_sqft"] if c in store_df.columns]
        if "store_id" in df.columns and "store_id" in store_cols:
            df = pd.merge(df, store_df[store_cols], on="store_id", how="left")

    if calendar_df is not None:
        cal_cols = [c for c in ["date", "is_holiday", "holiday_name", "season"] if c in calendar_df.columns]
        if "date" in cal_cols:
            df = pd.merge(df, calendar_df[cal_cols], on="date", how="left")

    # 2. Calendar features
    df = create_calendar_features(df, date_col="date")

    # 3. Pricing & promo features
    df = create_pricing_promotional_features(df)

    # 4. Lag features
    df = create_lag_features(df, group_cols=group_cols, target_col=target_col, lags=[1, 2, 3, 7, 14, 21, 28, 30])

    # 5. Rolling statistics
    df = create_rolling_features(df, group_cols=group_cols, target_col=target_col, windows=[7, 14, 30])

    # 6. Clean up NaNs from lags
    if drop_na_rows:
        df = df.dropna().reset_index(drop=True)

    return df
