"""Documented input/output schemas for Phase 12 inference.

Feature names are taken from Phase 11 / Phase 8 contracts. Do not invent columns.
"""

from __future__ import annotations

GRAIN = ["date", "source_dataset", "entity_id", "product_key"]
KEY_COLUMNS = ["source_dataset", "entity_id", "product_key"]
DATE_COLUMNS = ("date", "forecast_date", "origin_date")

OUTPUT_COLUMNS = [
    "forecast_date",
    "source_dataset",
    "entity_id",
    "product_key",
    "horizon",
    "prediction",
    "lower_bound",
    "upper_bound",
    "model_name",
    "model_version",
    "generated_at",
]
OPTIONAL_OUTPUT = ["actual", "origin_date"]

# Phase 8 / 11 h=1 features (both datasets).
NUMERIC_FEATURES_BOTH = [
    "year", "month", "quarter", "week_of_year", "day_of_week",
    "day_of_month", "day_of_year", "is_weekend",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_holiday",
    "units_sold_lag_1", "units_sold_lag_2", "units_sold_lag_3",
    "units_sold_lag_7", "units_sold_lag_14", "units_sold_lag_21",
    "units_sold_lag_28", "units_sold_lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_std_7", "rolling_std_14", "rolling_std_30",
    "demand_change_1", "demand_change_7", "demand_growth_7", "demand_growth_30",
    "average_unit_price", "price_lag_1",
]
NUMERIC_FEATURES_SYNTHETIC_EXTRA = [
    "base_price", "discount_pct", "price_change",
    "promotion_flag", "promotion_available", "promo_rolling_7",
    "store_size_sqft",
    "ending_inventory", "on_order_qty", "stockout_flag", "historical_doi",
]
CATEGORICAL_FEATURES_BOTH = ["season"]
CATEGORICAL_FEATURES_SYNTHETIC = [
    "category", "sub_category", "brand", "region", "store_type",
]
LEAKAGE_FORBIDDEN = (
    "units_sold", "revenue", "transaction_count", "unique_customers",
)
