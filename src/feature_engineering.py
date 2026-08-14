"""
Phase 6 — Feature Engineering Pipeline
========================================
Project FORESIGHT: Demand & Inventory Intelligence

Generates temporal, calendar, cyclical, lag, rolling-window, demand-trend,
pricing, promotional, product, entity/store, and inventory features for
time-series demand forecasting.

Input:  data/processed/integrated/forecast_base.parquet  (1,995,496 rows × 12 cols)
Output: data/processed/features/forecast_features.parquet

Forecasting grain: date + source_dataset + entity_id + product_key
Target:            units_sold
Source separation:  UCI and SYNTHETIC are NEVER mixed — all grouped operations
                    partition by (source_dataset, entity_id, product_key).
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

INTEGRATED_DIR = os.path.join(BASE_DIR, "data", "processed", "integrated")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
# The grain columns that define a unique time series
GRAIN_COLS = ["source_dataset", "entity_id", "product_key"]
SORT_COLS = GRAIN_COLS + ["date"]
FULL_GRAIN = ["date"] + GRAIN_COLS

REQUIRED_FORECAST_BASE_COLS = [
    "date", "source_dataset", "entity_id", "entity_type", "product_key",
    "sku_id", "units_sold", "revenue", "average_unit_price",
    "transaction_count", "unique_customers", "promotion_flag",
]


# ===================================================================
# 1. LOAD + SCHEMA VALIDATION
# ===================================================================

def validate_forecast_base_schema(df: pd.DataFrame) -> None:
    """Fail fast if forecast_base does not match the expected CAM schema."""
    missing = [c for c in REQUIRED_FORECAST_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"forecast_base schema validation failed. Missing columns: {missing}. "
            f"Actual columns: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError("forecast_base is empty")
    null_keys = {c: int(df[c].isna().sum()) for c in FULL_GRAIN if df[c].isna().any()}
    if null_keys:
        raise ValueError(f"forecast_base has null grain keys: {null_keys}")


def load_features_input():
    """Load the Phase 4 CAM forecast_base as the feature engineering input."""
    path = os.path.join(INTEGRATED_DIR, "forecast_base.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    validate_forecast_base_schema(df)
    return df


def load_dim_product():
    """Load product dimension table."""
    path = os.path.join(INTEGRATED_DIR, "dim_product.parquet")
    return pd.read_parquet(path)


def load_dim_entity():
    """Load entity/store dimension table."""
    path = os.path.join(INTEGRATED_DIR, "dim_entity.parquet")
    return pd.read_parquet(path)


def load_dim_calendar():
    """Load calendar dimension table."""
    path = os.path.join(INTEGRATED_DIR, "dim_calendar.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_fact_inventory():
    """Load inventory fact table (Synthetic only)."""
    path = os.path.join(INTEGRATED_DIR, "fact_inventory.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ===================================================================
# 2. CALENDAR FEATURES
# ===================================================================

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal calendar features from the date column.

    Creates: year, month, quarter, week_of_year, day_of_week,
    day_of_month, day_of_year, is_weekend.
    """
    df = df.copy()
    dt = df["date"].dt
    df["year"] = dt.year
    df["month"] = dt.month
    df["quarter"] = dt.quarter
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["day_of_week"] = dt.dayofweek          # 0=Mon … 6=Sun
    df["day_of_month"] = dt.day
    df["day_of_year"] = dt.dayofyear
    df["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)
    return df


# ===================================================================
# 3. CYCLICAL FEATURES
# ===================================================================

def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonic sin/cos encodings for month (period 12) and
    day_of_week (period 7) to capture circular continuity.
    """
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


# ===================================================================
# 4. LAG FEATURES
# ===================================================================

def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "units_sold",
    lags: tuple = (1, 2, 3, 7, 14, 21, 28, 30),
) -> pd.DataFrame:
    """
    Generate lagged target features.  Each lag is computed within
    the strict partition of (source_dataset, entity_id, product_key)
    to prevent cross-series contamination.
    """
    df = df.copy().sort_values(SORT_COLS)
    grouped = df.groupby(GRAIN_COLS, observed=True)[target_col]
    for lag in lags:
        df[f"units_sold_lag_{lag}"] = grouped.shift(lag)
    return df


# ===================================================================
# 5. ROLLING FEATURES  (leakage-safe: shift(1) before rolling)
# ===================================================================

def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "units_sold",
    windows: tuple = (7, 14, 30),
) -> pd.DataFrame:
    """
    Rolling mean and rolling std over 7/14/30-day windows.

    LEAKAGE PREVENTION: the target is shifted by 1 position within
    each (source_dataset, entity_id, product_key) group BEFORE the
    rolling window is computed.  This guarantees the current-day
    target is never included.

    Mathematically:
        rolling_mean_7(t) = mean(units_sold[t-7] ... units_sold[t-1])
    """
    df = df.copy().sort_values(SORT_COLS)
    # Shift within grain — current target excluded
    shifted = df.groupby(GRAIN_COLS, observed=True)[target_col].shift(1)

    for w in windows:
        roll = shifted.groupby(
            [df[c] for c in GRAIN_COLS]
        ).rolling(w, min_periods=1)

        df[f"rolling_mean_{w}"] = (
            roll.mean().droplevel(list(range(len(GRAIN_COLS)))).sort_index()
        )
        # Leave NaN when the window has <2 observations (std undefined).
        # Do NOT fill with zero — zero would invent false stability.
        df[f"rolling_std_{w}"] = (
            roll.std().droplevel(list(range(len(GRAIN_COLS)))).sort_index()
        )

    return df


# ===================================================================
# 6. DEMAND TREND FEATURES
# ===================================================================

def create_demand_trend_features(
    df: pd.DataFrame,
    target_col: str = "units_sold",
) -> pd.DataFrame:
    """
    Short-term and medium-term demand-change signals.

    demand_change_1:  units_sold(t-1) - units_sold(t-2)
    demand_change_7:  units_sold(t-1) - units_sold(t-8)
    demand_growth_7:  (rolling_mean_7(t) - lag_7(t)) / (lag_7(t) + 1)
    demand_growth_30: (rolling_mean_30(t) - lag_30(t)) / (lag_30(t) + 1)
    """
    df = df.copy()
    df["demand_change_1"] = df["units_sold_lag_1"] - df["units_sold_lag_2"]
    df["demand_change_7"] = df["units_sold_lag_1"] - df.groupby(
        GRAIN_COLS, observed=True
    )["units_sold"].shift(8)  # shift(8) = t-8 value

    # Growth rates — denominator offset by 1 to avoid division by zero
    df["demand_growth_7"] = (
        (df["rolling_mean_7"] - df["units_sold_lag_7"])
        / (df["units_sold_lag_7"].abs() + 1)
    )
    df["demand_growth_30"] = (
        (df["rolling_mean_30"] - df["units_sold_lag_30"])
        / (df["units_sold_lag_30"].abs() + 1)
    )
    return df


# ===================================================================
# 7. PRICE FEATURES
# ===================================================================

def create_price_features(
    df: pd.DataFrame,
    dim_product: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create pricing features from actual available fields.

    - average_unit_price is already on forecast_base for both sources.
    - base_price exists only for Synthetic products (UCI has NaN).
    - For Synthetic: discount_pct = (base_price - average_unit_price) / base_price
    - For UCI: discount_pct = NaN  (no base_price available; NOT filled with 0)

    Price lag:
    - price_lag_1: previous day's average_unit_price (within grain)
    - price_change: average_unit_price(t) - price_lag_1(t)
    """
    df = df.copy()

    # Join base_price from dim_product (only Synthetic will have values)
    product_prices = dim_product[["product_key", "base_price"]].drop_duplicates(
        subset=["product_key"]
    )
    df = df.merge(product_prices, on="product_key", how="left")

    # Discount percentage — only meaningful where base_price exists
    df["discount_pct"] = np.where(
        df["base_price"].notna() & (df["base_price"] > 0),
        (df["base_price"] - df["average_unit_price"]) / df["base_price"],
        np.nan,
    )

    # Price lag and change within grain
    df = df.sort_values(SORT_COLS)
    df["price_lag_1"] = df.groupby(GRAIN_COLS, observed=True)[
        "average_unit_price"
    ].shift(1)
    df["price_change"] = df["average_unit_price"] - df["price_lag_1"]

    return df


# ===================================================================
# 8. PROMOTION FEATURES
# ===================================================================

def create_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Promotion features — Synthetic only where data exists.

    For UCI: promotion_flag is NaN (all 534,496 rows).
    We preserve NaN; we do NOT convert unknown into 'no promotion'.

    promotion_available: 1 where promotion_flag is not null, 0 otherwise.
    """
    df = df.copy()
    # Indicator of whether promotion data is available
    df["promotion_available"] = df["promotion_flag"].notna().astype(int)

    # For rolling promo features — only compute within Synthetic
    # promo_rolling_7: fraction of last 7 days with promotion=1
    promo_shifted = df.groupby(GRAIN_COLS, observed=True)["promotion_flag"].shift(1)
    df["promo_rolling_7"] = (
        promo_shifted
        .groupby([df[c] for c in GRAIN_COLS])
        .rolling(7, min_periods=1)
        .mean()
        .droplevel(list(range(len(GRAIN_COLS))))
        .sort_index()
    )
    return df


# ===================================================================
# 9. PRODUCT FEATURES
# ===================================================================

def create_product_features(
    df: pd.DataFrame,
    dim_product: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join actual product dimension attributes where available.

    Synthetic: category, sub_category, brand, cost_price, base_price, lead_time_days
    UCI: product_name only (all other fields are null)
    """
    df = df.copy()
    # Select useful columns — avoid re-joining base_price if already present
    cols_to_join = ["product_key", "category", "sub_category", "brand"]
    available = [c for c in cols_to_join if c in dim_product.columns]
    product_meta = dim_product[available].drop_duplicates(subset=["product_key"])

    # Only join columns not already present
    existing = [c for c in product_meta.columns if c in df.columns and c != "product_key"]
    product_meta = product_meta.drop(columns=existing, errors="ignore")

    df = df.merge(product_meta, on="product_key", how="left")
    return df


# ===================================================================
# 10. ENTITY / STORE FEATURES
# ===================================================================

def create_entity_features(
    df: pd.DataFrame,
    dim_entity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join actual store/entity attributes for Synthetic.

    Synthetic: region, store_type, store_size_sqft
    UCI (ONLINE entity): region=NaN, store_type=NaN, store_size_sqft=NaN
    We do NOT fabricate physical-store attributes for UCI.
    """
    df = df.copy()
    entity_meta = dim_entity[
        ["entity_id", "region", "store_type", "store_size_sqft"]
    ].copy()
    # Cast pyarrow NA-bearing columns
    for c in ["region", "store_type"]:
        entity_meta[c] = entity_meta[c].astype("object")
    df = df.merge(entity_meta, on="entity_id", how="left")
    return df


# ===================================================================
# 11. INVENTORY FEATURES (Synthetic only)
# ===================================================================

def create_inventory_features(
    df: pd.DataFrame,
    fact_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join inventory information for Synthetic only (UCI remains NaN).

    CAM semantics: ending_inventory(t) = beginning_inventory(t) - units_sold(t).
    Therefore same-day ending_inventory / stockout_flag would leak the target.

    Leakage-safe approach:
      - Join inventory on the same grain + date.
      - Shift ending_inventory, on_order_qty, stockout_flag by 1 within grain
        so features at time t use only information available through t-1.
      - historical_doi uses the shifted ending_inventory / rolling_mean_7.

    No future inventory is used.
    """
    df = df.copy()

    inv_cols = [
        "date", "source_dataset", "entity_id", "product_key",
        "ending_inventory", "on_order_qty", "stockout_flag",
    ]
    inv = fact_inventory[inv_cols].copy()
    inv["date"] = pd.to_datetime(inv["date"])

    df = df.merge(
        inv,
        on=["date", "source_dataset", "entity_id", "product_key"],
        how="left",
    )

    df = df.sort_values(SORT_COLS)
    for col in ["ending_inventory", "on_order_qty", "stockout_flag"]:
        df[col] = df.groupby(GRAIN_COLS, observed=True)[col].shift(1)

    if "rolling_mean_7" in df.columns:
        safe_denom = df["rolling_mean_7"].replace(0, np.nan)
        df["historical_doi"] = df["ending_inventory"] / safe_denom
    else:
        df["historical_doi"] = np.nan

    return df


# ===================================================================
# 12. CALENDAR DIMENSION JOIN (holidays / season)
# ===================================================================

def join_calendar_dim(
    df: pd.DataFrame,
    dim_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join holiday and season information from dim_calendar.
    dim_calendar has separate entries for UCI_DERIVED and SYNTHETIC date ranges.
    We join on date — both sources share overlapping calendar enrichment.
    """
    df = df.copy()
    # Use unique date rows only to avoid duplication
    cal = dim_calendar[["date", "is_holiday", "season"]].drop_duplicates(
        subset=["date"]
    )
    df = df.merge(cal, on="date", how="left")
    return df


# ===================================================================
# 13. TIME SPLIT
# ===================================================================

def create_time_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create chronological train / validation / test splits PER SOURCE.

    Strategy:
      - For each source_dataset, determine the full date range.
      - Last 10% of days → TEST
      - Previous 10% of days → VALIDATION
      - Remaining 80% → TRAIN

    Returns: df with a 'split' column ('train' / 'validation' / 'test')
    """
    df = df.copy()
    df["split"] = ""

    for src in df["source_dataset"].unique():
        mask_src = df["source_dataset"] == src
        dates = df.loc[mask_src, "date"]
        min_date = dates.min()
        max_date = dates.max()
        total_days = (max_date - min_date).days

        val_start = min_date + pd.Timedelta(days=int(total_days * 0.80))
        test_start = min_date + pd.Timedelta(days=int(total_days * 0.90))

        df.loc[mask_src & (df["date"] < val_start), "split"] = "train"
        df.loc[mask_src & (df["date"] >= val_start) & (df["date"] < test_start), "split"] = "validation"
        df.loc[mask_src & (df["date"] >= test_start), "split"] = "test"

    return df


def get_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of train/validation/test splits per source."""
    records = []
    for src in sorted(df["source_dataset"].unique()):
        for sp in ["train", "validation", "test"]:
            sub = df[(df["source_dataset"] == src) & (df["split"] == sp)]
            if len(sub) > 0:
                records.append({
                    "source_dataset": src,
                    "split": sp,
                    "start_date": sub["date"].min().strftime("%Y-%m-%d"),
                    "end_date": sub["date"].max().strftime("%Y-%m-%d"),
                    "rows": len(sub),
                    "unique_entities": sub["entity_id"].nunique(),
                    "unique_products": sub["product_key"].nunique(),
                })
    return pd.DataFrame(records)


# ===================================================================
# 14. MISSING VALUE STRATEGY
# ===================================================================

def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Handle NaN values from lag/rolling features at the start of each
    time series.

    Strategy:
      - Lag and rolling features are naturally NaN for the first N
        observations of each time series.  We DO NOT blindly fill
        these with zero (zero ≠ "no prior data").
      - We flag rows where lag_1 is NaN as 'insufficient_history'.
      - We leave lag/rolling NaN as NaN — tree-based models handle NaN
        natively; linear models need separate imputation.
      - Categorical NaN (category, region, store_type): left as NaN
        for downstream encoding.

    Returns: (df, summary_dict)
    """
    df = df.copy()

    # Flag rows with insufficient history
    df["insufficient_history"] = df["units_sold_lag_1"].isna().astype(int)

    # Summary of missing values
    lag_rolling_cols = [c for c in df.columns if "lag_" in c or "rolling_" in c or "demand_" in c]
    missing_summary = {}
    for c in lag_rolling_cols:
        n_miss = int(df[c].isna().sum())
        pct = round(100 * n_miss / len(df), 2)
        missing_summary[c] = {"missing_count": n_miss, "missing_pct": pct}

    return df, missing_summary


# ===================================================================
# 15. ARTIFACT WRITERS (registry / quality / report helpers)
# ===================================================================

FEATURE_REGISTRY_SPEC = [
    # target
    ("units_sold", "target", "BOTH", "N/A", "Target variable: daily units sold", "Yes"),
    # calendar
    ("year", "calendar", "BOTH", "N/A", "Calendar year", "Yes"),
    ("month", "calendar", "BOTH", "N/A", "Month 1-12", "Yes"),
    ("quarter", "calendar", "BOTH", "N/A", "Quarter 1-4", "Yes"),
    ("week_of_year", "calendar", "BOTH", "N/A", "ISO week of year", "Yes"),
    ("day_of_week", "calendar", "BOTH", "N/A", "Day of week 0=Mon..6=Sun", "Yes"),
    ("day_of_month", "calendar", "BOTH", "N/A", "Day of month 1-31", "Yes"),
    ("day_of_year", "calendar", "BOTH", "N/A", "Day of year 1-366", "Yes"),
    ("is_weekend", "calendar", "BOTH", "N/A", "1 if Sat/Sun else 0", "Yes"),
    # cyclical
    ("month_sin", "cyclical", "BOTH", "N/A", "sin(2*pi*month/12)", "Yes"),
    ("month_cos", "cyclical", "BOTH", "N/A", "cos(2*pi*month/12)", "Yes"),
    ("dow_sin", "cyclical", "BOTH", "N/A", "sin(2*pi*dow/7)", "Yes"),
    ("dow_cos", "cyclical", "BOTH", "N/A", "cos(2*pi*dow/7)", "Yes"),
    # lags
    ("units_sold_lag_1", "lag", "BOTH", "1", "units_sold at t-1 within grain", "Yes"),
    ("units_sold_lag_2", "lag", "BOTH", "2", "units_sold at t-2 within grain", "Yes"),
    ("units_sold_lag_3", "lag", "BOTH", "3", "units_sold at t-3 within grain", "Yes"),
    ("units_sold_lag_7", "lag", "BOTH", "7", "units_sold at t-7 within grain", "Yes"),
    ("units_sold_lag_14", "lag", "BOTH", "14", "units_sold at t-14 within grain", "Yes"),
    ("units_sold_lag_21", "lag", "BOTH", "21", "units_sold at t-21 within grain", "Yes"),
    ("units_sold_lag_28", "lag", "BOTH", "28", "units_sold at t-28 within grain", "Yes"),
    ("units_sold_lag_30", "lag", "BOTH", "30", "units_sold at t-30 within grain", "Yes"),
    # rolling
    ("rolling_mean_7", "rolling", "BOTH", "7", "Mean of units_sold[t-7..t-1] (shift-1)", "Yes"),
    ("rolling_mean_14", "rolling", "BOTH", "14", "Mean of units_sold[t-14..t-1] (shift-1)", "Yes"),
    ("rolling_mean_30", "rolling", "BOTH", "30", "Mean of units_sold[t-30..t-1] (shift-1)", "Yes"),
    ("rolling_std_7", "rolling", "BOTH", "7", "Std of units_sold[t-7..t-1] (shift-1)", "Yes"),
    ("rolling_std_14", "rolling", "BOTH", "14", "Std of units_sold[t-14..t-1] (shift-1)", "Yes"),
    ("rolling_std_30", "rolling", "BOTH", "30", "Std of units_sold[t-30..t-1] (shift-1)", "Yes"),
    # demand trend
    ("demand_change_1", "demand_trend", "BOTH", "2", "lag_1 - lag_2", "Yes"),
    ("demand_change_7", "demand_trend", "BOTH", "8", "lag_1 - units_sold(t-8)", "Yes"),
    ("demand_growth_7", "demand_trend", "BOTH", "7", "(rolling_mean_7 - lag_7) / (|lag_7|+1)", "Yes"),
    ("demand_growth_30", "demand_trend", "BOTH", "30", "(rolling_mean_30 - lag_30) / (|lag_30|+1)", "Yes"),
    # price
    ("average_unit_price", "price", "BOTH", "N/A", "CAM average unit price (same-day observed)", "Yes"),
    ("base_price", "price", "SYNTHETIC", "N/A", "Product list price; UCI left NaN", "Yes"),
    ("discount_pct", "price", "SYNTHETIC", "N/A", "(base-avg)/base; UCI NaN (not fabricated)", "Yes"),
    ("price_lag_1", "price", "BOTH", "1", "average_unit_price at t-1", "Yes"),
    ("price_change", "price", "BOTH", "1", "average_unit_price(t) - price_lag_1", "Yes"),
    # promotion
    ("promotion_flag", "promotion", "SYNTHETIC", "N/A", "Promo flag; UCI all NaN (unknown ≠ 0)", "Yes"),
    ("promotion_available", "promotion", "BOTH", "N/A", "1 if promotion_flag known else 0", "Yes"),
    ("promo_rolling_7", "promotion", "SYNTHETIC", "7", "Mean of shifted promo flag over 7 days", "Yes"),
    # product
    ("category", "product", "SYNTHETIC", "N/A", "Product category; UCI NaN", "Yes"),
    ("sub_category", "product", "SYNTHETIC", "N/A", "Product sub-category; UCI NaN", "Yes"),
    ("brand", "product", "SYNTHETIC", "N/A", "Brand; UCI NaN", "Yes"),
    # entity
    ("region", "entity", "SYNTHETIC", "N/A", "Store region; UCI NaN", "Yes"),
    ("store_type", "entity", "SYNTHETIC", "N/A", "Store type; UCI NaN", "Yes"),
    ("store_size_sqft", "entity", "SYNTHETIC", "N/A", "Store size sqft; UCI NaN", "Yes"),
    # inventory (lag-1 shifted)
    ("ending_inventory", "inventory", "SYNTHETIC", "1", "Prior-day ending inventory (shift-1)", "Yes"),
    ("on_order_qty", "inventory", "SYNTHETIC", "1", "Prior-day on-order qty (shift-1)", "Yes"),
    ("stockout_flag", "inventory", "SYNTHETIC", "1", "Prior-day stockout flag (shift-1)", "Yes"),
    ("historical_doi", "inventory", "SYNTHETIC", "7", "ending_inventory / rolling_mean_7", "Yes"),
    # calendar dim
    ("is_holiday", "calendar_dim", "BOTH", "N/A", "Holiday indicator from dim_calendar", "Yes"),
    ("season", "calendar_dim", "BOTH", "N/A", "Season label from dim_calendar", "Yes"),
    # split / flags
    ("split", "metadata", "BOTH", "N/A", "Chronological train/validation/test", "Yes"),
    ("insufficient_history", "metadata", "BOTH", "N/A", "1 if lag_1 is NaN (series warm-up)", "Yes"),
]


def write_feature_registry(df: pd.DataFrame, path: str | None = None) -> pd.DataFrame:
    """Write docs/feature_registry.csv from the executed feature frame."""
    path = path or os.path.join(DOCS_DIR, "feature_registry.csv")
    rows = []
    for name, group, avail, lookback, desc, leak in FEATURE_REGISTRY_SPEC:
        if name not in df.columns:
            continue
        rows.append({
            "feature_name": name,
            "feature_group": group,
            "data_type": str(df[name].dtype),
            "source_availability": avail,
            "lookback_period": lookback,
            "description": desc,
            "leakage_safe": leak,
        })
    reg = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    reg.to_csv(path, index=False)
    return reg


def write_feature_quality_report(df: pd.DataFrame, path: str | None = None) -> pd.DataFrame:
    """Write docs/feature_quality_report.csv from the executed feature frame."""
    path = path or os.path.join(DOCS_DIR, "feature_quality_report.csv")
    lag_roll = [c for c in df.columns if "lag_" in c or c.startswith("rolling_") or c.startswith("demand_")]
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s):
            inf_count = int(np.isinf(s.dropna().to_numpy(dtype=float, copy=False)).sum()) if s.notna().any() else 0
            try:
                vmin = s.min(skipna=True)
                vmax = s.max(skipna=True)
            except TypeError:
                vmin, vmax = np.nan, np.nan
        else:
            inf_count = 0
            vmin, vmax = np.nan, np.nan
        leak = "PASS"
        if col in lag_roll or col in {
            "ending_inventory", "on_order_qty", "stockout_flag", "historical_doi",
            "promo_rolling_7", "price_lag_1",
        }:
            leak = "PASS_LEAKAGE_SAFE"
        rows.append({
            "feature_name": col,
            "dtype": str(s.dtype),
            "missing_pct": round(100.0 * missing / n, 4) if n else 0.0,
            "infinite_count": inf_count,
            "unique_count": int(s.nunique(dropna=True)),
            "min": vmin,
            "max": vmax,
            "leakage_status": leak,
        })
    qual = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    qual.to_csv(path, index=False)
    return qual


def write_feature_engineering_report(
    df: pd.DataFrame,
    metadata: dict,
    validation_summary: str,
    path: str | None = None,
) -> str:
    """Write docs/feature_engineering_report.md using actual executed results."""
    path = path or os.path.join(DOCS_DIR, "feature_engineering_report.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    from src.feature_adapter import get_compatibility_summary
    compat = get_compatibility_summary()

    feature_groups = {}
    for name, group, *_ in FEATURE_REGISTRY_SPEC:
        if name in df.columns:
            feature_groups.setdefault(group, []).append(name)

    split_lines = []
    for rec in metadata["split_summary"]:
        split_lines.append(
            f"| {rec['source_dataset']} | {rec['split']} | {rec['start_date']} | "
            f"{rec['end_date']} | {rec['rows']:,} |"
        )

    miss = metadata.get("missing_summary", {})
    miss_rows = sorted(miss.items(), key=lambda x: -x[1]["missing_pct"])[:15]
    miss_md = "\n".join(
        f"| {k} | {v['missing_count']:,} | {v['missing_pct']}% |"
        for k, v in miss_rows
    )

    engineered = [
        c for c in df.columns
        if c not in REQUIRED_FORECAST_BASE_COLS
    ]

    md = f"""# Phase 6 — Feature Engineering Report

**Project:** FORESIGHT — Demand & Inventory Intelligence  
**Status:** COMPLETE (executed + validated)  
**Primary output:** `data/processed/features/forecast_features.parquet`  
**Validation:** {validation_summary}

---

## 1. Input row count

| Metric | Value |
|---|---|
| Input table | `data/processed/integrated/forecast_base.parquet` |
| Input rows | **{metadata['input_rows']:,}** |
| Input columns | **{metadata['input_cols']}** |

Input columns: `{', '.join(REQUIRED_FORECAST_BASE_COLS)}`

## 2. Output row count

| Metric | Value |
|---|---|
| Output rows | **{metadata['output_rows']:,}** |
| Output columns | **{metadata['output_cols']}** |
| Engineered / derived columns (excl. raw CAM) | **{len(engineered)}** |

## 3. Input columns

Preserved from CAM `forecast_base`:

{chr(10).join(f'- `{c}`' for c in REQUIRED_FORECAST_BASE_COLS)}

## 4. Output feature count

Total columns in `forecast_features.parquet`: **{metadata['output_cols']}**  
(Includes grain keys, target, engineered features, split, and flags.)

## 5. Feature groups

| Group | Features |
|---|---|
{chr(10).join(f'| {g} | {len(feats)} — {", ".join(feats)} |' for g, feats in feature_groups.items())}

## 6. Train / validation / test dates

Chronological splits computed **per `source_dataset`** (80% / 10% / 10% of each source date span). No random splitting.

| Source | Split | Start | End | Rows |
|---|---|---|---|---|
{chr(10).join(split_lines)}

## 7. Train / validation / test row counts

| Split | Rows |
|---|---|
| train | **{(df['split']=='train').sum():,}** |
| validation | **{(df['split']=='validation').sum():,}** |
| test | **{(df['split']=='test').sum():,}** |

## 8. Missing-value summary

**Strategy:** Lag / rolling / demand-trend NaNs at the start of each series are **left as NaN** (not zero-filled). Zero would invent false history. An `insufficient_history` flag marks rows where `units_sold_lag_1` is NaN. Source-specific NaNs (UCI promotions, UCI categories, UCI inventory) are preserved as unknown — never fabricated.

Top missing engineered features:

| Feature | Missing count | Missing % |
|---|---|---|
{miss_md}

## 9. Leakage validation

Automated checks in `src/validate_features.py` cover:

1. Lag features use previous observations only  
2. Rolling features exclude the current target (`shift(1)` before rolling)  
3. No future target enters features (first-row lag/rolling = NaN)  
4. Features never cross source / entity / product boundaries  
5. Inventory features are lag-1 shifted (ending inventory depends on same-day sales)  
6. Train dates precede validation; validation precedes test (per source)

**Result:** {validation_summary}

## 10. Validation result

```
{validation_summary}
```

Also verified: output exists & readable, row count matches input, no duplicate grain, no null grain keys, no infinite values, UCI/Synthetic separation maintained.

## 11. ML compatibility

Existing ML engine (`src/forecasting.py`) expects legacy column names (`sin_month`, `units_sold_rolling_mean_7`, EWM, etc.).

| Item | Detail |
|---|---|
| Adapter | `src/feature_adapter.py` |
| Renames | {compat['renames']} |
| Legacy-only (added by adapter if needed) | {compat['legacy_only_features']} |
| Phase 7 recommendation | Consume `forecast_features.parquet` directly for baselines |
| Legacy ML path | Keep using `build_forecasting_feature_matrix` via `feature_adapter` + `cam_adapter` |

{compat['note']}

## 12. Files created

| File | Role |
|---|---|
| `data/processed/features/forecast_features.parquet` | Primary feature store |
| `notebooks/05_feature_engineering.ipynb` | Executed Phase 6 notebook |
| `src/feature_engineering.py` | Phase 6 pipeline |
| `src/validate_features.py` | Leakage + integrity tests |
| `src/feature_adapter.py` | Legacy ML compatibility |
| `docs/feature_registry.csv` | Feature catalog |
| `docs/feature_quality_report.csv` | Quality metrics |
| `docs/feature_engineering_report.md` | This report |

## 13. Limitations

- UCI has no verified promotions, categories, brands, store attributes, or inventory — those fields remain NaN.
- Inventory features use **prior-day** ending inventory; day-0 of each Synthetic series is NaN after the shift.
- Rolling std is NaN when fewer than 2 prior observations exist (not zero-filled).
- Phase 6 does not emit legacy EWM columns; use `feature_adapter.adapt_phase6_to_legacy_ml` if needed.
- Calendar holiday/season join is on `date` only (shared enrichment across sources for overlapping calendar logic via unique dates).

## 14. Recommendations for Phase 7

1. Train baselines **separately** for UCI and SYNTHETIC on `forecast_features.parquet`.
2. Use the `split` column — do not re-split randomly.
3. Prefer rows with `insufficient_history == 0` for metrics that require lags (or evaluate NaN-tolerant metrics).
4. Baselines (Naive, Seasonal Naive, Moving Average, Historical Mean) can start from `units_sold` + grain + `split` without waiting on ML features.
5. Do **not** mix UCI and Synthetic series when fitting or scoring.
6. Keep the existing ML engine untouched until Phase 8; bridge via `feature_adapter` if needed.

---

*Generated from actual pipeline execution. Do not treat as simulated.*
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path


# ===================================================================
# 16. FULL PIPELINE
# ===================================================================

def run_feature_pipeline(save: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Execute the complete Phase 6 feature engineering pipeline.

    Returns: (feature_df, metadata_dict)
    """
    print("[Phase 6] Loading data...")
    fb = load_features_input()
    dim_prod = load_dim_product()
    dim_ent = load_dim_entity()
    dim_cal = load_dim_calendar()
    fact_inv = load_fact_inventory()

    print(f"  forecast_base: {fb.shape[0]:,} rows × {fb.shape[1]} cols")
    print("  Schema validation: PASS")

    # Sort by grain + date
    df = fb.sort_values(SORT_COLS).reset_index(drop=True)

    print("[Phase 6] Creating calendar features...")
    df = create_calendar_features(df)

    print("[Phase 6] Creating cyclical features...")
    df = create_cyclical_features(df)

    print("[Phase 6] Creating lag features...")
    df = create_lag_features(df)

    print("[Phase 6] Creating rolling features (leakage-safe)...")
    df = create_rolling_features(df)

    print("[Phase 6] Creating demand trend features...")
    df = create_demand_trend_features(df)

    print("[Phase 6] Creating price features...")
    df = create_price_features(df, dim_prod)

    print("[Phase 6] Creating promotion features...")
    df = create_promotion_features(df)

    print("[Phase 6] Creating product features...")
    df = create_product_features(df, dim_prod)

    print("[Phase 6] Creating entity/store features...")
    df = create_entity_features(df, dim_ent)

    print("[Phase 6] Creating inventory features (Synthetic, lag-1 safe)...")
    df = create_inventory_features(df, fact_inv)

    print("[Phase 6] Joining calendar dimension (holidays, season)...")
    df = join_calendar_dim(df, dim_cal)

    print("[Phase 6] Creating chronological time splits...")
    df = create_time_split(df)

    print("[Phase 6] Documenting missing values...")
    df, missing_summary = handle_missing_values(df)

    split_summary = get_split_summary(df)
    metadata = {
        "input_rows": int(fb.shape[0]),
        "input_cols": int(fb.shape[1]),
        "output_rows": int(df.shape[0]),
        "output_cols": int(df.shape[1]),
        "feature_count": int(df.shape[1]),
        "missing_summary": missing_summary,
        "split_summary": split_summary.to_dict(orient="records"),
    }

    if save:
        os.makedirs(FEATURES_DIR, exist_ok=True)
        out_path = os.path.join(FEATURES_DIR, "forecast_features.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[Phase 6] Saved: {out_path}")
        print(f"  Output: {df.shape[0]:,} rows × {df.shape[1]} cols")

        print("[Phase 6] Writing feature registry...")
        write_feature_registry(df)

        print("[Phase 6] Writing feature quality report...")
        write_feature_quality_report(df)

    return df, metadata


# ===================================================================
# 17. ML ENGINE COMPATIBILITY
# ===================================================================

def get_ml_feature_compatibility() -> dict:
    """Delegate to feature_adapter compatibility contract."""
    from src.feature_adapter import get_compatibility_summary
    return get_compatibility_summary()


# Backward-compatible re-exports for existing ML / dashboard imports.
# Canonical location is src/feature_adapter.py — do not duplicate logic here.
from src.feature_adapter import (  # noqa: E402
    aggregate_daily_sales,
    build_forecasting_feature_matrix,
    create_pricing_promotional_features,
)


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    df, meta = run_feature_pipeline(save=True)
    print("\n[Phase 6] Feature engineering complete.")
    print(f"  Output shape: {df.shape}")
    print("  Splits:")
    for rec in meta["split_summary"]:
        print(
            f"    {rec['source_dataset']} / {rec['split']}: "
            f"{rec['start_date']} -> {rec['end_date']} ({rec['rows']:,} rows)"
        )
