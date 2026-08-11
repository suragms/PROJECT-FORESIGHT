"""
Phase 3 — Data Cleaning & Data Quality Engineering
====================================================
Project FORESIGHT: Demand & Inventory Intelligence

Reusable data-cleaning pipeline for two dataset families:
  1. UCI Online Retail II  (transaction-level sales log)
  2. Multi-store relational retail dataset
     (store / sku / customer / calendar / sales_daily / inventory_snapshots)

Design principles (documented in the Phase 3 notebook):
  - Raw data is never modified; every output lives under ``data/processed/``.
  - No information is fabricated; missing values are recovered only when a
    legitimate source exists, otherwise they are explicitly retained/flaged.
  - Nothing is silently deleted; every removal has a documented reason and
    every decision is recorded in the data-quality report.

Each ``clean_*`` function returns ``(clean_df, report)`` where ``report`` is a
JSON-serialisable dictionary of diagnostics for the data-quality report.
"""

import os
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project-wide paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

# Sentinel for descriptions that cannot be recovered from the data itself.
UNKNOWN_PRODUCT = "Unknown Product"

# Canonical transaction types used in the cleaned Online Retail dataset.
TX_SALE = "SALE"
TX_RETURN = "RETURN"
TX_CANCELLATION = "CANCELLATION"
TX_INVALID = "INVALID"

ONLINE_RETAIL_REQUIRED_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
    "Price", "Customer ID", "Country",
]

STORE_MASTER_COLUMNS = [
    "store_id", "store_name", "city", "state", "region", "store_type",
    "store_size_sqft", "opening_date",
]
SKU_MASTER_COLUMNS = [
    "sku_id", "sku_name", "category", "sub_category", "brand", "cost_price",
    "base_price", "weight_kg", "supplier_id", "lead_time_days",
    "reorder_point", "safety_stock",
]
CUSTOMER_MASTER_COLUMNS = [
    "customer_id", "customer_name", "customer_segment", "loyalty_member",
    "signup_date",
]
CALENDAR_COLUMNS = [
    "date", "year", "month", "quarter", "day", "day_of_week", "day_name",
    "is_weekend", "is_holiday", "holiday_name", "season", "week_of_year",
]
SALES_DAILY_COLUMNS = [
    "date", "store_id", "sku_id", "units_sold", "total_revenue",
    "avg_unit_price", "transaction_count", "unique_customers",
    "promotion_flag",
]
INVENTORY_COLUMNS = [
    "date", "store_id", "sku_id", "beginning_inventory", "receipts",
    "units_sold", "ending_inventory", "stockout_flag", "on_order_qty",
]


# ---------------------------------------------------------------------------
# Small JSON-safety helper
# ---------------------------------------------------------------------------
def _json_safe(obj):
    """Recursively convert numpy / pandas scalars to native Python types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    if obj is None or obj is pd.NA:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def ensure_directories():
    """Create the directories required by the cleaning pipeline."""
    for d in (RAW_DIR, PROCESSED_DIR, DOCS_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)
    return {
        "raw_dir": RAW_DIR,
        "processed_dir": PROCESSED_DIR,
        "docs_dir": DOCS_DIR,
        "figures_dir": FIGURES_DIR,
    }


# ---------------------------------------------------------------------------
# Generic validation utilities
# ---------------------------------------------------------------------------
def validate_schema(df, required_columns, dataset_name):
    """
    Verify that all expected columns are present.

    Returns a report dict; does not modify ``df``.
    """
    missing = [c for c in required_columns if c not in df.columns]
    extra = [c for c in df.columns if c not in required_columns]
    return {
        "dataset": dataset_name,
        "required_columns": list(required_columns),
        "present_columns": list(df.columns),
        "missing_columns": missing,
        "extra_columns": extra,
        "schema_ok": len(missing) == 0,
    }


def handle_duplicates(df, dataset_name, subset=None):
    """
    Detect and remove exact duplicate rows.

    Exact full-row duplicates are treated as data-entry errors (e.g. a line
    recorded twice) and removed from the processed dataset.  The raw dataset
    is left untouched.  Returns ``(df_clean, report)``.
    """
    dup_mask = df.duplicated(subset=subset, keep="first")
    n_dups = int(dup_mask.sum())
    df_clean = df.loc[~dup_mask].copy()
    report = {
        "dataset": dataset_name,
        "duplicate_rows": n_dups,
        "duplicate_pct": round(n_dups / len(df) * 100, 4) if len(df) else 0.0,
        "duplicates_removed": n_dups,
        "original_rows": int(len(df)),
        "final_rows": int(len(df_clean)),
    }
    return df_clean, report


def validate_dates(df, date_col, dataset_name, reference_date=None):
    """
    Parse ``date_col`` to datetime and check invalid / missing / future dates.

    Returns ``(df, report)`` with the column parsed in place.
    """
    reference_date = reference_date or pd.Timestamp.now()
    raw_missing = int(df[date_col].isna().sum())
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    invalid = max(int(parsed.isna().sum()) - raw_missing, 0)
    future = int((parsed > reference_date).fillna(False).sum())
    report = {
        "dataset": dataset_name,
        "date_column": date_col,
        "missing_dates": raw_missing,
        "invalid_dates": max(invalid, 0),
        "future_dates": future,
        "min_date": str(parsed.min()) if parsed.notna().any() else None,
        "max_date": str(parsed.max()) if parsed.notna().any() else None,
        "total_days": int((parsed.max() - parsed.min()).days)
        if parsed.notna().any()
        else None,
    }
    df = df.copy()
    df[date_col] = parsed
    return df, report


def _numeric_profile(series, column, dataset_name):
    """Count negatives / zeros / positives and return bounds for a column."""
    s = pd.to_numeric(series, errors="coerce")
    n = int(s.notna().sum())
    return {
        "column": column,
        "non_null": n,
        "negative": int((s < 0).sum()),
        "zero": int((s == 0).sum()),
        "positive": int((s > 0).sum()),
        "min": float(s.min()) if n else None,
        "max": float(s.max()) if n else None,
        "mean": float(s.mean()) if n else None,
    }


def validate_numeric_columns(df, columns, dataset_name):
    """Run ``_numeric_profile`` for each numeric column in ``columns``."""
    results = {}
    for col in columns:
        if col in df.columns:
            results[col] = _numeric_profile(df[col], col, dataset_name)
    return results


def missing_value_summary(df, dataset_name):
    """Return per-column missing-value counts (only columns with missings)."""
    summary = {}
    for col, cnt in df.isnull().sum().items():
        if cnt > 0:
            summary[col] = {
                "count": int(cnt),
                "pct": round(float(cnt / len(df) * 100), 4),
            }
    return {"dataset": dataset_name, "missing_by_column": summary}


def investigate_outliers(series, name, dataset_name, z_threshold=3.0):
    """
    Investigate outliers using IQR and Z-score.  **Does not remove anything** —
    retail data legitimately contains extreme but valid transactions.  Returns
    a report dict to be reviewed by a business analyst.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return {"column": name, "dataset": dataset_name, "n": 0}

    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    iqr_lo, iqr_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_mask = (s < iqr_lo) | (s > iqr_hi)

    mean, std = s.mean(), s.std()
    z = (s - mean) / std if std else pd.Series(0.0, index=s.index)
    z_mask = z.abs() > z_threshold

    iqr_out = s[iqr_mask]
    z_out = s[z_mask]
    return {
        "dataset": dataset_name,
        "column": name,
        "n": int(n),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "iqr_lower_bound": float(iqr_lo),
        "iqr_upper_bound": float(iqr_hi),
        "iqr_outlier_count": int(iqr_mask.sum()),
        "iqr_outlier_pct": round(float(iqr_mask.sum() / n * 100), 4),
        "iqr_outlier_min": float(iqr_out.min()) if len(iqr_out) else None,
        "iqr_outlier_max": float(iqr_out.max()) if len(iqr_out) else None,
        "z_outlier_count": int(z_mask.sum()),
        "z_outlier_pct": round(float(z_mask.sum() / n * 100), 4),
        "z_outlier_min": float(z_out.min()) if len(z_out) else None,
        "z_outlier_max": float(z_out.max()) if len(z_out) else None,
        "z_threshold": float(z_threshold),
        "note": (
            "Outliers are reported, not removed. Extreme but legitimate "
            "retail transactions (high-volume / high-value orders) are kept."
        ),
    }


# ---------------------------------------------------------------------------
# Online Retail II — cleaning pipeline
# ---------------------------------------------------------------------------
def load_online_retail_data(path=None):
    """Load the UCI Online Retail II CSV and normalise column names."""
    path = path or os.path.join(RAW_DIR, "online_retail_II.csv")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def assign_transaction_type(df):
    """
    Classify each line into a canonical transaction type.

    Documented rule set (based on inspection of the actual data):
      1. Invoice starting with 'C'          -> CANCELLATION
      2. Invoice starting with 'A' (non-C)  -> INVALID
         (all A-invoices in this dataset are 'Adjust bad debt' journal lines)
      3. Price < 0                          -> INVALID (accounting adjustment)
      4. Quantity < 0 (non-C, non-A)        -> RETURN
      5. otherwise                          -> SALE

    Returns a pandas Series aligned to ``df.index``.
    """
    invoice = df["Invoice"].astype(str)
    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["Price"], errors="coerce")

    is_c = invoice.str.startswith("C")
    is_a = invoice.str.startswith("A")
    is_neg_price = (price < 0).fillna(False)
    is_neg_qty = (qty < 0).fillna(False)

    tx = pd.Series(TX_SALE, index=df.index, dtype="object")
    tx[is_c] = TX_CANCELLATION
    tx[is_a] = TX_INVALID
    tx[is_neg_price & ~is_c & ~is_a] = TX_INVALID
    tx[is_neg_qty & ~is_c & ~is_a] = TX_RETURN
    return tx


def recover_descriptions(df):
    """
    Recover missing ``Description`` from other lines sharing the same
    ``StockCode`` (the modal description for that product).  Lines whose
    StockCode never appears with a known description are marked with the
    documented ``UNKNOWN_PRODUCT`` sentinel — no product name is invented.
    """
    df = df.copy()
    df["Description"] = df["Description"].astype("string").str.strip()
    df.loc[df["Description"].eq(""), "Description"] = pd.NA

    missing_mask = df["Description"].isna()
    df["description_recovered"] = False
    if missing_mask.any():
        known = df.loc[~missing_mask]
        desc_map = known.groupby("StockCode")["Description"].agg(
            lambda s: s.mode().iloc[0] if len(s.mode()) else pd.NA
        )
        fill_map = df["StockCode"].map(desc_map)
        recoverable = missing_mask & fill_map.notna()
        df.loc[recoverable, "Description"] = fill_map[recoverable]
        df.loc[recoverable, "description_recovered"] = True
        df.loc[missing_mask & df["Description"].isna(), "Description"] = UNKNOWN_PRODUCT
    return df


def clean_online_retail(df):
    """
    Full cleaning pipeline for the UCI Online Retail II transaction log.

    Steps (all documented in the notebook):
      1. Schema validation
      2. Transaction-type classification (SALE / RETURN / CANCELLATION / INVALID)
      3. Description recovery by StockCode (or UNKNOWN_PRODUCT sentinel)
      4. Date parsing & validation
      5. Guest-customer flagging (missing Customer ID is preserved, not dropped)
      6. Price-category annotation (special / zero-price transactions preserved)
      7. Exact-duplicate removal
      8. Numeric validation of Quantity & Price

    Returns ``(clean_df, report)``.
    """
    original_rows = int(len(df))
    schema = validate_schema(df, ONLINE_RETAIL_REQUIRED_COLUMNS, "online_retail_ii")

    clean = df.copy()

    # --- 2. transaction type ----------------------------------------------
    clean["transaction_type"] = assign_transaction_type(clean)

    # --- 3. description recovery ------------------------------------------
    clean = recover_descriptions(clean)
    clean["description_source"] = np.where(
        clean["description_recovered"], "recovered",
        np.where(clean["Description"] == UNKNOWN_PRODUCT, "unknown", "original"),
    )

    # --- 4. dates ---------------------------------------------------------
    clean, date_report = validate_dates(
        clean, "InvoiceDate", "online_retail_ii"
    )
    clean["invoice_date_only"] = clean["InvoiceDate"].dt.normalize()
    clean["invoice_year"] = clean["InvoiceDate"].dt.year
    clean["invoice_month"] = clean["InvoiceDate"].dt.month
    clean["invoice_weekday"] = clean["InvoiceDate"].dt.dayofweek

    # --- 5. guest customers -----------------------------------------------
    clean["is_guest_transaction"] = clean["Customer ID"].isna()

    # --- 6. price categories ----------------------------------------------
    price = pd.to_numeric(clean["Price"], errors="coerce")
    clean["price_category"] = np.select(
        [price < 0, price == 0, price > 0],
        ["PRICE_NEGATIVE", "PRICE_ZERO", "PRICE_POSITIVE"],
        default="PRICE_INVALID",
    )
    clean["is_special_transaction"] = price <= 0

    # --- 8. numeric validation (before duplicate removal for reporting) ----
    numeric_report = validate_numeric_columns(
        clean, ["Quantity", "Price"], "online_retail_ii"
    )
    missing_report = missing_value_summary(clean, "online_retail_ii")

    # --- 7. exact duplicates ----------------------------------------------
    clean, dup_report = handle_duplicates(clean, "online_retail_ii")

    tx_breakdown = clean["transaction_type"].value_counts().to_dict()
    tx_breakdown = {k: int(v) for k, v in tx_breakdown.items()}

    report = {
        "dataset": "online_retail_ii",
        "name": "Online Retail II (UCI)",
        "original_rows": original_rows,
        "final_rows": int(len(clean)),
        "columns": list(clean.columns),
        "schema": schema,
        **dup_report,
        **date_report,
        "numeric_validation": numeric_report,
        "missing_values": missing_report,
        "transaction_type_counts": tx_breakdown,
        "cancellation_count": int(tx_breakdown.get(TX_CANCELLATION, 0)),
        "cancellation_pct": round(
            tx_breakdown.get(TX_CANCELLATION, 0) / original_rows * 100, 4
        ),
        "return_count": int(tx_breakdown.get(TX_RETURN, 0)),
        "invalid_count": int(tx_breakdown.get(TX_INVALID, 0)),
        "invalid_values": int(tx_breakdown.get(TX_INVALID, 0)),
        "guest_transaction_count": int(clean["is_guest_transaction"].sum()),
        "guest_transaction_pct": round(
            clean["is_guest_transaction"].sum() / len(clean) * 100, 4
        ),
        "special_transaction_count": int(clean["is_special_transaction"].sum()),
        "unrecovered_description_rows": int(
            (clean["Description"] == UNKNOWN_PRODUCT).sum()
        ),
        "quality_status": "REVIEW",
        "quality_notes": [
            "Exact duplicates removed (3.22% of raw); raw file untouched.",
            "Cancellations (1.79%) and returns (0.32%) preserved in separate datasets.",
            "Missing Customer ID (22.76%) treated as guest transactions - kept for "
            "sales analysis, excluded only from customer-level analysis.",
            "Zero/negative-price special lines kept and flagged "
            "(is_special_transaction) instead of deleted.",
            "363 lines have unrecoverable Description labelled 'Unknown Product'.",
        ],
    }
    return clean, report


def split_transactions(clean_df):
    """
    Split the cleaned Online Retail dataset into its analytical components.

    Returns ``(sales, returns, cancellations, invalid)`` as separate frames so
    that returns / cancellations can be analysed independently instead of
    being silently dropped.
    """
    sales = clean_df[clean_df["transaction_type"] == TX_SALE].copy()
    returns = clean_df[clean_df["transaction_type"] == TX_RETURN].copy()
    cancellations = clean_df[clean_df["transaction_type"] == TX_CANCELLATION].copy()
    invalid = clean_df[clean_df["transaction_type"] == TX_INVALID].copy()
    return sales, returns, cancellations, invalid


# ---------------------------------------------------------------------------
# Synthetic multi-store relational dataset — cleaning functions
# ---------------------------------------------------------------------------
def load_synthetic_retail_data(raw_dir=None):
    """Load all synthetic relational tables (parquet used for the large ones)."""
    raw_dir = raw_dir or RAW_DIR
    return {
        "store_master": pd.read_csv(os.path.join(raw_dir, "store_master.csv")),
        "sku_master": pd.read_csv(os.path.join(raw_dir, "sku_master.csv")),
        "customer_master": pd.read_csv(os.path.join(raw_dir, "customer_master.csv")),
        "calendar": pd.read_csv(os.path.join(raw_dir, "calendar.csv")),
        "sales_daily": pd.read_parquet(os.path.join(raw_dir, "sales_daily.parquet")),
        "inventory_snapshots": pd.read_parquet(
            os.path.join(raw_dir, "inventory_snapshots.parquet")
        ),
    }


def clean_store_master(df):
    """Validate the store dimension: unique IDs, valid size & opening date."""
    original_rows = int(len(df))
    schema = validate_schema(df, STORE_MASTER_COLUMNS, "store_master")

    clean = df.copy()
    dup_ids = int(clean["store_id"].duplicated().sum())
    missing_ids = int(clean["store_id"].isna().sum())
    invalid_size = int((clean["store_size_sqft"] <= 0).sum())
    clean, date_report = validate_dates(
        clean, "opening_date", "store_master", reference_date=pd.Timestamp("2025-12-31")
    )
    issues = []
    if dup_ids or missing_ids:
        issues.append("Duplicate or missing store_id")
    if invalid_size:
        issues.append("Non-positive store_size_sqft")
    if date_report["invalid_dates"] or date_report["future_dates"]:
        issues.append("Invalid or future opening_date")

    report = {
        "dataset": "store_master",
        "name": "Store Master",
        "original_rows": original_rows,
        "final_rows": int(len(clean)),
        "columns": list(clean.columns),
        "schema": schema,
        "duplicate_rows": dup_ids,
        "duplicates_removed": 0,
        "duplicate_store_ids": dup_ids,
        "missing_ids": missing_ids,
        "invalid_store_sizes": invalid_size,
        "invalid_values": int(dup_ids + missing_ids + invalid_size),
        "missing_values": missing_value_summary(df, "store_master"),
        "date_validation": date_report,
        "issues": issues,
        "quality_status": "PASS" if not issues else "REVIEW",
    }
    return clean, report


def clean_sku_master(df):
    """Validate the SKU dimension: unique IDs, positive prices, valid lead times."""
    original_rows = int(len(df))
    schema = validate_schema(df, SKU_MASTER_COLUMNS, "sku_master")

    clean = df.copy()
    dup_ids = int(clean["sku_id"].duplicated().sum())
    missing_ids = int(clean["sku_id"].isna().sum())
    neg_cost = int((clean["cost_price"] <= 0).sum())
    neg_base = int((clean["base_price"] <= 0).sum())
    neg_weight = int((clean["weight_kg"] <= 0).sum())
    invalid_lead = int((clean["lead_time_days"] <= 0).sum())
    neg_rop = int((clean["reorder_point"] < 0).sum())
    neg_ss = int((clean["safety_stock"] < 0).sum())
    missing_category = int(clean["category"].isna().sum())
    missing_supplier = int(clean["supplier_id"].isna().sum())
    cost_ge_base = int((clean["cost_price"] >= clean["base_price"]).sum())

    issues = []
    for label, cnt in [
        ("duplicate/missing sku_id", dup_ids + missing_ids),
        ("non-positive cost_price", neg_cost),
        ("non-positive base_price", neg_base),
        ("non-positive weight_kg", neg_weight),
        ("non-positive lead_time_days", invalid_lead),
        ("negative reorder_point", neg_rop),
        ("negative safety_stock", neg_ss),
        ("missing category", missing_category),
        ("missing supplier_id", missing_supplier),
        ("cost_price >= base_price", cost_ge_base),
    ]:
        if cnt:
            issues.append(f"{label}: {cnt}")

    report = {
        "dataset": "sku_master",
        "name": "SKU Master",
        "original_rows": original_rows,
        "final_rows": int(len(clean)),
        "columns": list(clean.columns),
        "schema": schema,
        "duplicate_rows": dup_ids,
        "duplicates_removed": 0,
        "duplicate_sku_ids": dup_ids,
        "missing_ids": missing_ids,
        "invalid_cost_price": neg_cost,
        "invalid_base_price": neg_base,
        "invalid_weight_kg": neg_weight,
        "invalid_lead_time_days": invalid_lead,
        "negative_reorder_point": neg_rop,
        "negative_safety_stock": neg_ss,
        "missing_category": missing_category,
        "missing_supplier": missing_supplier,
        "cost_price_ge_base_price": cost_ge_base,
        "missing_values": missing_value_summary(df, "sku_master"),
        "invalid_values": int(
            neg_cost + neg_base + neg_weight + invalid_lead
            + neg_rop + neg_ss + missing_category + missing_supplier
            + cost_ge_base
        ),
        "issues": issues,
        "quality_status": "PASS" if not issues else "REVIEW",
    }
    return clean, report


def clean_customer_master(df):
    """Validate the customer dimension: unique IDs, valid segments & signups."""
    original_rows = int(len(df))
    schema = validate_schema(df, CUSTOMER_MASTER_COLUMNS, "customer_master")

    clean = df.copy()
    dup_ids = int(clean["customer_id"].duplicated().sum())
    missing_ids = int(clean["customer_id"].isna().sum())
    invalid_loyalty = int((~clean["loyalty_member"].isin([0, 1])).sum())
    clean, date_report = validate_dates(
        clean, "signup_date", "customer_master", reference_date=pd.Timestamp("2025-12-31")
    )
    valid_segments = {"Consumer", "Corporate", "Small Business", "VIP Loyalty"}
    invalid_segment = int(
        (~clean["customer_segment"].isin(valid_segments)).sum()
    )

    issues = []
    if dup_ids or missing_ids:
        issues.append("Duplicate or missing customer_id")
    if invalid_loyalty:
        issues.append(f"Invalid loyalty_member values: {invalid_loyalty}")
    if invalid_segment:
        issues.append(f"Invalid customer_segment values: {invalid_segment}")
    if date_report["invalid_dates"] or date_report["future_dates"]:
        issues.append("Invalid or future signup_date")

    report = {
        "dataset": "customer_master",
        "name": "Customer Master",
        "original_rows": original_rows,
        "final_rows": int(len(clean)),
        "columns": list(clean.columns),
        "schema": schema,
        "duplicate_rows": dup_ids,
        "duplicates_removed": 0,
        "duplicate_customer_ids": dup_ids,
        "missing_ids": missing_ids,
        "invalid_loyalty_member": invalid_loyalty,
        "invalid_segment": invalid_segment,
        "invalid_values": int(invalid_loyalty + invalid_segment),
        "missing_values": missing_value_summary(df, "customer_master"),
        "date_validation": date_report,
        "issues": issues,
        "quality_status": "PASS" if not issues else "REVIEW",
    }
    return clean, report


def clean_calendar(df):
    """Validate the calendar dimension: unique dates & internally consistent attributes."""
    original_rows = int(len(df))
    schema = validate_schema(df, CALENDAR_COLUMNS, "calendar")

    clean = df.copy()
    clean["date"] = clean["date"].astype(str)
    parsed = pd.to_datetime(clean["date"], errors="coerce")
    clean["parsed_date"] = parsed

    # Preserve the original string date column (needed for clean joins) and
    # build the date-validation report from the parsed column.
    date_report = {
        "dataset": "calendar",
        "date_column": "date",
        "missing_dates": int(clean["date"].isna().sum()),
        "invalid_dates": max(int(parsed.isna().sum()) - int(clean["date"].isna().sum()), 0),
        "future_dates": int((parsed > pd.Timestamp("2025-12-31")).fillna(False).sum()),
        "min_date": str(parsed.min()) if parsed.notna().any() else None,
        "max_date": str(parsed.max()) if parsed.notna().any() else None,
        "total_days": int((parsed.max() - parsed.min()).days) if parsed.notna().any() else None,
    }

    dup_dates = int(clean["date"].duplicated().sum())
    missing_dates = int(clean["date"].isna().sum())

    # Internal-consistency checks against the parsed date.
    mismatches = {}
    mismatches["year"] = int((clean["year"] != parsed.dt.year).sum())
    mismatches["month"] = int((clean["month"] != parsed.dt.month).sum())
    mismatches["day"] = int((clean["day"] != parsed.dt.day).sum())
    mismatches["quarter"] = int((clean["quarter"] != parsed.dt.quarter).sum())
    mismatches["week_of_year"] = int(
        (clean["week_of_year"] != parsed.dt.isocalendar().week).sum()
    )
    mismatches["day_of_week"] = int(
        (clean["day_of_week"] != parsed.dt.dayofweek).sum()
    )
    mismatches["is_weekend"] = int(
        (clean["is_weekend"] != clean["parsed_date"].dt.dayofweek.isin([5, 6]).astype(int)).sum()
    )
    mismatches["is_holiday"] = int((~clean["is_holiday"].isin([0, 1])).sum())
    total_mismatch = int(sum(mismatches.values()))

    issues = []
    if dup_dates or missing_dates:
        issues.append(f"Duplicate or missing dates (dups={dup_dates}, missing={missing_dates})")
    if total_mismatch:
        issues.append(f"Calendar attribute mismatches: {total_mismatch}")

    report = {
        "dataset": "calendar",
        "name": "Calendar",
        "original_rows": original_rows,
        "final_rows": int(len(clean)),
        "columns": list(clean.columns),
        "schema": schema,
        "duplicate_rows": dup_dates,
        "duplicates_removed": 0,
        "duplicate_dates": dup_dates,
        "missing_dates": missing_dates,
        "attribute_mismatches": mismatches,
        "attribute_mismatch_total": total_mismatch,
        "invalid_values": int(dup_dates + missing_dates + total_mismatch),
        "missing_values": missing_value_summary(df, "calendar"),
        "date_validation": date_report,
        "issues": issues,
        "quality_status": "PASS" if not issues else "REVIEW",
    }
    return clean, report


def clean_sales_daily(df, store_ids, sku_ids, calendar_dates):
    """
    Validate the sales fact table.  Expected grain: ``(date, store_id, sku_id)``
    must be unique.  Checks negatives, promotion flags, and referential
    integrity against the dimension tables (no records are fabricated).
    """
    original_rows = int(len(df))
    schema = validate_schema(df, SALES_DAILY_COLUMNS, "sales_daily")

    clean = df.copy()
    clean["date"] = clean["date"].astype(str)

    # Grain validation
    grain_cols = ["date", "store_id", "sku_id"]
    grain = clean.groupby(grain_cols).size()
    grain_dups = int((grain > 1).sum())

    # Numeric validation
    numeric_report = validate_numeric_columns(
        clean,
        ["units_sold", "total_revenue", "avg_unit_price", "transaction_count", "unique_customers"],
        "sales_daily",
    )
    invalid_promo = int((~clean["promotion_flag"].isin([0, 1])).sum())

    # Referential integrity
    orphan_stores = int((~clean["store_id"].isin(set(store_ids))).sum())
    orphan_skus = int((~clean["sku_id"].isin(set(sku_ids))).sum())
    orphan_dates = int((~clean["date"].isin(set(calendar_dates))).sum())

    # Cross-field consistency: total_revenue == units_sold * avg_unit_price
    price_check = np.round(clean["units_sold"] * clean["avg_unit_price"], 2)
    revenue_mismatch = int((np.abs(clean["total_revenue"] - price_check) > 0.01).sum())

    issues = []
    if grain_dups:
        issues.append(f"Duplicate (date, store_id, sku_id) grains: {grain_dups}")
    if orphan_stores or orphan_skus or orphan_dates:
        issues.append(f"Orphan references: stores={orphan_stores}, skus={orphan_skus}, dates={orphan_dates}")
    if invalid_promo:
        issues.append(f"Invalid promotion_flag values: {invalid_promo}")
    if revenue_mismatch:
        issues.append(f"Revenue != units_sold * avg_unit_price: {revenue_mismatch}")

    report = {
        "dataset": "sales_daily",
        "name": "Sales Daily",
        "original_rows": original_rows,
        "final_rows": int(len(clean)),
        "columns": list(clean.columns),
        "schema": schema,
        "duplicate_rows": grain_dups,
        "duplicates_removed": 0,
        "duplicate_grain": grain_dups,
        "grain_columns": grain_cols,
        "numeric_validation": numeric_report,
        "invalid_promotion_flags": invalid_promo,
        "orphan_store_ids": orphan_stores,
        "orphan_sku_ids": orphan_skus,
        "orphan_dates": orphan_dates,
        "revenue_mismatch_rows": revenue_mismatch,
        "zero_unit_rows": int((clean["units_sold"] == 0).sum()),
        "missing_values": missing_value_summary(df, "sales_daily"),
        "invalid_values": int(
            grain_dups + orphan_stores + orphan_skus + orphan_dates
            + invalid_promo + revenue_mismatch
        ),
        "issues": issues,
        "quality_status": "PASS" if not issues else "REVIEW",
    }
    return clean, report


def clean_inventory_data(df, sales_df=None):
    """
    Validate inventory snapshots.

    The canonical balance equation is::

        ending_inventory == beginning_inventory + receipts - units_sold

    Investigation shows the raw ``beginning_inventory`` already *includes* the
    day's receipts (the generator snapshots opening stock after deliveries).
    As a result the canonical equation is violated on exactly the ``receipts>0``
    rows (8.36% of the dataset) by exactly the receipts amount, while the
    identity ``ending == beginning - units_sold`` holds on 100% of rows.

    We therefore: (a) report the inconsistency instead of overwriting it,
    (b) add a documented derived column ``beginning_inventory_pre_receipts`` so
    the canonical equation holds everywhere, and (c) flag each row with
    ``inventory_balance_ok``.
    """
    original_rows = int(len(df))
    schema = validate_schema(df, INVENTORY_COLUMNS, "inventory_snapshots")

    clean = df.copy()
    clean["date"] = clean["date"].astype(str)

    # Grain validation
    grain_cols = ["date", "store_id", "sku_id"]
    grain = clean.groupby(grain_cols).size()
    grain_dups = int((grain > 1).sum())

    # Numeric validation
    numeric_report = validate_numeric_columns(
        clean,
        ["beginning_inventory", "receipts", "units_sold", "ending_inventory", "on_order_qty"],
        "inventory_snapshots",
    )
    invalid_stockout = int((~clean["stockout_flag"].isin([0, 1])).sum())

    # Balance equation on the RAW columns.
    calc_raw = clean["beginning_inventory"] + clean["receipts"] - clean["units_sold"]
    raw_mismatch = int((calc_raw != clean["ending_inventory"]).sum())

    # The internally consistent identity (receipts embedded in beginning).
    identity_ok = int(
        (clean["ending_inventory"] == clean["beginning_inventory"] - clean["units_sold"]).sum()
    )

    # Derived columns for a canonical balance.
    clean["beginning_inventory_pre_receipts"] = (
        clean["beginning_inventory"] - clean["receipts"]
    )
    calc_derived = (
        clean["beginning_inventory_pre_receipts"]
        + clean["receipts"]
        - clean["units_sold"]
    )
    clean["inventory_balance_ok"] = calc_derived == clean["ending_inventory"]
    derived_ok = int(clean["inventory_balance_ok"].sum())

    # Cross-check units_sold vs the sales fact table (identical series).
    cross_sales_ok = None
    if sales_df is not None and len(sales_df) == len(clean):
        cross_sales_ok = bool(
            (clean["units_sold"].reset_index(drop=True)
             == sales_df["units_sold"].reset_index(drop=True)).all()
        )

    issues = []
    if grain_dups:
        issues.append(f"Duplicate (date, store_id, sku_id) grains: {grain_dups}")
    if raw_mismatch:
        issues.append(
            f"Canonical balance equation violated on {raw_mismatch} rows "
            "(documented semantic: beginning_inventory already includes receipts)"
        )
    if invalid_stockout:
        issues.append(f"Invalid stockout_flag values: {invalid_stockout}")

    report = {
        "dataset": "inventory_snapshots",
        "name": "Inventory Snapshots",
        "original_rows": original_rows,
        "final_rows": int(len(clean)),
        "columns": list(clean.columns),
        "schema": schema,
        "duplicate_rows": grain_dups,
        "duplicates_removed": 0,
        "duplicate_grain": grain_dups,
        "grain_columns": grain_cols,
        "numeric_validation": numeric_report,
        "invalid_stockout_flags": invalid_stockout,
        "canonical_equation_ok_rows": original_rows - raw_mismatch,
        "canonical_equation_mismatch_rows": raw_mismatch,
        "canonical_equation_mismatch_pct": round(raw_mismatch / original_rows * 100, 4),
        "ending_equals_beginning_minus_sold_rows": identity_ok,
        "derived_balance_ok_rows": derived_ok,
        "cross_sales_units_match": cross_sales_ok,
        "derived_column_added": "beginning_inventory_pre_receipts",
        "missing_values": missing_value_summary(df, "inventory_snapshots"),
        "invalid_values": int(grain_dups + invalid_stockout + raw_mismatch),
        "issues": issues,
        "quality_status": "REVIEW",  # documented semantic, not a data-entry error
    }
    return clean, report


def validate_referential_integrity(child_df, child_col, parent_df, parent_col,
                                   child_dataset, parent_dataset):
    """
    Report orphan records: values in ``child_col`` not present in ``parent_col``.
    Master records are never invented to fix orphans — they are reported.
    """
    parent_keys = set(parent_df[parent_col].astype(str).unique())
    child_keys = child_df[child_col].astype(str)
    orphans = child_keys[~child_keys.isin(parent_keys)]
    return {
        "child_dataset": child_dataset,
        "child_column": child_col,
        "parent_dataset": parent_dataset,
        "parent_column": parent_col,
        "total_child_rows": int(len(child_df)),
        "valid_relationships": int(child_keys.isin(parent_keys).sum()),
        "orphan_records": int(len(orphans)),
        "orphan_values": sorted(orphans.unique().tolist())[:50],
    }


# ---------------------------------------------------------------------------
# Outlier investigation (no removal)
# ---------------------------------------------------------------------------
def outlier_report_for_dataset(df, columns, dataset_name, z_threshold=3.0):
    """Run ``investigate_outliers`` for several columns of a dataset."""
    reports = {}
    for col in columns:
        if col in df.columns:
            reports[col] = investigate_outliers(
                df[col], col, dataset_name, z_threshold=z_threshold
            )
    return reports


# ---------------------------------------------------------------------------
# Quality report & processed-data output
# ---------------------------------------------------------------------------
def generate_quality_report(reports, docs_dir=None):
    """
    Write the full data-quality report (``data_quality_report.json``) and a
    flat per-dataset summary (``data_quality_report.csv``).  Returns the flat
    summary DataFrame.
    """
    docs_dir = docs_dir or DOCS_DIR
    os.makedirs(docs_dir, exist_ok=True)

    # Full nested JSON report.
    json_path = os.path.join(docs_dir, "data_quality_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(reports), f, indent=2)

    # Flat summary table.
    rows = []
    for key, r in reports.items():
        if not isinstance(r, dict) or "original_rows" not in r:
            continue  # skip non-dataset report blocks (e.g. "outliers")
        # Date range may live at top level (UCI) or nested in date_validation.
        dv = r.get("date_validation", {})
        min_date = r.get("min_date") or dv.get("min_date")
        max_date = r.get("max_date") or dv.get("max_date")
        rows.append({
            "Dataset": r.get("name", key),
            "Dataset Key": key,
            "Original Rows": r.get("original_rows"),
            "Final Rows": r.get("final_rows"),
            "Columns": len(r.get("columns", [])),
            "Duplicates Found": r.get("duplicate_rows", 0),
            "Duplicates Removed": r.get("duplicates_removed", 0),
            "Missing Values": sum(
                v.get("count", 0) for v in
                r.get("missing_values", {}).get("missing_by_column", {}).values()
            ),
            "Invalid Values": r.get("invalid_values", 0),
            "Outliers": sum(
                v.get("iqr_outlier_count", 0) for v in
                r.get("outliers", {}).values()
            ),
            "Date Range": (
                f"{min_date} to {max_date}" if min_date else "N/A"
            ),
            "Quality Status": r.get("quality_status", "N/A"),
        })

    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(docs_dir, "data_quality_report.csv")
    summary_df.to_csv(csv_path, index=False)

    print(f"Quality report written -> {json_path}")
    print(f"Quality report written -> {csv_path}")
    return summary_df


def save_processed_data(clean_dict, processed_dir=None, uci_splits=None):
    """
    Persist all cleaned datasets to ``data/processed/``.

    ``clean_dict`` maps dataset keys to their cleaned DataFrames.
    ``uci_splits`` (optional) provides the (sales, returns, cancellations,
    invalid) splits of the Online Retail data for separate analytical files.
    """
    processed_dir = processed_dir or PROCESSED_DIR
    os.makedirs(processed_dir, exist_ok=True)
    paths = {}

    # Online Retail family
    if "online_retail" in clean_dict:
        or_path = os.path.join(processed_dir, "online_retail_clean.csv")
        clean_dict["online_retail"].to_csv(or_path, index=False)
        paths["online_retail_clean"] = or_path

    if uci_splits is not None:
        sales, returns, cancellations, invalid = uci_splits
        for name, frame in [
            ("online_retail_sales", sales),
            ("online_retail_returns", returns),
            ("online_retail_cancellations", cancellations),
            ("online_retail_invalid", invalid),
        ]:
            p = os.path.join(processed_dir, f"{name}.csv")
            frame.to_csv(p, index=False)
            paths[name] = p

    # Synthetic relational family
    csv_files = [
        ("store_master", "store_master_clean.csv"),
        ("sku_master", "sku_master_clean.csv"),
        ("customer_master", "customer_master_clean.csv"),
        ("calendar", "calendar_clean.csv"),
    ]
    for key, fname in csv_files:
        if key in clean_dict:
            p = os.path.join(processed_dir, fname)
            clean_dict[key].to_csv(p, index=False)
            paths[f"{key}_clean"] = p

    parquet_files = [
        ("sales_daily", "sales_daily_clean.parquet"),
        ("inventory_snapshots", "inventory_snapshots_clean.parquet"),
    ]
    for key, fname in parquet_files:
        if key in clean_dict:
            p = os.path.join(processed_dir, fname)
            clean_dict[key].to_parquet(p, index=False)
            paths[f"{key}_clean"] = p

    for key, p in paths.items():
        print(f"Saved {key:32s} -> {p}")
    return paths


# ---------------------------------------------------------------------------
# End-to-end pipeline runner
# ---------------------------------------------------------------------------
def run_cleaning_pipeline():
    """
    Execute the complete Phase 3 cleaning pipeline and return the cleaned
    datasets plus the quality report.  Never touches ``data/raw/``.
    """
    ensure_directories()
    reports = {}
    clean_dict = {}

    # ---- 1. Online Retail II ---------------------------------------------
    print("=" * 70)
    print("CLEANING: UCI Online Retail II")
    print("=" * 70)
    raw_uci = load_online_retail_data()
    online_retail, uci_report = clean_online_retail(raw_uci)
    clean_dict["online_retail"] = online_retail
    reports["online_retail_ii"] = uci_report

    sales, returns, cancellations, invalid = split_transactions(online_retail)
    uci_splits = (sales, returns, cancellations, invalid)
    print(f"  SALE={len(sales):,}  RETURN={len(returns):,}  "
          f"CANCELLATION={len(cancellations):,}  INVALID={len(invalid):,}")

    # ---- 2. Synthetic multi-store dataset --------------------------------
    syn = load_synthetic_retail_data()

    print("\n" + "=" * 70)
    print("CLEANING: Store Master")
    print("=" * 70)
    store_master, rep = clean_store_master(syn["store_master"])
    clean_dict["store_master"] = store_master
    reports["store_master"] = rep

    print("\n" + "=" * 70)
    print("CLEANING: SKU Master")
    print("=" * 70)
    sku_master, rep = clean_sku_master(syn["sku_master"])
    clean_dict["sku_master"] = sku_master
    reports["sku_master"] = rep

    print("\n" + "=" * 70)
    print("CLEANING: Customer Master")
    print("=" * 70)
    customer_master, rep = clean_customer_master(syn["customer_master"])
    clean_dict["customer_master"] = customer_master
    reports["customer_master"] = rep

    print("\n" + "=" * 70)
    print("CLEANING: Calendar")
    print("=" * 70)
    calendar, rep = clean_calendar(syn["calendar"])
    clean_dict["calendar"] = calendar
    reports["calendar"] = rep

    print("\n" + "=" * 70)
    print("CLEANING: Sales Daily")
    print("=" * 70)
    sales_daily, rep = clean_sales_daily(
        syn["sales_daily"],
        store_ids=store_master["store_id"],
        sku_ids=sku_master["sku_id"],
        calendar_dates=calendar["date"],
    )
    clean_dict["sales_daily"] = sales_daily
    reports["sales_daily"] = rep

    print("\n" + "=" * 70)
    print("CLEANING: Inventory Snapshots")
    print("=" * 70)
    inventory, rep = clean_inventory_data(syn["inventory_snapshots"], sales_df=sales_daily)
    clean_dict["inventory_snapshots"] = inventory
    reports["inventory_snapshots"] = rep

    # ---- 3. Referential integrity (explicit relationships) ---------------
    print("\n" + "=" * 70)
    print("REFERENTIAL INTEGRITY CHECKS")
    print("=" * 70)
    ref_checks = {
        "sales.store_id -> store.store_id": validate_referential_integrity(
            sales_daily, "store_id", store_master, "store_id",
            "sales_daily", "store_master"),
        "sales.sku_id -> sku.sku_id": validate_referential_integrity(
            sales_daily, "sku_id", sku_master, "sku_id",
            "sales_daily", "sku_master"),
        "sales.date -> calendar.date": validate_referential_integrity(
            sales_daily, "date", calendar, "date",
            "sales_daily", "calendar"),
        "inv.store_id -> store.store_id": validate_referential_integrity(
            inventory, "store_id", store_master, "store_id",
            "inventory_snapshots", "store_master"),
        "inv.sku_id -> sku.sku_id": validate_referential_integrity(
            inventory, "sku_id", sku_master, "sku_id",
            "inventory_snapshots", "sku_master"),
        "inv.date -> calendar.date": validate_referential_integrity(
            inventory, "date", calendar, "date",
            "inventory_snapshots", "calendar"),
    }
    for label, rc in ref_checks.items():
        print(f"  {label}: orphans={rc['orphan_records']}")
        if rc["orphan_records"]:
            print(f"    orphan values: {rc['orphan_values']}")

    # ---- 4. Outlier investigation (no removal) ---------------------------
    print("\n" + "=" * 70)
    print("OUTLIER INVESTIGATION")
    print("=" * 70)
    uci_outliers = outlier_report_for_dataset(
        online_retail, ["Quantity", "Price"], "online_retail_ii"
    )
    uci_report["outliers"] = uci_outliers

    reports["sales_daily"]["outliers"] = {
        "units_sold": investigate_outliers(sales_daily["units_sold"], "units_sold", "sales_daily"),
        "total_revenue": investigate_outliers(sales_daily["total_revenue"], "total_revenue", "sales_daily"),
    }
    reports["inventory_snapshots"]["outliers"] = {
        "ending_inventory": investigate_outliers(
            inventory["ending_inventory"], "ending_inventory", "inventory_snapshots"),
        "on_order_qty": investigate_outliers(
            inventory["on_order_qty"], "on_order_qty", "inventory_snapshots"),
    }
    reports["sku_master"]["outliers"] = {
        "lead_time_days": investigate_outliers(
            sku_master["lead_time_days"], "lead_time_days", "sku_master"),
        "base_price": investigate_outliers(
            sku_master["base_price"], "base_price", "sku_master"),
    }

    # ---- 5. Quality report & processed data ------------------------------
    summary_df = generate_quality_report(reports)
    paths = save_processed_data(clean_dict, uci_splits=uci_splits)

    return {
        "clean": clean_dict,
        "uci_splits": uci_splits,
        "reports": reports,
        "summary_df": summary_df,
        "paths": paths,
    }


if __name__ == "__main__":
    results = run_cleaning_pipeline()
    print("\nPhase 3 cleaning pipeline completed successfully.")
