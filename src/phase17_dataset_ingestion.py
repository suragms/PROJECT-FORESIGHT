"""
Phase 17 — Dataset Ingestion & Processing
==========================================
Project FORESIGHT: Demand & Inventory Intelligence

Ingests UCI and Synthetic datasets from data/raw/ into data/phase17/.
Produces weekly SKU-level demand tables for forecasting.
Never modifies data/raw/, data/processed/, or models/final/.
"""

import os
import sys
import json
import hashlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
P17_DIR = os.path.join(BASE_DIR, "data", "phase17")
P17_RAW = os.path.join(P17_DIR, "raw")
P17_PROC = os.path.join(P17_DIR, "processed")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

for d in [P17_RAW, P17_PROC]:
    os.makedirs(d, exist_ok=True)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================
# 1. UCI INGESTION & PROCESSING
# =====================================================================

def ingest_uci() -> dict:
    """Load, clean, and aggregate UCI Online Retail II to weekly SKU demand."""
    path = os.path.join(RAW_DIR, "online_retail_II.csv")
    if not os.path.exists(path):
        return {"status": "FAIL", "reason": "online_retail_II.csv not found"}

    print("--- UCI Ingestion ---")
    sha = _sha256(path)
    df = pd.read_csv(path, encoding="latin-1")
    raw_rows = len(df)
    print(f"  Raw rows: {raw_rows:,}")

    # Schema
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower().replace(" ", "_")
        col_map[c] = cl
    df.rename(columns=col_map, inplace=True)

    # Standardize column names
    rename = {}
    if "invoiceno" in df.columns:
        rename["invoiceno"] = "invoice"
    if "unitprice" in df.columns:
        rename["unitprice"] = "price"
    if "customerid" in df.columns:
        rename["customerid"] = "customer_id"
    df.rename(columns=rename, inplace=True)

    # Parse dates
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
    invalid_dates = int(df["invoicedate"].isna().sum())

    # Quality metrics
    quality = {
        "raw_rows": raw_rows,
        "columns": list(df.columns),
        "invalid_dates": invalid_dates,
        "null_invoice": int(df["invoice"].isna().sum()),
        "null_stockcode": int(df["stockcode"].isna().sum()),
        "null_description": int(df["description"].isna().sum()),
        "null_customer_id": int(df["customer_id"].isna().sum()),
        "negative_qty_rows": int((df["quantity"] < 0).sum()),
        "zero_price_rows": int((df["price"] == 0).sum()),
        "negative_price_rows": int((df["price"] < 0).sum()),
    }

    # Identify cancellations (C-prefix invoices) and returns (negative qty)
    df["invoice_str"] = df["invoice"].astype(str)
    is_cancel = df["invoice_str"].str.startswith("C")
    is_return = (~is_cancel) & (df["quantity"] < 0)
    is_sale = (~is_cancel) & (df["quantity"] > 0) & (df["price"] > 0)

    quality["cancellation_rows"] = int(is_cancel.sum())
    quality["return_rows"] = int(is_return.sum())
    quality["valid_sale_rows"] = int(is_sale.sum())

    # Keep only valid sales for demand forecasting
    sales = df[is_sale].copy()
    sales.dropna(subset=["invoicedate"], inplace=True)
    sales["revenue"] = sales["quantity"] * sales["price"]

    # Remove exact duplicates
    before_dedup = len(sales)
    sales = sales.drop_duplicates()
    quality["duplicates_removed"] = before_dedup - len(sales)
    quality["clean_sale_rows"] = len(sales)

    # Weekly SKU-level aggregation
    sales["date"] = sales["invoicedate"].dt.date
    sales["week_start"] = pd.to_datetime(sales["invoicedate"]).dt.to_period("W-MON").dt.start_time

    weekly = sales.groupby(["week_start", "stockcode"]).agg(
        units_sold=("quantity", "sum"),
        revenue=("revenue", "sum"),
        avg_unit_price=("price", "mean"),
        transaction_count=("invoice_str", "nunique"),
        unique_customers=("customer_id", "nunique"),
    ).reset_index()

    weekly.rename(columns={"week_start": "week", "stockcode": "product_key"}, inplace=True)
    weekly["source_dataset"] = "UCI"
    weekly["product_key"] = "UCI_" + weekly["product_key"].astype(str)

    # Remove weeks with impossible aggregates
    weekly = weekly[weekly["units_sold"] > 0].copy()
    weekly.sort_values(["product_key", "week"], inplace=True)

    quality["weekly_rows"] = len(weekly)
    quality["weekly_skus"] = int(weekly["product_key"].nunique())
    quality["date_min"] = str(weekly["week"].min())
    quality["date_max"] = str(weekly["week"].max())
    quality["total_weeks"] = int(weekly["week"].nunique())

    # Save
    out_path = os.path.join(P17_PROC, "uci_weekly_demand.parquet")
    weekly.to_parquet(out_path, index=False)
    print(f"  Weekly demand: {len(weekly):,} rows, {weekly['product_key'].nunique()} SKUs")
    print(f"  Date range: {weekly['week'].min()} to {weekly['week'].max()}")
    print(f"  Saved: {out_path}")

    quality["sha256_raw"] = sha
    quality["inventory_source"] = "NOT_AVAILABLE"
    quality["lead_time_source"] = "NOT_AVAILABLE"
    quality["reorder_point_source"] = "NOT_AVAILABLE"

    return {"status": "PASS", "quality": quality}


# =====================================================================
# 2. SYNTHETIC INGESTION & PROCESSING
# =====================================================================

def ingest_synthetic() -> dict:
    """Load, validate, and aggregate Synthetic dataset to weekly SKU demand."""
    sales_path = os.path.join(RAW_DIR, "sales_daily.parquet")
    inv_path = os.path.join(RAW_DIR, "inventory_snapshots.parquet")
    sku_path = os.path.join(RAW_DIR, "sku_master.csv")
    store_path = os.path.join(RAW_DIR, "store_master.csv")
    cal_path = os.path.join(RAW_DIR, "calendar.csv")

    print("\n--- Synthetic Ingestion ---")
    quality = {}

    # Sales daily
    if not os.path.exists(sales_path):
        return {"status": "FAIL", "reason": "sales_daily.parquet not found"}

    sales = pd.read_parquet(sales_path)
    sales["date"] = pd.to_datetime(sales["date"])
    quality["sales_raw_rows"] = len(sales)
    quality["sales_columns"] = list(sales.columns)
    quality["sales_stores"] = int(sales["store_id"].nunique())
    quality["sales_skus"] = int(sales["sku_id"].nunique())
    quality["sales_date_min"] = str(sales["date"].min().date())
    quality["sales_date_max"] = str(sales["date"].max().date())
    quality["sales_negative_qty"] = int((sales["units_sold"] < 0).sum())
    quality["sales_null_store"] = int(sales["store_id"].isna().sum())
    quality["sales_null_sku"] = int(sales["sku_id"].isna().sum())
    quality["sales_duplicates"] = int(sales.duplicated(subset=["date", "store_id", "sku_id"]).sum())

    print(f"  Sales: {len(sales):,} rows, {sales['store_id'].nunique()} stores, {sales['sku_id'].nunique()} SKUs")

    # Inventory
    inv = None
    if os.path.exists(inv_path):
        inv = pd.read_parquet(inv_path)
        inv["date"] = pd.to_datetime(inv["date"])
        quality["inv_rows"] = len(inv)
        quality["inv_stores"] = int(inv["store_id"].nunique())
        quality["inv_skus"] = int(inv["sku_id"].nunique())
        print(f"  Inventory: {len(inv):,} rows")
    else:
        quality["inv_rows"] = 0
        quality["inventory_source"] = "NOT_AVAILABLE"

    # SKU master
    skus = pd.read_csv(sku_path) if os.path.exists(sku_path) else None
    if skus is not None:
        quality["sku_master_rows"] = len(skus)
        quality["has_lead_time"] = "lead_time_days" in skus.columns
        quality["has_reorder_point"] = "reorder_point" in skus.columns
        quality["has_safety_stock"] = "safety_stock" in skus.columns
        quality["has_cost_price"] = "cost_price" in skus.columns
        quality["has_base_price"] = "base_price" in skus.columns

    # Weekly SKU-level aggregation (aggregated across stores for Zidio weekly SKU demand)
    sales["week_start"] = sales["date"].dt.to_period("W-MON").dt.start_time

    # SKU-level weekly (aggregated across stores)
    weekly_sku = sales.groupby(["week_start", "sku_id"]).agg(
        units_sold=("units_sold", "sum"),
        revenue=("total_revenue", "sum"),
        avg_unit_price=("avg_unit_price", "mean"),
        transaction_count=("transaction_count", "sum"),
        unique_customers=("unique_customers", "sum"),
        promotion_flag=("promotion_flag", "max"),
        store_count=("store_id", "nunique"),
    ).reset_index()

    weekly_sku.rename(columns={"week_start": "week", "sku_id": "product_key"}, inplace=True)
    weekly_sku["source_dataset"] = "SYNTHETIC"
    weekly_sku["product_key"] = "SYN_" + weekly_sku["product_key"].astype(str)
    weekly_sku.sort_values(["product_key", "week"], inplace=True)

    quality["weekly_rows"] = len(weekly_sku)
    quality["weekly_skus"] = int(weekly_sku["product_key"].nunique())
    quality["date_min"] = str(weekly_sku["week"].min())
    quality["date_max"] = str(weekly_sku["week"].max())
    quality["total_weeks"] = int(weekly_sku["week"].nunique())

    out_path = os.path.join(P17_PROC, "synthetic_weekly_demand.parquet")
    weekly_sku.to_parquet(out_path, index=False)
    print(f"  Weekly SKU demand: {len(weekly_sku):,} rows, {weekly_sku['product_key'].nunique()} SKUs")
    print(f"  Date range: {weekly_sku['week'].min()} to {weekly_sku['week'].max()}")

    # Store-level weekly (preserve for risk scoring)
    weekly_store = sales.groupby(["week_start", "store_id", "sku_id"]).agg(
        units_sold=("units_sold", "sum"),
        revenue=("total_revenue", "sum"),
        avg_unit_price=("avg_unit_price", "mean"),
        promotion_flag=("promotion_flag", "max"),
    ).reset_index()
    weekly_store.rename(columns={"week_start": "week"}, inplace=True)
    weekly_store["source_dataset"] = "SYNTHETIC"
    store_out = os.path.join(P17_PROC, "synthetic_weekly_store_sku.parquet")
    weekly_store.to_parquet(store_out, index=False)

    # Weekly inventory snapshots (latest per week per store-SKU)
    if inv is not None:
        inv["week_start"] = inv["date"].dt.to_period("W-MON").dt.start_time
        inv_weekly = inv.sort_values("date").groupby(
            ["week_start", "store_id", "sku_id"]
        ).last().reset_index()
        inv_weekly.rename(columns={"week_start": "week"}, inplace=True)
        inv_out = os.path.join(P17_PROC, "synthetic_weekly_inventory.parquet")
        inv_weekly.to_parquet(inv_out, index=False)
        quality["inv_weekly_rows"] = len(inv_weekly)
        print(f"  Weekly inventory: {len(inv_weekly):,} rows")

    # Copy SKU master for risk scoring
    if skus is not None:
        skus.to_csv(os.path.join(P17_PROC, "sku_master.csv"), index=False)

    quality["sha256_sales"] = _sha256(sales_path)
    if os.path.exists(inv_path):
        quality["sha256_inv"] = _sha256(inv_path)

    return {"status": "PASS", "quality": quality}


# =====================================================================
# 3. MAIN
# =====================================================================

def run_ingestion():
    print("=" * 60)
    print("PHASE 17 — DATASET INGESTION")
    print("=" * 60)

    uci_result = ingest_uci()
    syn_result = ingest_synthetic()

    manifest = {
        "phase": 17,
        "step": "dataset_ingestion",
        "timestamp": datetime.utcnow().isoformat() + "+00:00",
        "uci": uci_result,
        "synthetic": syn_result,
    }

    manifest_path = os.path.join(P17_DIR, "ingestion_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest: {manifest_path}")

    return manifest


if __name__ == "__main__":
    run_ingestion()
