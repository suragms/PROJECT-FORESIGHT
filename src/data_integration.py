"""
Phase 4 — Data Integration & Common Analytical Model (CAM)
============================================================
Project FORESIGHT: Demand & Inventory Intelligence

Provides unified data access, dimension merging, and business aggregation
across multi-store sales, inventory snapshots, product dimensions, calendar,
and UCI Online Retail II transactions.
"""

import os
import functools
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


@functools.lru_cache(maxsize=1)
def load_store_master(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned store master dataset."""
    path = os.path.join(processed_dir, "store_master_clean.csv")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "store_master.csv")
    df = pd.read_csv(path)
    df["store_id"] = df["store_id"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_sku_master(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned SKU master dataset."""
    path = os.path.join(processed_dir, "sku_master_clean.csv")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "sku_master.csv")
    df = pd.read_csv(path)
    df["sku_id"] = df["sku_id"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_customer_master(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned customer master dataset."""
    path = os.path.join(processed_dir, "customer_master_clean.csv")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "customer_master.csv")
    df = pd.read_csv(path)
    df["customer_id"] = df["customer_id"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_calendar(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned calendar dataset."""
    path = os.path.join(processed_dir, "calendar_clean.csv")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "calendar.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@functools.lru_cache(maxsize=1)
def load_sales_daily(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned sales daily parquet dataset."""
    path = os.path.join(processed_dir, "sales_daily_clean.parquet")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "sales_daily.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["store_id"] = df["store_id"].astype(str)
    df["sku_id"] = df["sku_id"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_inventory_snapshots(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned inventory snapshots parquet dataset."""
    path = os.path.join(processed_dir, "inventory_snapshots_clean.parquet")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "inventory_snapshots.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["store_id"] = df["store_id"].astype(str)
    df["sku_id"] = df["sku_id"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_online_retail(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned Online Retail II parquet dataset."""
    parquet_path = os.path.join(processed_dir, "online_retail_sales.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        csv_path = os.path.join(processed_dir, "online_retail_sales.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, low_memory=False)
        else:
            clean_csv = os.path.join(processed_dir, "online_retail_clean.csv")
            df = pd.read_csv(clean_csv, low_memory=False)
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def build_integrated_cam(
    sample_store_count: int = None,
    sample_sku_count: int = None,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Build the Common Analytical Model (CAM) joining Sales, Inventory,
    SKU Master, Store Master, and Calendar.
    """
    sales = load_sales_daily()
    inventory = load_inventory_snapshots()
    skus = load_sku_master()
    stores = load_store_master()
    calendar = load_calendar()

    # Optional filtering
    if start_date:
        sales = sales[sales["date"] >= pd.to_datetime(start_date)]
        inventory = inventory[inventory["date"] >= pd.to_datetime(start_date)]
    if end_date:
        sales = sales[sales["date"] <= pd.to_datetime(end_date)]
        inventory = inventory[inventory["date"] <= pd.to_datetime(end_date)]
    if sample_store_count:
        selected_stores = stores["store_id"].head(sample_store_count).tolist()
        sales = sales[sales["store_id"].isin(selected_stores)]
        inventory = inventory[inventory["store_id"].isin(selected_stores)]
    if sample_sku_count:
        selected_skus = skus["sku_id"].head(sample_sku_count).tolist()
        sales = sales[sales["sku_id"].isin(selected_skus)]
        inventory = inventory[inventory["sku_id"].isin(selected_skus)]

    # Merge sales and inventory
    merged = pd.merge(
        sales,
        inventory,
        on=["date", "store_id", "sku_id"],
        how="inner",
        suffixes=("", "_inv"),
    )

    # Merge SKU Master
    sku_cols = [
        "sku_id", "sku_name", "category", "sub_category", "brand",
        "cost_price", "base_price", "lead_time_days", "reorder_point", "safety_stock"
    ]
    merged = pd.merge(merged, skus[sku_cols], on="sku_id", how="left")

    # Merge Store Master
    store_cols = ["store_id", "store_name", "city", "state", "region", "store_type"]
    merged = pd.merge(merged, stores[store_cols], on="store_id", how="left")

    # Merge Calendar
    cal_cols = ["date", "year", "month", "quarter", "day_of_week", "day_name", "is_weekend", "is_holiday", "season"]
    merged = pd.merge(merged, calendar[cal_cols], on="date", how="left")

    # Calculate derived financial & inventory metrics
    merged["cogs"] = merged["units_sold"] * merged["cost_price"]
    merged["gross_profit"] = merged["total_revenue"] - merged["cogs"]
    merged["margin_pct"] = np.where(
        merged["total_revenue"] > 0,
        merged["gross_profit"] / merged["total_revenue"],
        0.0
    )
    merged["ending_inventory_value"] = merged["ending_inventory"] * merged["cost_price"]
    merged["is_stockout"] = (merged["stockout_flag"] == 1) | (merged["ending_inventory"] == 0)

    return merged


def get_executive_kpis(cam_df: pd.DataFrame = None) -> dict:
    """Compute high-level executive KPIs across sales, inventory, and service levels."""
    if cam_df is None:
        sales = load_sales_daily()
        inventory = load_inventory_snapshots()
        skus = load_sku_master()
        stores = load_store_master()

        total_revenue = float(sales["total_revenue"].sum())
        total_units = int(sales["units_sold"].sum())
        total_transactions = int(sales["transaction_count"].sum())

        # Latest inventory snapshot
        latest_date = inventory["date"].max()
        latest_inv = inventory[inventory["date"] == latest_date].copy()
        latest_inv = pd.merge(latest_inv, skus[["sku_id", "cost_price", "base_price", "safety_stock", "reorder_point"]], on="sku_id", how="left")

        total_inventory_units = int(latest_inv["ending_inventory"].sum())
        total_inventory_value = float((latest_inv["ending_inventory"] * latest_inv["cost_price"]).sum())
        stockout_incidents = int(latest_inv["stockout_flag"].sum())
        stockout_rate = float(latest_inv["stockout_flag"].mean() * 100)

        # Reorder breaches
        reorder_triggered = int((latest_inv["ending_inventory"] <= latest_inv["reorder_point"]).sum())
        safety_breaches = int((latest_inv["ending_inventory"] < latest_inv["safety_stock"]).sum())

        return {
            "total_revenue": total_revenue,
            "total_units_sold": total_units,
            "total_transactions": total_transactions,
            "total_inventory_units": total_inventory_units,
            "total_inventory_value": total_inventory_value,
            "current_stockout_count": stockout_incidents,
            "current_stockout_rate_pct": stockout_rate,
            "reorder_triggered_count": reorder_triggered,
            "safety_stock_breaches": safety_breaches,
            "total_stores": len(stores),
            "total_skus": len(skus),
            "active_skus": int(sales["sku_id"].nunique()),
            "latest_date": str(latest_date)[:10],
        }
    else:
        total_revenue = float(cam_df["total_revenue"].sum())
        total_units = int(cam_df["units_sold"].sum())
        total_profit = float(cam_df["gross_profit"].sum()) if "gross_profit" in cam_df.columns else 0.0
        avg_margin = float(cam_df["margin_pct"].mean() * 100) if "margin_pct" in cam_df.columns else 0.0

        latest_date = cam_df["date"].max()
        latest_subset = cam_df[cam_df["date"] == latest_date]
        total_inventory_value = float(latest_subset["ending_inventory_value"].sum())
        stockout_rate = float(latest_subset["is_stockout"].mean() * 100)

        return {
            "total_revenue": total_revenue,
            "total_units_sold": total_units,
            "total_gross_profit": total_profit,
            "avg_margin_pct": avg_margin,
            "total_inventory_value": total_inventory_value,
            "current_stockout_rate_pct": stockout_rate,
            "latest_date": str(latest_date)[:10],
        }


def get_top_bottom_skus(top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve top performing and bottom performing SKUs by revenue and volume."""
    sales = load_sales_daily()
    skus = load_sku_master()

    agg = sales.groupby("sku_id").agg(
        total_units=("units_sold", "sum"),
        total_revenue=("total_revenue", "sum"),
        avg_price=("avg_unit_price", "mean"),
        active_days=("date", "nunique")
    ).reset_index()

    merged = pd.merge(agg, skus[["sku_id", "sku_name", "category", "brand", "cost_price", "base_price"]], on="sku_id", how="left")
    merged["margin_pct"] = ((merged["base_price"] - merged["cost_price"]) / merged["base_price"]) * 100

    top_df = merged.sort_values(by="total_revenue", ascending=False).head(top_n).reset_index(drop=True)
    bottom_df = merged.sort_values(by="total_revenue", ascending=True).head(top_n).reset_index(drop=True)
    return top_df, bottom_df
