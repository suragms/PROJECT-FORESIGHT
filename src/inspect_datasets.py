"""
Comprehensive Dataset Profiling and Inspection Script
Inspects:
1. Online Retail II (UCI)
2. Synthetic Retail Multi-Store Relational Dataset (Store, SKU, Customer, Calendar, Sales Daily, Inventory Snapshots)
"""

import os
import json
import pandas as pd
import numpy as np

def profile_dataframe(name, df, date_col=None, product_col=None, customer_col=None):
    profile = {
        "name": name,
        "rows": int(len(df)),
        "columns_count": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": {col: int(cnt) for col, cnt in df.isnull().sum().items()},
        "null_pct": {col: round(float(cnt / len(df) * 100), 2) for col, cnt in df.isnull().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct": round(float(df.duplicated().sum() / len(df) * 100), 2)
    }
    
    if date_col and date_col in df.columns:
        dt_series = pd.to_datetime(df[date_col], errors="coerce")
        profile["date_range"] = {
            "min": str(dt_series.min()),
            "max": str(dt_series.max()),
            "total_days": int((dt_series.max() - dt_series.min()).days) if pd.notnull(dt_series.min()) else None
        }
        
    if product_col and product_col in df.columns:
        profile["unique_products"] = int(df[product_col].nunique())
        
    if customer_col and customer_col in df.columns:
        profile["unique_customers"] = int(df[customer_col].nunique())
        
    print(f"\n{'='*50}\nDATASET: {name}\n{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Duplicates: {profile['duplicate_rows']} ({profile['duplicate_pct']}%)")
    print("Null Counts:")
    for col, cnt in profile["null_counts"].items():
        if cnt > 0:
            print(f"  - {col}: {cnt:,} ({profile['null_pct'][col]}%)")
    if "date_range" in profile:
        print(f"Date Range: {profile['date_range']['min']} to {profile['date_range']['max']}")
    if "unique_products" in profile:
        print(f"Unique Products: {profile['unique_products']:,}")
    if "unique_customers" in profile:
        print(f"Unique Customers: {profile['unique_customers']:,}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nSummary Statistics:")
    print(df.describe(include=[np.number]).T)
    return profile

def run_all_inspections():
    raw_dir = "data/raw"
    results = {}
    
    # 1. Online Retail II
    print("\n>>> PROFILING DATASET 1: Online Retail II (UCI)")
    df_uci = pd.read_csv(os.path.join(raw_dir, "online_retail_II.csv"), low_memory=False)
    # Check column names
    print("Online Retail II columns:", df_uci.columns.tolist())
    # find invoice date, product code, customer id
    inv_col = [c for c in df_uci.columns if "InvoiceDate" in c or "date" in c.lower()][0] if any("date" in c.lower() for c in df_uci.columns) else "InvoiceDate"
    stock_col = [c for c in df_uci.columns if "StockCode" in c or "stock" in c.lower()][0] if any("stock" in c.lower() for c in df_uci.columns) else "StockCode"
    cust_col = [c for c in df_uci.columns if "Customer" in c or "cust" in c.lower()][0] if any("cust" in c.lower() for c in df_uci.columns) else "Customer ID"
    
    results["online_retail_ii"] = profile_dataframe("Online Retail II (UCI)", df_uci, date_col=inv_col, product_col=stock_col, customer_col=cust_col)
    
    # 2. Store Master
    print("\n>>> PROFILING DATASET 2.1: store_master.csv")
    df_store = pd.read_csv(os.path.join(raw_dir, "store_master.csv"))
    results["store_master"] = profile_dataframe("Store Master", df_store)
    
    # 3. SKU Master
    print("\n>>> PROFILING DATASET 2.2: sku_master.csv")
    df_sku = pd.read_csv(os.path.join(raw_dir, "sku_master.csv"))
    results["sku_master"] = profile_dataframe("SKU Master", df_sku, product_col="sku_id")
    
    # 4. Customer Master
    print("\n>>> PROFILING DATASET 2.3: customer_master.csv")
    df_cust = pd.read_csv(os.path.join(raw_dir, "customer_master.csv"))
    results["customer_master"] = profile_dataframe("Customer Master", df_cust, customer_col="customer_id")
    
    # 5. Calendar
    print("\n>>> PROFILING DATASET 2.4: calendar.csv")
    df_cal = pd.read_csv(os.path.join(raw_dir, "calendar.csv"))
    results["calendar"] = profile_dataframe("Calendar", df_cal, date_col="date")
    
    # 6. Sales Daily
    print("\n>>> PROFILING DATASET 2.5: sales_daily.csv")
    df_sales = pd.read_parquet(os.path.join(raw_dir, "sales_daily.parquet"))
    results["sales_daily"] = profile_dataframe("Sales Daily", df_sales, date_col="date", product_col="sku_id")
    
    # 7. Inventory Snapshots
    print("\n>>> PROFILING DATASET 2.6: inventory_snapshots.csv")
    df_inv = pd.read_parquet(os.path.join(raw_dir, "inventory_snapshots.parquet"))
    results["inventory_snapshots"] = profile_dataframe("Inventory Snapshots", df_inv, date_col="date", product_col="sku_id")
    
    # Save inspection metadata
    with open("docs/data_profiling_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved comprehensive profiling metadata to docs/data_profiling_summary.json")

if __name__ == "__main__":
    run_all_inspections()
