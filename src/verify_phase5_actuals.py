"""
Phase 5 Comprehensive Ground Truth & Actual Statistics Verifier
Reads directly from data/processed/integrated/ and data/processed/eda/
to calculate exact, 100% verified numbers for the final report.
"""
import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INT_DIR = os.path.join(BASE_DIR, "data", "processed", "integrated")
EDA_DIR = os.path.join(BASE_DIR, "data", "processed", "eda")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures", "eda")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

def verify_all():
    print("=" * 80)
    print("PHASE 5 ACTUAL GROUND TRUTH VERIFICATION")
    print("=" * 80)
    
    # 1. CAM Tables
    tables = {
        "dim_calendar": pd.read_parquet(os.path.join(INT_DIR, "dim_calendar.parquet")),
        "dim_product": pd.read_parquet(os.path.join(INT_DIR, "dim_product.parquet")),
        "dim_entity": pd.read_parquet(os.path.join(INT_DIR, "dim_entity.parquet")),
        "dim_customer": pd.read_parquet(os.path.join(INT_DIR, "dim_customer.parquet")),
        "fact_sales": pd.read_parquet(os.path.join(INT_DIR, "fact_sales.parquet")),
        "fact_inventory": pd.read_parquet(os.path.join(INT_DIR, "fact_inventory.parquet")),
        "fact_returns": pd.read_parquet(os.path.join(INT_DIR, "fact_returns.parquet")),
        "fact_cancellations": pd.read_parquet(os.path.join(INT_DIR, "fact_cancellations.parquet")),
        "customer_analytics": pd.read_parquet(os.path.join(INT_DIR, "customer_analytics.parquet")),
        "inventory_analytics": pd.read_parquet(os.path.join(INT_DIR, "inventory_analytics.parquet")),
        "forecast_base": pd.read_parquet(os.path.join(INT_DIR, "forecast_base.parquet")),
    }
    
    print("\n--- 1. CAM TABLE ACTUALS ---")
    for name, df in tables.items():
        srcs = sorted(df["source_dataset"].unique().tolist()) if "source_dataset" in df.columns else ["N/A"]
        date_info = ""
        if "date" in df.columns:
            dmin = df["date"].min()
            dmax = df["date"].max()
            date_info = f" | Dates: {dmin} to {dmax}"
        print(f"{name:<22}: Rows={len(df):>10,}, Cols={len(df.columns):>2}, Sources={srcs}{date_info}")

    # Inspect dim_calendar breakdown
    cal = tables["dim_calendar"]
    cal["date"] = pd.to_datetime(cal["date"])
    print(f"\ndim_calendar breakdown:")
    print(f"Total rows: {len(cal):,}")
    print(f"Min date: {cal['date'].min().date()}, Max date: {cal['date'].max().date()}")
    print(f"date_source distribution:\n{cal['date_source'].value_counts().to_string()}")
    
    # Calendar distinct years
    print(f"Calendar years present: {sorted(cal['date'].dt.year.unique().tolist())}")
    
    # 2. Fact Sales Breakdown
    sales = tables["fact_sales"]
    sales["date"] = pd.to_datetime(sales["date"])
    
    uci_sales = sales[sales["source_dataset"] == "UCI"]
    syn_sales = sales[sales["source_dataset"] == "SYNTHETIC"]
    
    print("\n--- 2. ACTUAL SALES STATISTICS ---")
    print(f"UCI Sales:")
    print(f"  Rows: {len(uci_sales):,}")
    print(f"  Date Range: {uci_sales['date'].min().date()} to {uci_sales['date'].max().date()} ({uci_sales['date'].nunique()} active days)")
    print(f"  Total Revenue: ${uci_sales['revenue'].sum():,.2f}")
    print(f"  Total Units Sold: {uci_sales['units_sold'].sum():,.0f}")
    print(f"  Total Transactions: {uci_sales['transaction_count'].sum():,.0f}")
    print(f"  Unique Products: {uci_sales['product_key'].nunique():,}")
    print(f"  Average Daily Revenue: ${uci_sales.groupby('date')['revenue'].sum().mean():,.2f}")
    print(f"  Average Daily Units: {uci_sales.groupby('date')['units_sold'].sum().mean():,.2f}")
    print(f"  Average Order Value (AOV): ${uci_sales['revenue'].sum() / uci_sales['transaction_count'].sum():,.2f}")
    print(f"  Average Realized Unit Price: ${uci_sales['revenue'].sum() / uci_sales['units_sold'].sum():,.2f}")

    print(f"\nSynthetic Sales:")
    print(f"  Rows: {len(syn_sales):,}")
    print(f"  Date Range: {syn_sales['date'].min().date()} to {syn_sales['date'].max().date()} ({syn_sales['date'].nunique()} active days)")
    print(f"  Total Revenue: ${syn_sales['revenue'].sum():,.2f}")
    print(f"  Total Units Sold: {syn_sales['units_sold'].sum():,.0f}")
    print(f"  Total Transactions: {syn_sales['transaction_count'].sum():,.0f}")
    print(f"  Unique Products: {syn_sales['product_key'].nunique():,}")
    print(f"  Unique Stores: {syn_sales['entity_id'].nunique():,}")
    print(f"  Average Daily Revenue: ${syn_sales.groupby('date')['revenue'].sum().mean():,.2f}")
    print(f"  Average Daily Units: {syn_sales.groupby('date')['units_sold'].sum().mean():,.2f}")
    print(f"  Average Order Value (AOV): ${syn_sales['revenue'].sum() / syn_sales['transaction_count'].sum():,.2f}")
    print(f"  Average Realized Unit Price: ${syn_sales['revenue'].sum() / syn_sales['units_sold'].sum():,.2f}")

    # 3. Inventory Actuals
    inv = tables["fact_inventory"]
    inv["date"] = pd.to_datetime(inv["date"])
    latest_inv_date = inv["date"].max()
    latest_inv = inv[inv["date"] == latest_inv_date]
    
    prod_syn = tables["dim_product"][tables["dim_product"]["source_dataset"] == "SYNTHETIC"].set_index("product_key")
    latest_inv = latest_inv.merge(prod_syn[["cost_price", "base_price", "safety_stock", "reorder_point"]], on="product_key", how="left")
    
    ending_inv_units = latest_inv["ending_inventory"].sum()
    inventory_val_cost = (latest_inv["ending_inventory"] * latest_inv["cost_price"]).sum()
    inventory_val_retail = (latest_inv["ending_inventory"] * latest_inv["base_price"]).sum()
    stockout_count = latest_inv["stockout_flag"].sum()
    stockout_rate = (stockout_count / len(latest_inv)) * 100
    
    # Trailing 30-day demand for DOI
    demand_30d = inv[inv["date"] >= latest_inv_date - pd.Timedelta(days=30)]["units_sold"].sum() / 30
    doi = ending_inv_units / demand_30d if demand_30d > 0 else 0
    
    print("\n--- 3. ACTUAL INVENTORY STATISTICS (Synthetic) ---")
    print(f"  Total Inventory Records: {len(inv):,}")
    print(f"  Date Range: {inv['date'].min().date()} to {inv['date'].max().date()}")
    print(f"  Ending Network Inventory Units (as of {latest_inv_date.date()}): {ending_inv_units:,}")
    print(f"  Ending Inventory Valuation (at Cost): ${inventory_val_cost:,.2f}")
    print(f"  Ending Inventory Valuation (at Retail): ${inventory_val_retail:,.2f}")
    print(f"  Current Stockout Incidents (Store-SKUs): {stockout_count:,} / {len(latest_inv):,} ({stockout_rate:.2f}%)")
    print(f"  Network Days of Inventory (DOI): {doi:.1f} days")
    print(f"  Historical Total Stockout Store-SKU-Days: {inv['stockout_flag'].sum():,} ({inv['stockout_flag'].mean()*100:.2f}%)")

    # 4. Returns & Cancellations Actuals
    ret = tables["fact_returns"]
    canc = tables["fact_cancellations"]
    print("\n--- 4. ACTUAL RETURNS & CANCELLATIONS (UCI) ---")
    print(f"  Total Return Records: {len(ret):,}")
    print(f"  Total Units Returned: {ret['quantity_returned'].sum():,}")
    print(f"  Total Return Transactions: {ret['return_transactions'].sum():,}")
    print(f"  Total Return Revenue Impact: ${ret['revenue_impact'].sum():,.2f}")
    print(f"  Total Cancellation Records: {len(canc):,}")
    print(f"  Total Units Cancelled: {canc['cancelled_quantity'].sum():,}")
    print(f"  Total Cancellation Transactions: {canc['cancellation_transactions'].sum():,}")
    print(f"  Total Cancellation Revenue Impact: ${canc['revenue_impact'].sum():,.2f}")

    # 5. Customer Analytics Actuals
    cust = tables["customer_analytics"]
    uci_cust = cust[cust["source_dataset"] == "UCI"]
    syn_cust = cust[cust["source_dataset"] == "SYNTHETIC"]
    print("\n--- 5. ACTUAL CUSTOMER ANALYTICS ---")
    print(f"  UCI Identified Customers: {len(uci_cust):,}")
    print(f"  UCI Identified Customer Revenue: ${uci_cust['total_revenue'].sum():,.2f}")
    print(f"  UCI Guest Checkout Revenue: ${uci_sales['revenue'].sum() - uci_cust['total_revenue'].sum():,.2f}")
    print(f"  UCI Identified Revenue Share: {uci_cust['total_revenue'].sum() / uci_sales['revenue'].sum() * 100:.2f}%")
    print(f"  UCI Avg Revenue per Identified Customer: ${uci_cust['total_revenue'].mean():,.2f}")
    print(f"  UCI Median Revenue per Identified Customer: ${uci_cust['total_revenue'].median():,.2f}")
    print(f"  UCI Max Revenue by Single Customer: ${uci_cust['total_revenue'].max():,.2f}")
    print(f"  Synthetic Customer Profiles: {len(syn_cust):,}")

    # 6. Figures check
    figures = [f for f in os.listdir(FIG_DIR) if f.endswith(".png")]
    print("\n--- 6. ACTUAL GENERATED FIGURES ---")
    print(f"Total Figures in {FIG_DIR}: {len(figures)}")
    for f in sorted(figures):
        fpath = os.path.join(FIG_DIR, f)
        print(f"  - {f:<38} ({os.path.getsize(fpath):>10,} bytes)")

    # 7. EDA Parquet files check
    parquets = [f for f in os.listdir(EDA_DIR) if f.endswith(".parquet")]
    print("\n--- 7. ACTUAL GENERATED EDA PARQUET ARTIFACTS ---")
    print(f"Total Parquets in {EDA_DIR}: {len(parquets)}")
    for p in sorted(parquets):
        ppath = os.path.join(EDA_DIR, p)
        print(f"  - {p:<38} ({os.path.getsize(ppath):>10,} bytes)")

if __name__ == "__main__":
    verify_all()
