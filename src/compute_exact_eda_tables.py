"""
Extracts exact tables and numbers for all 17 EDA sections
"""
import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INT_DIR = os.path.join(BASE_DIR, "data", "processed", "integrated")

# Load tables
dim_calendar = pd.read_parquet(os.path.join(INT_DIR, "dim_calendar.parquet"))
dim_product = pd.read_parquet(os.path.join(INT_DIR, "dim_product.parquet"))
dim_entity = pd.read_parquet(os.path.join(INT_DIR, "dim_entity.parquet"))
dim_customer = pd.read_parquet(os.path.join(INT_DIR, "dim_customer.parquet"))
fact_sales = pd.read_parquet(os.path.join(INT_DIR, "fact_sales.parquet"))
fact_inv = pd.read_parquet(os.path.join(INT_DIR, "fact_inventory.parquet"))
fact_returns = pd.read_parquet(os.path.join(INT_DIR, "fact_returns.parquet"))
fact_canc = pd.read_parquet(os.path.join(INT_DIR, "fact_cancellations.parquet"))
customer_analytics = pd.read_parquet(os.path.join(INT_DIR, "customer_analytics.parquet"))

fact_sales["date"] = pd.to_datetime(fact_sales["date"])
fact_inv["date"] = pd.to_datetime(fact_inv["date"])

print("--- 1. CATEGORY PERFORMANCE (SYNTHETIC) ---")
syn_sales = fact_sales[fact_sales["source_dataset"] == "SYNTHETIC"]
syn_prod = dim_product[dim_product["source_dataset"] == "SYNTHETIC"]
syn_sp = syn_sales.merge(syn_prod[["product_key", "category", "base_price", "cost_price"]], on="product_key", how="left")
syn_cat = syn_sp.groupby("category").agg(
    total_revenue=("revenue", "sum"),
    total_units=("units_sold", "sum"),
    sku_count=("product_key", "nunique"),
    avg_price=("base_price", "mean"),
    avg_cost=("cost_price", "mean")
).reset_index().sort_values("total_revenue", ascending=False)
syn_cat["rev_share_pct"] = syn_cat["total_revenue"] / syn_cat["total_revenue"].sum() * 100
syn_cat["unit_share_pct"] = syn_cat["total_units"] / syn_cat["total_units"].sum() * 100
syn_cat["margin_pct"] = (syn_cat["avg_price"] - syn_cat["avg_cost"]) / syn_cat["avg_price"] * 100
print(syn_cat.to_string(index=False))

print("\n--- 2. STORE PERFORMANCE (SYNTHETIC) ---")
syn_ent = dim_entity[dim_entity["source_dataset"] == "SYNTHETIC"]
syn_store_m = syn_sales.merge(syn_ent, on="entity_id", how="left")
store_perf = syn_store_m.groupby(["entity_id", "store_name", "region", "store_type", "store_size_sqft"]).agg(
    total_revenue=("revenue", "sum"),
    total_units=("units_sold", "sum"),
    transactions=("transaction_count", "sum")
).reset_index().sort_values("total_revenue", ascending=False)
store_perf["rev_share_pct"] = store_perf["total_revenue"] / store_perf["total_revenue"].sum() * 100
store_perf["rev_per_sqft"] = store_perf["total_revenue"] / store_perf["store_size_sqft"]
print(store_perf.to_string(index=False))

print("\n--- 3. PROMOTIONS (SYNTHETIC) ---")
promo_agg = syn_sales.groupby("promotion_flag").agg(
    total_revenue=("revenue", "sum"),
    total_units=("units_sold", "sum"),
    records=("date", "count"),
    avg_price=("average_unit_price", "mean")
).reset_index()
promo_agg["rev_per_rec"] = promo_agg["total_revenue"] / promo_agg["records"]
promo_agg["units_per_rec"] = promo_agg["total_units"] / promo_agg["records"]
print(promo_agg.to_string(index=False))

print("\n--- 4. TOP 10 SKUS (SYNTHETIC) ---")
syn_sku_agg = syn_sales.groupby("product_key").agg(
    revenue=("revenue", "sum"),
    units=("units_sold", "sum")
).reset_index().merge(syn_prod[["product_key", "sku_id", "product_name", "category", "base_price", "cost_price"]], on="product_key", how="left")
syn_sku_agg = syn_sku_agg.sort_values("revenue", ascending=False)
print(syn_sku_agg.head(10)[["sku_id", "product_name", "category", "units", "revenue"]].to_string(index=False))

print("\n--- 5. TOP 10 SKUS (UCI) ---")
uci_sales = fact_sales[fact_sales["source_dataset"] == "UCI"]
uci_prod = dim_product[dim_product["source_dataset"] == "UCI"]
uci_sku_agg = uci_sales.groupby("product_key").agg(
    revenue=("revenue", "sum"),
    units=("units_sold", "sum")
).reset_index().merge(uci_prod[["product_key", "sku_id", "product_name"]], on="product_key", how="left")
uci_sku_agg = uci_sku_agg.sort_values("revenue", ascending=False)
print(uci_sku_agg.head(10)[["sku_id", "product_name", "units", "revenue"]].to_string(index=False))

print("\n--- 6. TOP 10 DESTINATION COUNTRIES (UCI) ---")
uci_cust = customer_analytics[customer_analytics["source_dataset"] == "UCI"]
uci_geo = uci_cust.groupby("country").agg(
    total_revenue=("total_revenue", "sum"),
    total_units=("total_units", "sum"),
    cust_count=("customer_key", "count")
).reset_index().sort_values("total_revenue", ascending=False)
uci_geo["rev_share_pct"] = uci_geo["total_revenue"] / uci_geo["total_revenue"].sum() * 100
print(uci_geo.head(10).to_string(index=False))

print("\n--- 7. REGIONAL SALES (SYNTHETIC) ---")
reg_sales = syn_store_m.groupby("region").agg(
    revenue=("revenue", "sum"),
    units=("units_sold", "sum"),
    stores=("entity_id", "nunique")
).reset_index().sort_values("revenue", ascending=False)
reg_sales["rev_share_pct"] = reg_sales["revenue"] / reg_sales["revenue"].sum() * 100
print(reg_sales.to_string(index=False))
