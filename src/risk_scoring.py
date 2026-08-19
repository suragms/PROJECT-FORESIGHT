"""
Phase 10 — Inventory Risk Scoring & Optimization Engine
=========================================================
Project FORESIGHT: Demand & Inventory Intelligence

Computes Days of Supply (DOS), Stockout Risk Scores, Overstock Capital,
Reorder Point (ROP) breaches, Recommended Order Quantities (ROQ/EOQ),
and automated answers to the 10 Core Business Questions.
"""

import os
import numpy as np
import pandas as pd

from src.data_integration import (
    load_sales_daily,
    load_inventory_snapshots,
    load_sku_master,
    load_store_master,
    load_calendar,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RISK_DIR = os.path.join(BASE_DIR, "outputs", "risk_scores")
os.makedirs(RISK_DIR, exist_ok=True)


def calculate_inventory_risk_matrix(
    lookback_days: int = 30,
    target_coverage_days: int = 45,
    holding_cost_annual_rate: float = 0.25,
) -> pd.DataFrame:
    """
    Compute comprehensive inventory risk and replenishment metrics across all active Store-SKU pairs.
    """
    sales = load_sales_daily()
    inventory = load_inventory_snapshots()
    skus = load_sku_master()
    stores = load_store_master()

    # Get recent sales rate
    max_date = sales["date"].max()
    recent_start = max_date - pd.Timedelta(days=lookback_days)
    recent_sales = sales[sales["date"] >= recent_start]

    demand_rate = recent_sales.groupby(["store_id", "sku_id"]).agg(
        avg_daily_demand=("units_sold", "mean"),
        std_daily_demand=("units_sold", "std"),
        total_recent_units=("units_sold", "sum"),
        total_recent_revenue=("total_revenue", "sum"),
        avg_price=("avg_unit_price", "mean"),
    ).reset_index()
    demand_rate["std_daily_demand"] = demand_rate["std_daily_demand"].fillna(0.0)

    # Get latest inventory snapshot
    latest_inv_date = inventory["date"].max()
    latest_inv = inventory[inventory["date"] == latest_inv_date].copy()

    # Merge snapshot with demand rates
    merged = pd.merge(latest_inv, demand_rate, on=["store_id", "sku_id"], how="left")
    merged["avg_daily_demand"] = merged["avg_daily_demand"].fillna(0.1)
    merged["std_daily_demand"] = merged["std_daily_demand"].fillna(0.0)

    # Merge SKU Master
    sku_cols = [
        "sku_id", "sku_name", "category", "sub_category", "brand",
        "cost_price", "base_price", "supplier_id", "lead_time_days",
        "reorder_point", "safety_stock"
    ]
    merged = pd.merge(merged, skus[sku_cols], on="sku_id", how="left")

    # Merge Store Master
    store_cols = ["store_id", "store_name", "city", "state", "region", "store_type"]
    merged = pd.merge(merged, stores[store_cols], on="store_id", how="left")

    # 1. Days of Supply (DOS / DOI)
    merged["effective_daily_demand"] = np.maximum(merged["avg_daily_demand"], 0.05)
    merged["days_of_supply"] = merged["ending_inventory"] / merged["effective_daily_demand"]

    # 2. Dynamic Lead Time Demand & Safety Stock verification
    merged["lead_time_demand"] = merged["effective_daily_demand"] * merged["lead_time_days"]
    merged["dynamic_rop"] = merged["lead_time_demand"] + merged["safety_stock"]

    # 3. Stockout Risk Scoring (0 to 100)
    # Risk factors: inventory < safety stock, DOS < lead time, current inventory <= 0
    inventory_position = merged["ending_inventory"] + merged["on_order_qty"]

    stockout_score = np.zeros(len(merged))
    # Immediate zero stock
    stockout_score = np.where(merged["ending_inventory"] <= 0, 100.0, stockout_score)
    # Below safety stock
    below_ss = (merged["ending_inventory"] > 0) & (merged["ending_inventory"] < merged["safety_stock"])
    stockout_score = np.where(below_ss, 75.0 + (1 - merged["ending_inventory"] / np.maximum(1, merged["safety_stock"])) * 20.0, stockout_score)
    # Below lead time demand but above safety stock
    below_ltd = (merged["ending_inventory"] >= merged["safety_stock"]) & (merged["days_of_supply"] < merged["lead_time_days"])
    stockout_score = np.where(below_ltd, 50.0 + (1 - merged["days_of_supply"] / np.maximum(1, merged["lead_time_days"])) * 24.0, stockout_score)
    # Breached ROP
    below_rop = (stockout_score == 0) & (inventory_position <= merged["reorder_point"])
    stockout_score = np.where(below_rop, 35.0, stockout_score)

    merged["stockout_risk_score"] = np.round(np.clip(stockout_score, 0, 100), 1)
    merged["stockout_risk_level"] = np.select(
        [merged["stockout_risk_score"] >= 70, merged["stockout_risk_score"] >= 35],
        ["CRITICAL / HIGH", "MEDIUM (REORDER)"],
        default="LOW / SAFE"
    )

    # 4. Overstock Risk Scoring (0 to 100) & Capital Tied Up
    excess_days = np.maximum(0, merged["days_of_supply"] - target_coverage_days)
    overstock_score = np.clip((excess_days / target_coverage_days) * 50.0, 0, 100.0)
    # Amplify if sales velocity is near zero
    overstock_score = np.where(merged["avg_daily_demand"] < 0.5, np.minimum(100.0, overstock_score + 25.0), overstock_score)

    merged["overstock_risk_score"] = np.round(overstock_score, 1)
    merged["overstock_risk_level"] = np.select(
        [merged["overstock_risk_score"] >= 65, merged["overstock_risk_score"] >= 30],
        ["SEVERE OVERSTOCK", "MODERATE OVERSTOCK"],
        default="OPTIMAL"
    )

    # Financial Exposure
    merged["excess_units"] = np.maximum(0, merged["ending_inventory"] - np.round(merged["effective_daily_demand"] * target_coverage_days)).astype(int)
    merged["capital_tied_up"] = merged["excess_units"] * merged["cost_price"]
    merged["annual_holding_cost"] = merged["capital_tied_up"] * holding_cost_annual_rate
    merged["potential_lost_daily_revenue"] = np.where(
        merged["ending_inventory"] <= 0,
        merged["effective_daily_demand"] * merged["base_price"],
        0.0
    )

    # 5. Replenishment Trigger & Recommended Order Quantity (ROQ)
    # Physical shelf position drives replenishment review (on-order does not hide zero shelf stock).
    merged["rop_position_triggered"] = inventory_position <= merged["reorder_point"]
    merged["reorder_triggered"] = merged["ending_inventory"] <= merged["reorder_point"]

    # Target maximum inventory S = LTD + SS + Review cycle (e.g. 14 days)
    target_max_inventory = (merged["effective_daily_demand"] * (merged["lead_time_days"] + 14)) + merged["safety_stock"]
    merged["recommended_reorder_qty"] = np.where(
        merged["reorder_triggered"],
        np.maximum(10, np.round(target_max_inventory - merged["ending_inventory"])).astype(int),
        0
    )
    merged["recommended_order_spend"] = merged["recommended_reorder_qty"] * merged["cost_price"]
    merged["replenishment_note"] = np.where(
        merged["reorder_triggered"] & (merged["on_order_qty"] > 0),
        "Shelf at/below ROP; on-order qty in transit — draft review only, not a supplier PO.",
        np.where(
            merged["reorder_triggered"],
            "Shelf at/below ROP — draft review only, not a supplier PO.",
            "",
        ),
    )

    # Save to disk
    output_path = os.path.join(RISK_DIR, "inventory_risk_matrix.parquet")
    merged.to_parquet(output_path, index=False)

    return merged


def answer_10_core_questions() -> dict:
    """
    Generate authoritative, data-backed analytical answers to the 10 Core Business Questions.
    """
    sales = load_sales_daily()
    inventory = load_inventory_snapshots()
    skus = load_sku_master()
    stores = load_store_master()
    calendar = load_calendar()
    risk_df = calculate_inventory_risk_matrix()

    # 1. Top Products
    sku_agg = sales.groupby("sku_id").agg(
        total_units=("units_sold", "sum"),
        total_revenue=("total_revenue", "sum"),
    ).reset_index()
    sku_agg = pd.merge(sku_agg, skus[["sku_id", "sku_name", "category", "brand", "cost_price", "base_price"]], on="sku_id", how="left")
    sku_agg["gross_profit"] = sku_agg["total_revenue"] - (sku_agg["total_units"] * sku_agg["cost_price"])
    sku_agg["margin_pct"] = (sku_agg["gross_profit"] / sku_agg["total_revenue"]) * 100
    top_10 = sku_agg.sort_values(by="total_revenue", ascending=False).head(10).to_dict(orient="records")

    # 2. Bottom Products / Deadstock
    bottom_10 = sku_agg.sort_values(by="total_revenue", ascending=True).head(10).to_dict(orient="records")

    # 3. Demand Dynamics Across Channels & Regions
    sales_with_dims = pd.merge(sales, stores[["store_id", "region", "store_type"]], on="store_id", how="left")
    channel_dynamics = sales_with_dims.groupby(["store_type", "region"]).agg(
        total_revenue=("total_revenue", "sum"),
        total_units=("units_sold", "sum"),
        avg_price=("avg_unit_price", "mean")
    ).reset_index().to_dict(orient="records")

    # 4. Seasonality Analysis
    sales_with_cal = pd.merge(sales, calendar[["date", "year", "month", "quarter", "is_holiday", "season", "day_name"]], on="date", how="left")
    monthly_seasonality = sales_with_cal.groupby("month").agg(
        total_revenue=("total_revenue", "sum"),
        total_units=("units_sold", "sum")
    ).reset_index().to_dict(orient="records")
    holiday_uplift = float(
        sales_with_cal.groupby("is_holiday")["units_sold"].mean().get(1, 0) /
        max(0.01, sales_with_cal.groupby("is_holiday")["units_sold"].mean().get(0, 1)) - 1
    ) * 100

    # 5. Demand Growth Trajectory
    yearly = sales_with_cal.groupby(["year", "sku_id"]).agg(units=("units_sold", "sum")).reset_index()
    # Growth between 2022 and 2025
    y22 = yearly[yearly["year"] == 2022].set_index("sku_id")["units"]
    y25 = yearly[yearly["year"] == 2025].set_index("sku_id")["units"]
    growth = ((y25 - y22) / np.maximum(1, y22) * 100).dropna()
    top_growth_skus = growth.sort_values(ascending=False).head(5).to_dict()

    # 6. Future Demand Overview
    total_avg_daily_demand = float(sales.groupby("date")["units_sold"].sum().mean())
    forecast_30d_projected = total_avg_daily_demand * 30

    # 7. Stockout Risk Summary
    high_stockout = risk_df[risk_df["stockout_risk_level"] == "CRITICAL / HIGH"]
    stockout_count = len(high_stockout)
    potential_lost_rev = float(high_stockout["potential_lost_daily_revenue"].sum() * 30)

    # 8. Overstock Risk Summary
    overstock_items = risk_df[risk_df["overstock_risk_level"] == "SEVERE OVERSTOCK"]
    total_capital_tied = float(risk_df["capital_tied_up"].sum())
    annual_carrying_cost = float(risk_df["annual_holding_cost"].sum())

    # 9. Replenishment Triggers
    reorder_items = risk_df[risk_df["reorder_triggered"]].sort_values(by="recommended_order_spend", ascending=False)
    reorder_count = len(reorder_items)
    total_reorder_spend = float(reorder_items["recommended_order_spend"].sum())

    # 10. Actionable Recommendations
    recommendations = [
        f"Immediate Purchase Orders: {reorder_count} Store-SKU items have shelf stock at or below their Reorder Point (ROP). Draft review spend totals ${total_reorder_spend:,.2f}. These are analytical recommendations — purchase orders are not sent automatically.",
        f"Critical Stockout Mitigation: {stockout_count} items are at critical stockout risk with potential monthly revenue loss of ${potential_lost_rev:,.2f}. Expedite lead times for high-margin category leaders.",
        f"Overstock Liquidation: ${total_capital_tied:,.2f} in working capital is locked in excess inventory costing ${annual_carrying_cost:,.2f}/yr in holding charges. Deploy targeted 10-15% promotional markdowns on severe overstock SKUs.",
        f"Seasonal Inventory Staging: Holiday periods show a +{holiday_uplift:.1f}% surge in daily unit velocity. Increase safety stock buffer by 25% starting 3 weeks prior to major holiday peaks.",
        "Supplier Lead Time SLA Enforcement: Prioritize suppliers with lead times > 14 days for dynamic safety stock expansion to avoid buffer depletion.",
    ]

    return {
        "q1_top_products": top_10,
        "q2_bottom_products": bottom_10,
        "q3_demand_dynamics": channel_dynamics,
        "q4_seasonality": {
            "monthly_profile": monthly_seasonality,
            "holiday_uplift_pct": round(holiday_uplift, 2),
        },
        "q5_growth": top_growth_skus,
        "q6_future_demand_30d": round(forecast_30d_projected, 2),
        "q7_stockout_risk": {
            "critical_count": stockout_count,
            "potential_lost_revenue_monthly": round(potential_lost_rev, 2),
            "top_stockout_skus": high_stockout[["store_id", "sku_id", "sku_name", "ending_inventory", "days_of_supply", "safety_stock"]].head(10).to_dict(orient="records"),
        },
        "q8_overstock_risk": {
            "severe_overstock_count": len(overstock_items),
            "total_capital_tied_up": round(total_capital_tied, 2),
            "annual_holding_cost": round(annual_carrying_cost, 2),
            "top_overstock_skus": overstock_items[["store_id", "sku_id", "sku_name", "ending_inventory", "days_of_supply", "capital_tied_up"]].sort_values(by="capital_tied_up", ascending=False).head(10).to_dict(orient="records"),
        },
        "q9_replenishment": {
            "reorder_triggered_count": reorder_count,
            "total_reorder_spend": round(total_reorder_spend, 2),
            "top_purchase_orders": reorder_items[["store_id", "sku_id", "sku_name", "supplier_id", "ending_inventory", "reorder_point", "recommended_reorder_qty", "recommended_order_spend"]].head(15).to_dict(orient="records"),
        },
        "q10_recommendations": recommendations,
    }
