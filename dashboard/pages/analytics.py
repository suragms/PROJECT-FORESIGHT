"""Phase 23 — Analytics pages."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.components.data_loader import load_executive_data, load_features_df, load_forecast_performance
from dashboard.components.status_cards import validation_disclaimer
from dashboard.components.ui import metric_row, page_header, safe_dataframe, show_empty


def render(page: str) -> None:
    feat = load_features_df()
    data = load_executive_data()
    bi = data.get("business_impact", {})

    if page == "business_analytics":
        page_header("Business Analytics", "Inventory risk impact metrics from the production risk extract.")
        if bi.get("status") == "AVAILABLE":
            metric_row(
                [
                    ("Sales at Risk", bi.get("total_sales_at_risk", "N/A")),
                    ("Locked Capital", bi.get("total_locked_capital", "N/A")),
                    ("At-Risk SKUs", bi.get("at_risk_skus", "N/A")),
                    ("High Overstock SKUs", bi.get("high_overstock_skus", "N/A")),
                ]
            )
        else:
            show_empty("Business impact metrics are not available.")
        st.caption("Profit, revenue totals, and customer metrics: not available at production grain.")

    elif page == "demand_trends":
        page_header("Demand Trends", "Historical weekly demand from the synthetic feature matrix.")
        if len(feat) == 0:
            show_empty("Feature data not available.")
            return
        if "units_sold" not in feat.columns:
            show_empty("Demand column not available in the feature extract.")
            return
        week_col = "week" if "week" in feat.columns else None
        if week_col is None:
            show_empty("Week column not available for trend charting.")
            return
        weekly = feat.groupby(week_col)["units_sold"].sum().reset_index()
        fig = px.line(
            weekly, x=week_col, y="units_sold",
            title="Weekly Total Demand (Synthetic Reference)",
            labels={week_col: "Week", "units_sold": "Units Sold"},
            color_discrete_sequence=["#ff4b4b"],
        )
        st.plotly_chart(fig, use_container_width=True)

    elif page == "sku_analysis":
        page_header("SKU Analysis", "Top SKUs by historical weekly demand.")
        if len(feat) == 0 or "units_sold" not in feat.columns:
            show_empty("Feature data not available.")
            return
        sku_vol = (
            feat.groupby("product_key")["units_sold"]
            .sum()
            .sort_values(ascending=False)
            .head(20)
            .reset_index()
        )
        fig = px.bar(
            sku_vol, x="product_key", y="units_sold",
            title="Top 20 SKUs by Historical Weekly Demand",
            labels={"product_key": "SKU", "units_sold": "Units Sold"},
            color_discrete_sequence=["#ff4b4b"],
        )
        st.plotly_chart(fig, use_container_width=True)
        safe_dataframe(sku_vol)

    elif page == "seasonality":
        page_header("Seasonality Analysis", "Average demand by season and holiday weeks.")
        if len(feat) == 0 or "season" not in feat.columns or "units_sold" not in feat.columns:
            show_empty("Seasonality fields are not available in the feature extract.")
            return
        season = feat.groupby("season")["units_sold"].mean().reset_index()
        fig = px.bar(
            season, x="season", y="units_sold",
            title="Average Demand by Season",
            labels={"season": "Season", "units_sold": "Avg Units Sold"},
            color_discrete_sequence=["#ff4b4b"],
        )
        st.plotly_chart(fig, use_container_width=True)
        if "is_holiday_week" in feat.columns:
            hol = feat.groupby("is_holiday_week")["units_sold"].mean().reset_index()
            hol.columns = ["Is Holiday Week", "Avg Units Sold"]
            safe_dataframe(hol)

    elif page == "performance_metrics":
        page_header("Performance Metrics", "Validation / backtest WAPE by horizon.")
        validation_disclaimer()
        perf = load_forecast_performance() or {}
        if perf:
            metric_row(
                [
                    ("Validation Overall WAPE %", perf.get("overall_wape_pct", "N/A")),
                    ("Validation h1–h6 WAPE %", perf.get("h1_h6_wape_pct", "N/A")),
                ]
            )
            st.caption(f"Live production status: {perf.get('production_actuals_status', 'PENDING ACTUALS')}")
            if perf.get("horizon_performance"):
                safe_dataframe(perf["horizon_performance"])
        else:
            st.info("Run Phase 21 monitoring for performance artifacts.")
