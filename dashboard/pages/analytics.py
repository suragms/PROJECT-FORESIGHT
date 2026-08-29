"""Phase 23 — Analytics pages."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.components.data_loader import load_executive_data, load_features_df, load_forecast_performance
from dashboard.components.status_cards import validation_disclaimer


def render(page: str) -> None:
    feat = load_features_df()
    data = load_executive_data()
    bi = data.get("business_impact", {})

    if page == "business_analytics":
        st.title("Business Analytics")
        if bi.get("status") == "AVAILABLE":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sales at Risk", bi.get("total_sales_at_risk", "N/A"))
            c2.metric("Locked Capital", bi.get("total_locked_capital", "N/A"))
            c3.metric("At-Risk SKUs", bi.get("at_risk_skus", "N/A"))
            c4.metric("High Overstock SKUs", bi.get("high_overstock_skus", "N/A"))
        else:
            st.write("NOT AVAILABLE")
        st.caption("Profit, revenue totals, and customer metrics: NOT AVAILABLE at production grain.")

    elif page == "demand_trends":
        st.title("Demand Trends")
        if len(feat) == 0:
            st.warning("Feature data not available.")
            return
        if "units_sold" not in feat.columns:
            st.write("NOT AVAILABLE")
            return
        weekly = feat.groupby("week")["units_sold"].sum().reset_index()
        fig = px.line(weekly, x="week", y="units_sold", title="Weekly Total Demand (Synthetic Reference)")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "sku_analysis":
        st.title("SKU Analysis")
        if len(feat) == 0:
            st.warning("Feature data not available.")
            return
        sku_vol = feat.groupby("product_key")["units_sold"].sum().sort_values(ascending=False).head(20).reset_index()
        fig = px.bar(sku_vol, x="product_key", y="units_sold", title="Top 20 SKUs by Historical Weekly Demand")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "seasonality":
        st.title("Seasonality Analysis")
        if len(feat) == 0 or "season" not in feat.columns:
            st.write("NOT AVAILABLE")
            return
        season = feat.groupby("season")["units_sold"].mean().reset_index()
        fig = px.bar(season, x="season", y="units_sold", title="Average Demand by Season")
        st.plotly_chart(fig, use_container_width=True)
        if "is_holiday_week" in feat.columns:
            hol = feat.groupby("is_holiday_week")["units_sold"].mean().reset_index()
            st.dataframe(hol, use_container_width=True)

    elif page == "performance_metrics":
        st.title("Performance Metrics")
        validation_disclaimer()
        perf = load_forecast_performance() or {}
        if perf:
            c1, c2 = st.columns(2)
            c1.metric("Validation Overall WAPE %", perf.get("overall_wape_pct", "N/A"))
            c2.metric("Validation h1–h6 WAPE %", perf.get("h1_h6_wape_pct", "N/A"))
            st.caption(f"Status: {perf.get('production_actuals_status', 'PENDING ACTUALS')}")
            if perf.get("horizon_performance"):
                st.dataframe(perf["horizon_performance"], use_container_width=True)
        else:
            st.info("Run Phase 21 monitoring for performance artifacts.")
