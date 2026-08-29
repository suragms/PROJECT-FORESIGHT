"""Phase 23 — Forecasting pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.charts import forecast_bar_chart, horizon_comparison_chart
from dashboard.components.data_loader import load_forecast_performance, load_production_bundle
from dashboard.components.status_cards import validation_disclaimer
from dashboard.components.ui import page_header, safe_dataframe, show_empty


def _model_banner() -> None:
    st.caption(
        "Production Model: **phase20_synthetic_lightgbm** · Grain: Weekly SKU-level · "
        "Validated Horizon: **6 Weeks** · Source: SYNTHETIC"
    )
    validation_disclaimer()


def render(page: str) -> None:
    bundle = load_production_bundle()
    forecasts = bundle.get("forecasts", pd.DataFrame())
    perf = load_forecast_performance() or {}

    if page == "forecasting":
        page_header("Demand Forecasting", "Validated 6-week production forecasts by SKU.")
        _model_banner()
        if len(forecasts) == 0:
            show_empty("No forecasts available.")
            return
        skus = sorted(forecasts["product_key"].unique())
        sku = st.selectbox("SKU", skus, key="fc_sku")
        sku_fc = forecasts[forecasts["product_key"] == sku].sort_values("horizon")
        prod = sku_fc[sku_fc["horizon"] <= 6]
        forecast_bar_chart(prod, "Validated 6-Week Production Forecast")
        safe_dataframe(prod)

    elif page == "forecast_explorer":
        page_header("Forecast Explorer", "Cross-SKU forecast totals and ranking.")
        _model_banner()
        if len(forecasts) == 0:
            show_empty("No forecasts available.")
            return
        summary = forecasts.groupby(["product_key", "horizon"])["forecast_demand"].sum().reset_index()
        safe_dataframe(summary)
        top = forecasts.groupby("product_key")["forecast_demand"].sum().nlargest(10).reset_index()
        fig = px.bar(
            top, x="product_key", y="forecast_demand",
            title="Top 10 SKUs by Total Forecast",
            labels={"product_key": "SKU", "forecast_demand": "Forecast Demand"},
            color_discrete_sequence=["#ff4b4b"],
        )
        st.plotly_chart(fig, use_container_width=True)

    elif page == "horizon_analysis":
        page_header("Horizon Analysis", "Validation WAPE by forecast horizon (backtest reference).")
        _model_banner()
        horizon_rows = perf.get("horizon_performance", [])
        if horizon_rows:
            horizon_comparison_chart(horizon_rows)
            safe_dataframe(pd.DataFrame(horizon_rows))
            st.caption("h7–h8 labeled EXTENDED_PARTIAL — not a primary production KPI.")
        else:
            if len(forecasts) > 0:
                agg = forecasts.groupby("horizon").agg(
                    mean_demand=("forecast_demand", "mean"),
                    total_demand=("forecast_demand", "sum"),
                    sku_count=("product_key", "nunique"),
                ).reset_index()
                agg["label"] = agg["horizon"].apply(
                    lambda h: "PRODUCTION" if h <= 6 else "EXTENDED_PARTIAL"
                )
                safe_dataframe(agg)
            else:
                show_empty("Horizon data not available.")
