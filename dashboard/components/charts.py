"""Phase 23 — Reusable chart helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def forecast_bar_chart(df: pd.DataFrame, title: str = "6-Week Production Forecast") -> None:
    if df is None or len(df) == 0:
        st.warning("No forecast data available.")
        return
    fig = px.bar(
        df, x="horizon", y="forecast_demand", title=title,
        labels={"horizon": "Week Ahead", "forecast_demand": "Forecast Demand"},
        color_discrete_sequence=["#ff4b4b"],
    )
    st.plotly_chart(fig, use_container_width=True)


def risk_pie_chart(risk: pd.DataFrame, title: str = "Risk Action Distribution") -> None:
    if risk is None or len(risk) == 0 or "recommended_action" not in risk.columns:
        st.warning("No risk data available.")
        return
    fig = px.pie(risk, names="recommended_action", title=title,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)


def horizon_comparison_chart(horizon_rows: list[dict]) -> None:
    if not horizon_rows:
        st.info("Horizon performance data not available.")
        return
    df = pd.DataFrame(horizon_rows)
    fig = px.bar(
        df, x="horizon", y="wape_pct", color="label",
        title="WAPE by Horizon (Validation Reference)",
        labels={"wape_pct": "WAPE %", "horizon": "Horizon"},
        color_discrete_map={"PRODUCTION": "#ff4b4b", "EXTENDED_PARTIAL": "#94a3b8"},
    )
    st.plotly_chart(fig, use_container_width=True)
