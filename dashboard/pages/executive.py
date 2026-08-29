"""Phase 23 — Executive dashboard (Phase 22 integration)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import forecast_bar_chart, risk_pie_chart
from dashboard.components.data_loader import load_executive_data
from dashboard.components.status_cards import status_badge, validation_disclaimer
from dashboard.components.ui import metric_row, page_header, safe_dataframe, show_error, show_empty


def render() -> None:
    page_header(
        "Executive Dashboard",
        "System status, validated forecasts, inventory risk, and monitoring health.",
    )

    try:
        data = load_executive_data()
    except Exception as ex:
        show_error(str(ex))
        return

    metric_row(
        [
            ("System Status", data["system_status"]),
            ("Monitoring", data["monitoring_status"]),
            ("Production Performance", data["production_performance"]),
            ("Health Score", data["monitoring"].get("health_score", "N/A")),
        ]
    )
    validation_disclaimer()

    forecasts = data["forecasts"]
    risk = data["risk"]
    selected_sku = None

    if len(forecasts) > 0:
        st.subheader("Forecast Overview")
        skus = sorted(forecasts["product_key"].unique())
        selected_sku = st.selectbox("Select SKU", skus, key="exec_sku")
        sku_fc = forecasts[forecasts["product_key"] == selected_sku].sort_values("horizon")
        prod_fc = sku_fc[sku_fc["horizon"] <= 6]
        ext_fc = sku_fc[sku_fc["horizon"] > 6]
        forecast_bar_chart(prod_fc)
        if len(ext_fc) > 0:
            st.warning("Weeks 7–8: **EXTENDED / PARTIAL ACCURACY**")
            safe_dataframe(ext_fc[["horizon", "forecast_demand", "forecast_status"]])
    else:
        show_empty("No forecast data available.")

    if len(risk) > 0:
        st.subheader("Inventory Risk")
        risk_pie_chart(risk)
        if selected_sku:
            sku_id = str(selected_sku).replace("SYN_", "")
            sku_risk = risk[risk["sku_id"] == sku_id]
            if len(sku_risk) > 0:
                safe_dataframe(sku_risk)
    else:
        show_empty("No inventory risk data available.")

    bi = data.get("business_impact", {})
    st.subheader("Business Impact")
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
        show_empty("Business impact metrics are not available for the current extract.")

    components = data.get("monitoring_components", {})
    if components:
        st.subheader("Model Health")
        st.warning("Forecast Performance: **PENDING ACTUALS**")
        cols = st.columns(min(4, len(components)) or 1)
        for i, (k, v) in enumerate(components.items()):
            with cols[i % len(cols)]:
                st.markdown(f"**{k.replace('_', ' ').title()}**")
                st.markdown(status_badge(str(v)), unsafe_allow_html=True)

    alerts = data.get("alerts", [])
    if alerts:
        st.subheader("Recent Alerts")
        safe_dataframe(pd.DataFrame(alerts))
