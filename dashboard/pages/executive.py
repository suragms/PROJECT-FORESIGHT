"""Phase 23 — Executive dashboard (Phase 22 integration)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import forecast_bar_chart, risk_pie_chart
from dashboard.components.data_loader import load_executive_data
from dashboard.components.status_cards import validation_disclaimer


def render() -> None:
    st.title("Executive Dashboard")

    try:
        data = load_executive_data()
    except Exception as ex:
        st.error(f"Failed to load executive data: {ex}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("System Status", data["system_status"])
    c2.metric("Monitoring", data["monitoring_status"])
    c3.metric("Production Performance", data["production_performance"])
    c4.metric("Health Score", data["monitoring"].get("health_score", "N/A"))
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
            st.dataframe(ext_fc[["horizon", "forecast_demand", "forecast_status"]], use_container_width=True)

    if len(risk) > 0:
        st.subheader("Inventory Risk")
        risk_pie_chart(risk)
        if selected_sku:
            sku_id = str(selected_sku).replace("SYN_", "")
            sku_risk = risk[risk["sku_id"] == sku_id]
            if len(sku_risk) > 0:
                st.dataframe(sku_risk, use_container_width=True)

    bi = data.get("business_impact", {})
    st.subheader("Business Impact")
    if bi.get("status") == "AVAILABLE":
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Sales at Risk", bi.get("total_sales_at_risk", "N/A"))
        bc2.metric("Locked Capital", bi.get("total_locked_capital", "N/A"))
        bc3.metric("At-Risk SKUs", bi.get("at_risk_skus", "N/A"))
        bc4.metric("High Overstock SKUs", bi.get("high_overstock_skus", "N/A"))
    else:
        st.write("NOT AVAILABLE")

    components = data.get("monitoring_components", {})
    if components:
        st.subheader("Model Health")
        st.warning("Forecast Performance: **PENDING ACTUALS**")
        st.json(components)

    alerts = data.get("alerts", [])
    if alerts:
        st.subheader("Recent Alerts")
        st.dataframe(pd.DataFrame(alerts), use_container_width=True)
