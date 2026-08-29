"""Phase 23 — Inventory intelligence pages."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import risk_pie_chart
from dashboard.components.data_loader import load_executive_data


def render(page: str) -> None:
    try:
        data = load_executive_data()
    except Exception as ex:
        st.error(f"Failed to load risk data: {ex}")
        return

    risk = data.get("risk")
    if risk is None or len(risk) == 0:
        st.warning("No risk data available.")
        return

    if page == "inventory_overview":
        st.title("Inventory Overview")
        c1, c2, c3, c4 = st.columns(4)
        actions = risk["recommended_action"].value_counts()
        c1.metric("Total SKUs", len(risk))
        c2.metric("REORDER NOW", int(actions.get("REORDER NOW", 0)))
        c3.metric("HEALTHY", int(actions.get("HEALTHY", 0)))
        c4.metric("WATCH / VOLATILE", int(actions.get("WATCH / VOLATILE", 0)))
        risk_pie_chart(risk)
        st.dataframe(risk, use_container_width=True)

    elif page == "stockout_risk":
        st.title("Stockout Risk")
        filtered = risk[risk["stockout_risk_level"].isin(["HIGH", "CRITICAL", "MEDIUM"])]
        st.metric("At-Risk SKUs", len(filtered))
        display = risk.sort_values("stockout_risk_score", ascending=False) if "stockout_risk_score" in risk.columns else risk
        st.dataframe(display[[c for c in [
            "sku_id", "forecast_weekly_demand", "on_hand_units", "weeks_of_supply",
            "stockout_risk_level", "stockout_risk_score", "recommended_action", "sales_at_risk",
        ] if c in risk.columns]], use_container_width=True)

    elif page == "overstock_risk":
        st.title("Overstock Risk")
        if "overstock_risk_level" in risk.columns:
            filtered = risk[risk["overstock_risk_level"].isin(["HIGH", "SEVERE", "MODERATE"])]
            st.metric("Overstock SKUs", len(filtered))
        st.dataframe(risk[[c for c in [
            "sku_id", "on_hand_units", "forecast_weekly_demand", "weeks_of_supply",
            "overstock_risk_level", "locked_capital", "recommended_action",
        ] if c in risk.columns]], use_container_width=True)

    elif page == "recommendations":
        st.title("Recommendations")
        for action in ["REORDER NOW", "MARKDOWN / CLEAR", "WATCH / VOLATILE", "HEALTHY"]:
            subset = risk[risk["recommended_action"] == action]
            st.markdown(f"### {action} ({len(subset)} SKUs)")
            if len(subset) > 0:
                st.dataframe(subset.head(25), use_container_width=True)
