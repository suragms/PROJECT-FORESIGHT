"""Phase 23 — Inventory intelligence pages."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import risk_pie_chart
from dashboard.components.data_loader import load_executive_data
from dashboard.components.ui import metric_row, page_header, safe_dataframe, show_empty, show_error


def render(page: str) -> None:
    try:
        data = load_executive_data()
    except Exception as ex:
        show_error(str(ex))
        return

    risk = data.get("risk")
    if risk is None or len(risk) == 0:
        show_empty("No risk data available.")
        return

    if page == "inventory_overview":
        page_header("Inventory Overview", "Risk action distribution across the production SKU extract.")
        actions = risk["recommended_action"].value_counts()
        metric_row(
            [
                ("Total SKUs", len(risk)),
                ("REORDER NOW", int(actions.get("REORDER NOW", 0))),
                ("HEALTHY", int(actions.get("HEALTHY", 0))),
                ("WATCH / VOLATILE", int(actions.get("WATCH / VOLATILE", 0))),
            ]
        )
        risk_pie_chart(risk)
        safe_dataframe(risk)

    elif page == "stockout_risk":
        page_header("Stockout Risk", "SKUs ranked by stockout exposure versus forecast demand.")
        filtered = risk[risk["stockout_risk_level"].isin(["HIGH", "CRITICAL", "MEDIUM"])]
        st.metric("At-Risk SKUs", len(filtered))
        display = (
            risk.sort_values("stockout_risk_score", ascending=False)
            if "stockout_risk_score" in risk.columns
            else risk
        )
        cols = [
            c for c in [
                "sku_id", "forecast_weekly_demand", "on_hand_units", "weeks_of_supply",
                "stockout_risk_level", "stockout_risk_score", "recommended_action", "sales_at_risk",
            ] if c in risk.columns
        ]
        safe_dataframe(display[cols] if cols else display)

    elif page == "overstock_risk":
        page_header("Overstock Risk", "SKUs with elevated inventory relative to forecast demand.")
        if "overstock_risk_level" in risk.columns:
            filtered = risk[risk["overstock_risk_level"].isin(["HIGH", "SEVERE", "MODERATE"])]
            st.metric("Overstock SKUs", len(filtered))
        cols = [
            c for c in [
                "sku_id", "on_hand_units", "forecast_weekly_demand", "weeks_of_supply",
                "overstock_risk_level", "locked_capital", "recommended_action",
            ] if c in risk.columns
        ]
        safe_dataframe(risk[cols] if cols else risk)

    elif page == "recommendations":
        page_header("Recommendations", "Prioritized actions from the inventory decision grid.")
        for action in ["REORDER NOW", "MARKDOWN / CLEAR", "WATCH / VOLATILE", "HEALTHY"]:
            subset = risk[risk["recommended_action"] == action]
            st.markdown(f"### {action} ({len(subset)} SKUs)")
            if len(subset) > 0:
                safe_dataframe(subset.head(25))
            else:
                show_empty(f"No SKUs currently tagged {action}.")
