"""Phase 23.5 — Home / landing page for authenticated users."""

from __future__ import annotations

import streamlit as st

from dashboard.components.data_loader import load_executive_data, load_monitoring_json
from dashboard.components.status_cards import metric_card, status_badge, validation_disclaimer
from dashboard.components.ui import page_header, show_empty, show_error


def _fmt_inr(value) -> str:
    try:
        n = float(value)
        return f"₹{n:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def render() -> None:
    st.markdown(
        """
<div style="margin-bottom:0.35rem;">
  <div style="font-size:1.65rem;font-weight:800;color:#0f172a;letter-spacing:-0.02em;">
    🚀 PROJECT FORESIGHT
  </div>
  <div style="font-size:1rem;font-weight:600;color:#334155;margin-top:0.2rem;">
    AI-Powered Demand &amp; Inventory Intelligence Platform
  </div>
  <div style="font-size:0.9rem;color:#64748b;margin-top:0.45rem;max-width:42rem;line-height:1.45;">
    Transforming retail data into actionable demand forecasts,
    inventory intelligence, and business recommendations.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    try:
        data = load_executive_data()
    except Exception as ex:
        show_error(str(ex))
        return

    impact = data.get("business_impact") or {}
    integrity = load_monitoring_json("model_integrity_report.json") or {}
    components = data.get("monitoring_components") or {}

    page_header("Key Metrics", "From production forecasts and inventory risk scoring.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total SKUs (Forecast)", str(data.get("sku_count", "N/A")))
    with c2:
        metric_card("Forecast Rows", str(data.get("forecast_count", "N/A")))
    with c3:
        at_risk = impact.get("at_risk_skus", "N/A")
        metric_card("Stockout-Risk SKUs", str(at_risk))
    with c4:
        overstock = impact.get("high_overstock_skus", "N/A")
        metric_card("Overstock-Risk SKUs", str(overstock))

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        sales = impact.get("total_sales_at_risk", "N/A")
        metric_card("Sales at Risk", _fmt_inr(sales) if sales != "NOT AVAILABLE" else "N/A")
    with c6:
        locked = impact.get("total_locked_capital", "N/A")
        metric_card("Locked Capital", _fmt_inr(locked) if locked != "NOT AVAILABLE" else "N/A")
    with c7:
        metric_card("Validation WAPE", f"{data.get('validation_overall_wape', 'N/A')}%")
    with c8:
        metric_card("Validated Horizon", "6 Weeks")

    validation_disclaimer()

    st.subheader("Quick Insights")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("**Recent Forecast Summary**")
        st.write(
            f"Production model **phase20_synthetic_lightgbm** · "
            f"Weekly SKU grain · {data.get('forecast_count', 'N/A')} forecast rows."
        )
        st.caption("Live production performance: PENDING ACTUALS")
    with i2:
        st.markdown("**Inventory Risk Summary**")
        if impact.get("status") == "AVAILABLE":
            st.write(
                f"High/Critical stockout SKUs: **{impact.get('at_risk_skus', 'N/A')}** · "
                f"High/Severe overstock SKUs: **{impact.get('high_overstock_skus', 'N/A')}**"
            )
        else:
            show_empty("Inventory risk summary not available.")
    with i3:
        st.markdown("**Model Status**")
        integrity_status = "PASS" if integrity.get("status") == "PASS" else integrity.get("status", "NOT_RUN")
        st.markdown(f"Integrity: {status_badge(integrity_status)}", unsafe_allow_html=True)
        st.markdown(
            f"Monitoring: {status_badge(str(data.get('monitoring_status', 'NOT_RUN')))}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"Live: {status_badge('PENDING ACTUALS')}",
            unsafe_allow_html=True,
        )

    st.subheader("Monitoring Snapshot")
    if components:
        cols = st.columns(min(4, len(components)))
        for i, (k, v) in enumerate(components.items()):
            with cols[i % len(cols)]:
                st.markdown(f"**{k.replace('_', ' ').title()}**")
                st.markdown(status_badge(str(v)), unsafe_allow_html=True)
    else:
        st.info("Run Phase 21 monitoring to populate component statuses.")
