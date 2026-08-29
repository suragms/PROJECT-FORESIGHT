"""Phase 23 — Home page."""

from __future__ import annotations

import streamlit as st

from dashboard.components.data_loader import load_executive_data, load_monitoring_json
from dashboard.components.status_cards import metric_card, status_badge, validation_disclaimer


def render() -> None:
    st.title("PROJECT FORESIGHT")
    st.markdown("### AI-Powered Demand & Inventory Intelligence Platform")

    try:
        data = load_executive_data()
    except Exception as ex:
        st.error(f"Unable to load platform data: {ex}")
        return

    components = data.get("monitoring_components", {})
    integrity = load_monitoring_json("model_integrity_report.json") or {}

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Production Model", "phase20_synthetic_lightgbm")
    with c2:
        metric_card("Forecast Horizon", "6 Weeks")
    with c3:
        metric_card("Monitoring", data.get("monitoring_status", "NOT_RUN"))
    with c4:
        integrity_status = "PASS" if integrity.get("status") == "PASS" else integrity.get("status", "NOT_RUN")
        st.markdown("**Model Integrity**")
        st.markdown(status_badge(integrity_status), unsafe_allow_html=True)
    with c5:
        st.markdown("**Live Performance**")
        st.markdown(status_badge("PENDING ACTUALS"), unsafe_allow_html=True)

    validation_disclaimer()

    st.subheader("System Overview")
    st.markdown("""
The platform provides:

- **Demand forecasting** — 6-week weekly SKU-level production forecasts
- **Inventory risk detection** — stockout and overstock scoring
- **Decision recommendations** — REORDER NOW, WATCH, HEALTHY, MARKDOWN / CLEAR
- **Production monitoring** — data quality, drift, alerts, integrity
- **Model integrity verification** — SHA-256 checks on frozen artifacts
    """)

    st.subheader("Monitoring Snapshot")
    if components:
        cols = st.columns(4)
        for i, (k, v) in enumerate(components.items()):
            with cols[i % 4]:
                st.markdown(f"**{k.replace('_', ' ').title()}**")
                st.markdown(status_badge(v), unsafe_allow_html=True)
    else:
        st.warning("Run `python src/run_phase21.py` to populate monitoring status.")
