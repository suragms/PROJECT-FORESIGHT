"""
Phase 20 Production Dashboard
Run: streamlit run dashboard/phase20_production.py
"""

import os
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.phase20_dashboard_adapter import dashboard_bundle, model_info_panel, to_dashboard_records

st.set_page_config(page_title="FORESIGHT | Phase 20 Production", layout="wide")

st.title("FORESIGHT — Phase 20 Production Forecast")

info = model_info_panel()
with st.expander("Model Information", expanded=True):
    st.markdown(f"""
| Field | Value |
|-------|-------|
| **Production Model** | {info['production_model']} |
| **Source Dataset** | {info['source_dataset']} |
| **Validated Horizon** | {info['validated_horizon']} |
| **Extended Forecast** | {info['extended_forecast']} |
| **Overall WAPE** | {info.get('overall_wape', 'N/A')}% |
| **Supported Horizon WAPE** | {info.get('supported_horizon_wape', 'N/A')}% |
    """)
    st.warning(info["known_limitation"])

try:
    bundle = dashboard_bundle()
    records = to_dashboard_records(bundle)
    df = pd.DataFrame(records)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SKUs", bundle["sku_count"])
    col2.metric("Forecasts", bundle["forecast_count"])
    risk = bundle["risk"]
    col3.metric("REORDER NOW", int((risk["recommended_action"] == "REORDER NOW").sum()))
    col4.metric("HEALTHY", int((risk["recommended_action"] == "HEALTHY").sum()))

    st.subheader("Weekly Demand Forecast (6-Week Production Horizon)")
    st.dataframe(df[df["horizon"] <= 6], use_container_width=True)

    st.subheader("Risk & Recommended Actions")
    risk_display = risk[[
        "sku_id", "forecast_weekly_demand", "on_hand_units", "stockout_risk_level",
        "overstock_risk_level", "recommended_action", "sales_at_risk"
    ]].copy()
    st.dataframe(risk_display, use_container_width=True)

except Exception as ex:
    st.error(f"Failed to load production data: {ex}")
    st.info("Run `python src/run_phase20.py` to complete promotion first.")
