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
from dashboard.components.theme import inject_theme
from dashboard.components.ui import kv_table, metric_row, page_header, safe_dataframe, show_error

st.set_page_config(page_title="FORESIGHT | Phase 20 Production", layout="wide")
inject_theme()

page_header(
    "Phase 20 Production Forecast",
    "Weekly SKU-level forecasts from the promoted production model.",
)

info = model_info_panel()
with st.expander("Model Information", expanded=True):
    kv_table(
        [
            ("Production Model", info.get("production_model", "phase20_synthetic_lightgbm")),
            ("Source Dataset", info.get("source_dataset", "SYNTHETIC")),
            ("Validated Horizon", info.get("validated_horizon", "6 Weeks")),
            ("Extended Forecast", info.get("extended_forecast", "N/A")),
            ("Overall Validation WAPE", f"{info.get('overall_wape', 'N/A')}%"),
            ("Supported Horizon WAPE (h1–h6)", f"{info.get('supported_horizon_wape', 'N/A')}%"),
            ("Live Production Performance", "PENDING ACTUALS"),
        ]
    )
    st.warning(info.get("known_limitation", "See project documentation for known limitations."))

try:
    bundle = dashboard_bundle()
    records = to_dashboard_records(bundle)
    df = pd.DataFrame(records)
    risk = bundle["risk"]

    metric_row(
        [
            ("SKUs", bundle["sku_count"]),
            ("Forecasts", bundle["forecast_count"]),
            ("REORDER NOW", int((risk["recommended_action"] == "REORDER NOW").sum())),
            ("HEALTHY", int((risk["recommended_action"] == "HEALTHY").sum())),
        ]
    )

    st.subheader("Weekly Demand Forecast (6-Week Production Horizon)")
    safe_dataframe(df[df["horizon"] <= 6] if "horizon" in df.columns else df)

    st.subheader("Risk & Recommended Actions")
    risk_cols = [
        c for c in [
            "sku_id", "forecast_weekly_demand", "on_hand_units", "stockout_risk_level",
            "overstock_risk_level", "recommended_action", "sales_at_risk",
        ] if c in risk.columns
    ]
    safe_dataframe(risk[risk_cols] if risk_cols else risk)

except Exception as ex:
    show_error(str(ex))
    st.info("Run `python src/run_phase20.py` to complete promotion first.")
