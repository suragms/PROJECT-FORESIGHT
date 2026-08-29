"""
PROJECT FORESIGHT — Unified Analytics Platform
Run: streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dashboard.components.auth_ui import render_auth_screen
from dashboard.components.sidebar import render_sidebar
from dashboard.components.theme import inject_theme
from dashboard.pages import analytics, executive, forecasting, home, inventory, ml, monitoring, system
from dashboard.session_auth import is_authenticated, page_allowed, current_role

st.set_page_config(
    page_title="PROJECT FORESIGHT | Unified Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

FORECAST_PAGES = {"forecasting", "forecast_explorer", "horizon_analysis"}
INVENTORY_PAGES = {"inventory_overview", "stockout_risk", "overstock_risk", "recommendations"}
ANALYTICS_PAGES = {"business_analytics", "demand_trends", "sku_analysis", "seasonality", "performance_metrics"}
ML_PAGES = {"model_overview", "feature_contract", "model_performance", "model_explainability"}
MONITORING_PAGES = {"system_health", "data_quality", "data_drift", "prediction_drift", "alerts", "integrity"}
SYSTEM_PAGES = {"model_information", "documentation", "validation_status", "about"}


def route_page(page_key: str) -> None:
    if not page_allowed(page_key, current_role()):
        st.error("403 Forbidden — session invalid. Please log in again.")
        return

    if page_key == "home":
        home.render()
    elif page_key == "executive":
        executive.render()
    elif page_key in FORECAST_PAGES:
        forecasting.render(page_key)
    elif page_key in INVENTORY_PAGES:
        inventory.render(page_key)
    elif page_key in ANALYTICS_PAGES:
        analytics.render(page_key)
    elif page_key in ML_PAGES:
        ml.render(page_key)
    elif page_key in MONITORING_PAGES:
        monitoring.render(page_key)
    elif page_key in SYSTEM_PAGES:
        system.render(page_key)
    else:
        st.error(f"Unknown page: {page_key}")


if not is_authenticated():
    render_auth_screen()
    st.stop()

with st.sidebar:
    page = render_sidebar()

route_page(page)
