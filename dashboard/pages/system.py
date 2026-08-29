"""Phase 23 — System information pages."""

from __future__ import annotations

import os

import streamlit as st

from dashboard.components.status_cards import validation_disclaimer
from dashboard.navigation import all_nav_items

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS = os.path.join(BASE, "docs")


def render(page: str) -> None:
    if page == "model_information":
        st.title("Model Information")
        validation_disclaimer()
        st.markdown("""
| Field | Value |
|-------|-------|
| Model Name | phase20_synthetic_lightgbm |
| Model Type | LightGBM |
| Dataset | Synthetic Retail Dataset |
| Forecast Grain | Weekly SKU-level |
| Feature Count | 45 |
| Production Horizon | 6 Weeks |
| Overall Validation WAPE | 13.96% |
| Validated h1–h6 WAPE | 11.03% |
| Live Performance | PENDING ACTUALS |
        """)

    elif page == "documentation":
        st.title("Documentation")
        docs = sorted(f for f in os.listdir(DOCS) if f.endswith(".md") and "phase" in f.lower())
        for doc in docs:
            st.markdown(f"- `{doc}`")
        st.info("Master report: `docs/PROJECT_FORESIGHT_FINAL_REPORT.md`")

    elif page == "validation_status":
        st.title("Validation Status")
        st.markdown("""
| Check | Status |
|-------|--------|
| Full Regression | 241/241 PASS |
| Phase 21 Tests | 24/24 PASS |
| Phase 22 Tests | 27/27 PASS |
| Frozen Models | 12/12 unchanged |
| Delivery Status | PROJECT DELIVERY READY |
| Live Production WAPE | PENDING ACTUALS |
        """)

    elif page == "about":
        st.title("About Project")
        st.markdown("""
**PROJECT FORESIGHT** — AI-Powered Demand & Inventory Intelligence

Project lifecycle:

Data Engineering → Feature Engineering → Forecasting → Candidate Validation →
Promotion Gate → Model Hardening → Production Promotion → Monitoring → Executive Intelligence

**Project Status:** PROJECT DELIVERY READY  
**Regression:** 241/241 PASS

This is a decision-support system. It does not automatically place purchase orders or retrain models in production.
        """)
        st.caption(f"Navigation pages: {len(all_nav_items())}")
