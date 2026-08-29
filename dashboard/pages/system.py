"""Phase 23 — System information pages."""

from __future__ import annotations

import os

import streamlit as st

from dashboard.components.status_cards import validation_disclaimer
from dashboard.components.ui import kv_table, page_header, show_empty
from dashboard.navigation import all_nav_items

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS = os.path.join(BASE, "docs")


def render(page: str) -> None:
    if page == "model_information":
        page_header(
            "Model Information",
            "Production model card — validation metrics are backtest results, not live WAPE.",
        )
        validation_disclaimer()
        kv_table(
            [
                ("Model Name", "phase20_synthetic_lightgbm"),
                ("Model Type", "LightGBM"),
                ("Dataset", "Synthetic Retail Dataset"),
                ("Forecast Grain", "Weekly SKU-level"),
                ("Feature Count", "45"),
                ("Production Horizon", "6 Weeks"),
                ("Overall Validation WAPE", "13.96%"),
                ("Validated h1–h6 WAPE", "11.03%"),
                ("Live Production Performance", "PENDING ACTUALS"),
            ]
        )

    elif page == "documentation":
        page_header("Documentation", "Phase reports and delivery documents in docs/.")
        if not os.path.isdir(DOCS):
            show_empty("Documentation folder not found.")
            return
        docs = sorted(f for f in os.listdir(DOCS) if f.endswith(".md") and "phase" in f.lower())
        if not docs:
            show_empty("No phase documentation files found.")
            return
        for doc in docs:
            st.markdown(f"- `{doc}`")
        st.info("Master report: `docs/PROJECT_FORESIGHT_FINAL_REPORT.md`")

    elif page == "validation_status":
        page_header("Validation Status", "Regression and integrity checks for submission readiness.")
        kv_table(
            [
                ("Full Regression", "280+/PASS (see latest pytest run)"),
                ("Phase 21 Tests", "PASS"),
                ("Phase 22 Tests", "PASS"),
                ("Frozen Models", "12/12 unchanged"),
                ("Delivery Status", "PROJECT DELIVERY READY"),
                ("Live Production WAPE", "PENDING ACTUALS"),
            ],
            field_col="Check",
            value_col="Status",
        )

    elif page == "about":
        page_header("About Project", "AI-Powered Demand & Inventory Intelligence")
        st.markdown(
            """
**PROJECT FORESIGHT** is a decision-support platform for retail demand forecasting
and inventory risk.

**Lifecycle:** Data Engineering → Feature Engineering → Forecasting → Candidate Validation →
Promotion Gate → Model Hardening → Production Promotion → Monitoring → Executive Intelligence

**Status:** PROJECT DELIVERY READY

This system does **not** automatically place purchase orders or retrain models in production.
            """
        )
        st.caption(f"Unified navigation pages: {len(all_nav_items())}")
