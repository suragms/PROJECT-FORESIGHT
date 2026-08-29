"""Phase 23 — Monitoring pages (Phase 21 integration)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.data_loader import load_monitoring_json
from dashboard.components.status_cards import status_badge
from dashboard.components.ui import kv_table, page_header, safe_dataframe, show_empty


def _require_summary():
    summary = load_monitoring_json("monitoring_summary.json")
    if not summary:
        st.warning("Monitoring not run. Execute: `python src/run_phase21.py`")
        st.stop()
    return summary


def render(page: str) -> None:
    summary = _require_summary()
    components = summary.get("components", {})

    if page == "system_health":
        page_header("System Health", "Aggregate monitoring health across quality, drift, and integrity.")
        st.metric("Health Score", summary.get("health_score", "UNKNOWN"))
        cols = st.columns(3)
        for i, (k, v) in enumerate(components.items()):
            with cols[i % 3]:
                st.markdown(f"**{k.replace('_', ' ').title()}**")
                st.markdown(status_badge(str(v)), unsafe_allow_html=True)
        st.warning("Forecast Performance: **PENDING ACTUALS**")

    elif page == "data_quality":
        page_header("Data Quality", "Schema and quality checks on monitoring inputs.")
        report = load_monitoring_json("data_quality_report.json") or {}
        st.markdown(status_badge(report.get("overall_status", "N/A")), unsafe_allow_html=True)
        flat = {k: v for k, v in report.items() if not isinstance(v, (dict, list))}
        if flat:
            kv_table(flat)
        else:
            show_empty("No summary quality fields available.")

    elif page == "data_drift":
        page_header("Data Drift", "Feature distribution drift versus reference window.")
        report = load_monitoring_json("data_drift_report.json") or {}
        st.markdown(status_badge(report.get("overall_status", "N/A")), unsafe_allow_html=True)
        if report.get("feature_drift"):
            df = pd.DataFrame([{"feature": k, **v} for k, v in report["feature_drift"].items()])
            safe_dataframe(df)
        else:
            show_empty("No feature drift rows available.")

    elif page == "prediction_drift":
        page_header("Prediction Drift", "Prediction distribution monitoring.")
        report = load_monitoring_json("prediction_drift_report.json") or {}
        st.markdown(status_badge(report.get("overall_status", "N/A")), unsafe_allow_html=True)
        flat = {k: v for k, v in report.items() if not isinstance(v, (dict, list))}
        if flat:
            kv_table(flat)
        else:
            show_empty("No prediction drift summary available.")

    elif page == "alerts":
        page_header("Alerts", "Structured monitoring alerts with severity and actions.")
        alerts = summary.get("alerts", [])
        if alerts:
            safe_dataframe(pd.DataFrame(alerts))
        else:
            st.success("No active alerts")

    elif page == "integrity":
        page_header("Model Integrity", "SHA-256 verification of frozen and Phase 20 models.")
        report = load_monitoring_json("model_integrity_report.json") or {}
        c1, c2 = st.columns(2)
        c1.metric("Frozen 12/12", "PASS" if report.get("frozen_12_unchanged") else "FAIL")
        c2.metric("Phase 20 Model", "PASS" if report.get("phase20_unchanged") else "FAIL")
        flat = {k: v for k, v in report.items() if not isinstance(v, (dict, list))}
        if flat:
            kv_table(flat)
