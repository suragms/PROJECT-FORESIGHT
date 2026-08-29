"""Phase 23 — Monitoring pages (Phase 21 integration)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.data_loader import load_monitoring_json
from dashboard.components.status_cards import status_badge


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
        st.title("System Health")
        st.metric("Health Score", summary.get("health_score", "UNKNOWN"))
        cols = st.columns(3)
        for i, (k, v) in enumerate(components.items()):
            with cols[i % 3]:
                st.markdown(f"**{k.replace('_', ' ').title()}**")
                st.markdown(status_badge(v), unsafe_allow_html=True)
        st.warning("Forecast Performance: **PENDING ACTUALS**")

    elif page == "data_quality":
        st.title("Data Quality")
        report = load_monitoring_json("data_quality_report.json") or {}
        st.markdown(status_badge(report.get("overall_status", "N/A")), unsafe_allow_html=True)
        st.json(report)

    elif page == "data_drift":
        st.title("Data Drift")
        report = load_monitoring_json("data_drift_report.json") or {}
        st.markdown(status_badge(report.get("overall_status", "N/A")), unsafe_allow_html=True)
        if report.get("feature_drift"):
            df = pd.DataFrame([{"feature": k, **v} for k, v in report["feature_drift"].items()])
            st.dataframe(df, use_container_width=True)

    elif page == "prediction_drift":
        st.title("Prediction Drift")
        report = load_monitoring_json("prediction_drift_report.json") or {}
        st.markdown(status_badge(report.get("overall_status", "N/A")), unsafe_allow_html=True)
        st.json(report)

    elif page == "alerts":
        st.title("Alerts")
        alerts = summary.get("alerts", [])
        if alerts:
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)
        else:
            st.success("No active alerts")

    elif page == "integrity":
        st.title("Model Integrity")
        report = load_monitoring_json("model_integrity_report.json") or {}
        c1, c2 = st.columns(2)
        c1.metric("Frozen 12/12", "PASS" if report.get("frozen_12_unchanged") else "FAIL")
        c2.metric("Phase 20 Model", "PASS" if report.get("phase20_unchanged") else "FAIL")
        st.json(report)
