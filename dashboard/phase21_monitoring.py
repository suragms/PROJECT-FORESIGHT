"""
Phase 21 Monitoring Dashboard
Run: streamlit run dashboard/phase21_monitoring.py
"""

import json
import os
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

P21_MON = os.path.join(BASE_DIR, "data", "phase21", "monitoring")


def load_json(name):
    path = os.path.join(P21_MON, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


st.set_page_config(page_title="FORESIGHT | Phase 21 Monitoring", layout="wide")
st.title("FORESIGHT — Phase 21 Production Monitoring")

summary = load_json("monitoring_summary.json")
if not summary:
    st.warning("Monitoring not yet run. Execute: `python src/run_phase21.py`")
    st.stop()

st.subheader("System Health")
health = summary.get("health_score", "UNKNOWN")
st.metric("Health Score", health)

cols = st.columns(4)
components = summary.get("components", {})
for i, (k, v) in enumerate(components.items()):
    cols[i % 4].metric(k.replace("_", " ").title(), v)

st.subheader("Data Quality")
dq = load_json("data_quality_report.json")
if dq:
    st.json({k: v for k, v in dq.items() if k != "null_rates"})

st.subheader("Feature Quality")
fq = load_json("feature_quality_report.json")
if fq:
    st.write(f"Features: {fq.get('actual_feature_count')}/{fq.get('expected_feature_count')} — {fq.get('overall_status')}")
    if fq.get("missing_features"):
        st.error(f"Missing: {fq['missing_features']}")
    if fq.get("unexpected_features"):
        st.warning(f"Unexpected: {fq['unexpected_features']}")

st.subheader("Data Drift")
dd = load_json("data_drift_report.json")
if dd:
    st.write(f"Overall: {dd.get('overall_status')} | Demand PSI: {dd.get('demand_psi')} ({dd.get('demand_drift_status')})")
    drift_df = pd.DataFrame([
        {"feature": k, **v} for k, v in dd.get("feature_drift", {}).items()
    ])
    if len(drift_df):
        st.dataframe(drift_df, use_container_width=True)

st.subheader("Prediction Drift")
pd_drift = load_json("prediction_drift_report.json")
if pd_drift:
    st.json(pd_drift)

st.subheader("Forecast Performance")
fp = load_json("forecast_performance_report.json")
if fp:
    st.info(fp.get("data_source", "Validation reference only — production actuals PENDING"))
    st.metric("Validation WAPE %", fp.get("overall_wape_pct", "N/A"))
    st.metric("H1-H6 WAPE %", fp.get("h1_h6_wape_pct", "N/A"))
    if fp.get("horizon_performance"):
        st.dataframe(pd.DataFrame(fp["horizon_performance"]), use_container_width=True)

st.subheader("Horizon Performance")
if fp and fp.get("horizon_performance"):
    hdf = pd.DataFrame(fp["horizon_performance"])
    prod = hdf[hdf["label"] == "PRODUCTION"]
    st.dataframe(prod, use_container_width=True)

st.subheader("Holiday Monitoring")
holiday = summary.get("holiday_monitoring", {})
st.json(holiday)

st.subheader("Risk Distribution")
risk = load_json("risk_consistency_report.json")
if risk:
    st.write(f"Consistency: {risk.get('consistency_status')} | Distribution: {risk.get('distribution_status')}")
    if risk.get("action_distribution"):
        st.bar_chart(pd.Series(risk["action_distribution"]))

st.subheader("Model Integrity")
integrity = load_json("model_integrity_report.json")
if integrity:
    c1, c2 = st.columns(2)
    c1.metric("Frozen 12/12", "PASS" if integrity.get("frozen_12_unchanged") else "FAIL")
    c2.metric("Phase 20 Model", "PASS" if integrity.get("phase20_unchanged") else "FAIL")

st.subheader("Recent Alerts")
alerts = summary.get("alerts", [])
if alerts:
    st.dataframe(pd.DataFrame(alerts), use_container_width=True)
else:
    st.success("No active alerts")

st.caption("Phase 21 observes only — does not retrain or modify production artifacts.")
