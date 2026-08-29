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
from dashboard.components.theme import inject_theme
from dashboard.components.ui import kv_table, page_header, safe_dataframe, show_empty
inject_theme()
page_header(
    "Phase 21 Production Monitoring",
    "Observability for quality, drift, forecast reference metrics, and model integrity.",
)

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
    flat = {k: v for k, v in dq.items() if not isinstance(v, (dict, list))}
    if flat:
        kv_table(flat)
    else:
        show_empty("No data quality summary fields.")
else:
    show_empty("Data quality report not available.")

st.subheader("Feature Quality")
fq = load_json("feature_quality_report.json")
if fq:
    kv_table(
        [
            ("Expected Features", fq.get("expected_feature_count", "N/A")),
            ("Actual Features", fq.get("actual_feature_count", "N/A")),
            ("Overall Status", fq.get("overall_status", "N/A")),
            ("Missing Features", fq.get("missing_features") or "None"),
            ("Unexpected Features", fq.get("unexpected_features") or "None"),
        ]
    )
else:
    show_empty("Feature quality report not available.")

st.subheader("Data Drift")
dd = load_json("data_drift_report.json")
if dd:
    st.write(f"Overall: {dd.get('overall_status')} | Demand PSI: {dd.get('demand_psi')} ({dd.get('demand_drift_status')})")
    drift_df = pd.DataFrame([
        {"feature": k, **v} for k, v in dd.get("feature_drift", {}).items()
    ])
    safe_dataframe(drift_df if len(drift_df) else None, empty_message="No feature drift rows available.")
else:
    show_empty("Data drift report not available.")

st.subheader("Prediction Drift")
pd_drift = load_json("prediction_drift_report.json")
if pd_drift:
    flat = {k: v for k, v in pd_drift.items() if not isinstance(v, (dict, list))}
    if flat:
        kv_table(flat)
    else:
        show_empty("No prediction drift summary fields.")
else:
    show_empty("Prediction drift report not available.")

st.subheader("Forecast Performance")
fp = load_json("forecast_performance_report.json")
if fp:
    st.info(fp.get("data_source", "Validation reference only — production actuals PENDING"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Validation WAPE %", fp.get("overall_wape_pct", "N/A"))
    c2.metric("H1-H6 WAPE %", fp.get("h1_h6_wape_pct", "N/A"))
    c3.metric("Live Performance", "PENDING ACTUALS")
    if fp.get("horizon_performance"):
        safe_dataframe(pd.DataFrame(fp["horizon_performance"]))
else:
    show_empty("Forecast performance report not available.")

st.subheader("Horizon Performance")
if fp and fp.get("horizon_performance"):
    hdf = pd.DataFrame(fp["horizon_performance"])
    prod = hdf[hdf["label"] == "PRODUCTION"] if "label" in hdf.columns else hdf
    safe_dataframe(prod)

st.subheader("Holiday Monitoring")
holiday = summary.get("holiday_monitoring", {})
if holiday:
    flat = {k: v for k, v in holiday.items() if not isinstance(v, (dict, list))}
    if flat:
        kv_table(flat)
    else:
        show_empty("Holiday monitoring details are nested only.")
else:
    show_empty("Holiday monitoring not available.")

st.subheader("Risk Consistency")
risk = load_json("risk_consistency_report.json")
if risk:
    st.write(f"Consistency: {risk.get('consistency_status')} | Distribution: {risk.get('distribution_status')}")
    if risk.get("action_distribution"):
        st.bar_chart(pd.Series(risk["action_distribution"]))
    else:
        show_empty("No risk action distribution available.")
else:
    show_empty("Risk consistency report not available.")

st.subheader("Model Integrity")
integrity = load_json("model_integrity_report.json")
if integrity:
    c1, c2 = st.columns(2)
    c1.metric("Frozen 12/12", "PASS" if integrity.get("frozen_12_unchanged") else "FAIL")
    c2.metric("Phase 20 Model", "PASS" if integrity.get("phase20_unchanged") else "FAIL")
else:
    show_empty("Model integrity report not available.")

st.subheader("Alerts")
alerts = summary.get("alerts", [])
if alerts:
    safe_dataframe(pd.DataFrame(alerts))
else:
    st.success("No active alerts")

st.caption("Phase 21 observes only — does not retrain or modify production artifacts.")
