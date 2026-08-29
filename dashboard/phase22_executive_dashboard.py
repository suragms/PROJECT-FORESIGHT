"""
Phase 22 Executive Dashboard — Business + Executive view.
Run: streamlit run dashboard/phase22_executive_dashboard.py
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.phase22_executive_adapter import executive_bundle
from dashboard.components.theme import inject_theme

st.set_page_config(page_title="FORESIGHT | Executive Dashboard", layout="wide", page_icon="📊")
inject_theme()

st.title("PROJECT FORESIGHT — Executive Dashboard")

try:
    data = executive_bundle()
except Exception as ex:
    st.error(f"Failed to load executive data: {ex}")
    st.info("Ensure Phase 20 promotion and Phase 21 monitoring have been run.")
    st.stop()

# ── Section 1: Executive Overview ──────────────────────────────────────────
st.header("Executive Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("System Status", data["system_status"])
c2.metric("Monitoring", data["monitoring_status"])
c3.metric("Production Performance", data["production_performance"])
c4.metric("Health Score", data["monitoring"].get("health_score", "N/A"))

st.info(
    f"**Validation / Backtest metrics** (NOT live production performance): "
    f"Overall WAPE {data['validation_overall_wape']}% | "
    f"h1–h6 WAPE {data['validation_h16_wape']}%"
)

# ── Section 2: Forecast Overview ─────────────────────────────────────────
st.header("Forecast Overview")

forecasts = data["forecasts"]
if len(forecasts) == 0:
    st.warning("No forecast data available.")
else:
    skus = sorted(forecasts["product_key"].unique())
    selected_sku = st.selectbox("Select SKU", skus)
    sku_fc = forecasts[forecasts["product_key"] == selected_sku].sort_values("horizon")

    origin = sku_fc["forecast_origin"].iloc[0] if "forecast_origin" in sku_fc.columns else "N/A"
    st.caption(f"Forecast origin: {origin} | Source: SYNTHETIC | Grain: Weekly SKU-level")

    prod_fc = sku_fc[sku_fc["horizon"] <= 6]
    ext_fc = sku_fc[sku_fc["horizon"] > 6]

    fig = px.bar(
        prod_fc, x="horizon", y="forecast_demand",
        title="6-Week Production Forecast (Validated Horizon)",
        labels={"horizon": "Week Ahead", "forecast_demand": "Forecast Demand"},
        color_discrete_sequence=["#ff4b4b"],
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(ext_fc) > 0:
        st.warning("Weeks 7–8 shown below are **EXTENDED / PARTIAL ACCURACY** — not equal to h1–h6.")
        st.dataframe(
            ext_fc[["horizon", "forecast_demand", "forecast_status"]].rename(
                columns={"forecast_status": "Status"}
            ),
            use_container_width=True,
        )

    st.subheader("Forecast Summary")
    summary = prod_fc.groupby("horizon")["forecast_demand"].sum().reset_index()
    summary.columns = ["Horizon", "Total Forecast Demand"]
    st.dataframe(summary, use_container_width=True)

# ── Section 3: Inventory Risk ─────────────────────────────────────────────
st.header("Inventory Risk")

risk = data["risk"]
if len(risk) == 0:
    st.warning("No risk data available.")
else:
    rc1, rc2, rc3, rc4 = st.columns(4)
    actions = risk["recommended_action"].value_counts()
    rc1.metric("REORDER NOW", int(actions.get("REORDER NOW", 0)))
    rc2.metric("WATCH / VOLATILE", int(actions.get("WATCH / VOLATILE", 0)))
    rc3.metric("HEALTHY", int(actions.get("HEALTHY", 0)))
    rc4.metric("MARKDOWN / CLEAR", int(actions.get("MARKDOWN / CLEAR", 0)))

    fig_risk = px.pie(
        risk, names="recommended_action", title="Risk Action Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    if selected_sku and len(risk) > 0:
        sku_id = str(selected_sku).replace("SYN_", "")
        sku_risk = risk[risk["sku_id"] == sku_id]
        if len(sku_risk) > 0:
            r = sku_risk.iloc[0]
            st.subheader(f"Risk Detail — {selected_sku}")
            st.markdown(f"""
| Field | Value |
|-------|-------|
| Stockout Risk | {r.get('stockout_risk_level', 'N/A')} |
| Overstock Risk | {r.get('overstock_risk_level', 'N/A')} |
| Recommended Action | **{r.get('recommended_action', 'N/A')}** |
| Weeks of Supply | {r.get('weeks_of_supply', 'N/A')} |
| Projected Balance | {r.get('projected_balance', 'N/A')} |
            """)

# ── Section 4: Business Impact ────────────────────────────────────────────
st.header("Business Impact")

bi = data["business_impact"]
if bi.get("status") == "AVAILABLE":
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Sales at Risk", bi.get("total_sales_at_risk", "N/A"))
    bc2.metric("Locked Capital", bi.get("total_locked_capital", "N/A"))
    bc3.metric("At-Risk SKUs", bi.get("at_risk_skus", "N/A"))
    bc4.metric("High Overstock SKUs", bi.get("high_overstock_skus", "N/A"))
else:
    st.write("NOT AVAILABLE")

# ── Section 5: Model Health ─────────────────────────────────────────────────
st.header("Model Health (Phase 21 Monitoring)")

components = data["monitoring_components"]
if components:
    mc1, mc2, mc3, mc4 = st.columns(4)
    items = list(components.items())
    for i, (k, v) in enumerate(items[:4]):
        [mc1, mc2, mc3, mc4][i].metric(k.replace("_", " ").title(), v)
    if len(items) > 4:
        mc5, mc6, mc7 = st.columns(3)
        for i, (k, v) in enumerate(items[4:7]):
            [mc5, mc6, mc7][i].metric(k.replace("_", " ").title(), v)
else:
    st.warning("Monitoring not run. Execute: `python src/run_phase21.py`")

st.warning("Forecast Performance: **PENDING ACTUALS** — live production WAPE not yet measured.")

alerts = data.get("alerts", [])
if alerts:
    st.subheader("Recent Alerts")
    st.dataframe(pd.DataFrame(alerts), use_container_width=True)

# ── Section 6: Model Information ────────────────────────────────────────────
st.header("Model Information")

info = data["model_info"]
st.markdown(f"""
| Field | Value |
|-------|-------|
| **Model ID** | {info.get('model_id', 'phase20_synthetic_lightgbm')} |
| **Source Dataset** | SYNTHETIC |
| **Forecast Grain** | Weekly SKU-level |
| **Feature Count** | {data.get('feature_count', 45)} |
| **Supported Horizon** | 6 Weeks |
| **Extended Horizon** | Weeks 7–8 (PARTIAL) |
| **Validation Method** | Rolling-origin backtest |
| **Overall Validation WAPE** | {data['validation_overall_wape']}% |
| **h1–h6 Validation WAPE** | {data['validation_h16_wape']}% |
| **Known Limitations** | {info.get('known_limitation', 'N/A')} |
""")

st.caption(
    "Decision-support system only. Validation metrics are backtest results, not guaranteed live performance. "
    "UCI dataset remains a RESEARCH CANDIDATE."
)
