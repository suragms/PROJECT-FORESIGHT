"""
Read-only executive intelligence dashboard (Phase 15).

Consumes compact BI extracts. Does not retrain models.
Run: streamlit run dashboard/executive_intelligence.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.bi.exports import load_bi_tables  # noqa: E402
from src.bi.filters import apply_filters  # noqa: E402
from src.bi.quality import freshness  # noqa: E402


def _freshness_caption() -> None:
    snap = freshness()
    st.caption(
        f"Data as of: `{snap.get('data_as_of')}` · "
        f"Forecast generated: `{snap.get('forecast_generated')}` · "
        f"Monitoring snapshot: `{snap.get('monitoring_snapshot')}` · "
        f"**{snap.get('label')}**"
    )


def _kpi_value(kpis: pd.DataFrame, col: str):
    if kpis is None or kpis.empty or col not in kpis.columns:
        return "NOT AVAILABLE"
    val = kpis.iloc[0][col]
    if pd.isna(val):
        return "NOT AVAILABLE"
    return val


def main() -> None:
    st.set_page_config(page_title="FORESIGHT Executive Intelligence", layout="wide")
    st.title("FORESIGHT — Executive Intelligence")
    st.caption(
        "Decision-support BI over frozen Phase 11 forecasts and the 1000-row inventory "
        "reference extract. This dashboard does not train models, send purchase orders, "
        "or label file snapshots as live data."
    )
    _freshness_caption()

    tables = load_bi_tables()
    kpis = tables["executive_kpis.parquet"]
    demand = tables["product_demand.parquet"]
    perf = tables["forecast_performance.parquet"]
    risk = tables["inventory_risk.parquet"]
    recs = tables["recommendations.parquet"]
    health = tables["system_health.parquet"]
    season = tables.get("seasonality.parquet", pd.DataFrame())

    st.sidebar.header("Filters")
    datasets = ["All"] + sorted(perf["source_dataset"].astype(str).unique().tolist()) if "source_dataset" in perf.columns else ["All"]
    dataset = st.sidebar.selectbox("Dataset", datasets)
    dataset_val = None if dataset == "All" else dataset
    horizons = ["All"] + sorted(perf["horizon"].dropna().astype(int).unique().tolist()) if "horizon" in perf.columns else ["All"]
    horizon = st.sidebar.selectbox("Forecast Horizon", horizons)
    horizon_val = None if horizon == "All" else int(horizon)
    products = ["All"] + sorted(demand["sku_id"].astype(str).unique().tolist())[:500]
    product = st.sidebar.selectbox("Product", products)
    product_val = None if product == "All" else product
    entities = ["All"]
    if "store_id" in demand.columns:
        entities += sorted(demand["store_id"].astype(str).unique().tolist())
    entity = st.sidebar.selectbox("Entity", entities)
    entity_val = None if entity == "All" else entity
    risk_levels = ["All"]
    if "stockout_risk_level" in risk.columns:
        risk_levels += sorted(risk["stockout_risk_level"].astype(str).unique().tolist())
    risk_level = st.sidebar.selectbox("Risk Level", risk_levels)
    risk_val = None if risk_level == "All" else risk_level

    date_start = date_end = None
    if "forecast_date" in perf.columns and not perf.empty:
        dmin = pd.to_datetime(perf["forecast_date"]).min()
        dmax = pd.to_datetime(perf["forecast_date"]).max()
        picked = st.sidebar.date_input("Date Range", value=(dmin.date(), dmax.date()))
        if isinstance(picked, tuple) and len(picked) == 2:
            date_start, date_end = picked

    filt_kw = dict(
        dataset=dataset_val,
        entity=entity_val,
        product=product_val,
        date_start=date_start,
        date_end=date_end,
        horizon=horizon_val,
        risk_level=risk_val,
    )
    demand_f = apply_filters(demand, **{k: v for k, v in filt_kw.items() if k != "horizon"})
    perf_f = apply_filters(perf, **filt_kw)
    risk_f = apply_filters(risk, **{k: v for k, v in filt_kw.items() if k not in {"dataset", "horizon", "date_start", "date_end"}})
    recs_f = apply_filters(recs, **{k: v for k, v in filt_kw.items() if k in {"product", "entity"}})

    overview, demand_tab, forecast_tab, inv_tab, rec_tab, health_tab = st.tabs([
        "Executive Overview",
        "Demand Intelligence",
        "Forecast Intelligence",
        "Inventory Intelligence",
        "Recommendations",
        "System Health",
    ])

    with overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total recent units (extract)", _kpi_value(kpis, "demand_total_recent_units"))
        c2.metric("Forecast MAE (TEST)", _kpi_value(kpis, "forecast_mae"))
        c3.metric("Critical / high stockout", _kpi_value(kpis, "inventory_stockout_critical_high"))
        c4.metric("Reorder-review flags", _kpi_value(kpis, "inventory_reorder_review_count"))
        st.info(_kpi_value(kpis, "inventory_extract_note"))
        top = demand.sort_values("total_recent_revenue", ascending=False).head(10) if "total_recent_revenue" in demand.columns else demand.head(10)
        st.subheader("Top products (total_recent_revenue ranking)")
        st.dataframe(top, use_container_width=True)
        st.caption("Ranking uses the existing extract field `total_recent_revenue`. Bottom ranks are labelled LOW DEMAND, not bad product.")
        rec_counts = recs["recommended_review"].value_counts().reset_index() if "recommended_review" in recs.columns else pd.DataFrame()
        st.subheader("Recommendation summary")
        st.dataframe(rec_counts, use_container_width=True)

    with demand_tab:
        st.subheader("Historical vs forecast demand")
        if not perf_f.empty:
            plot = perf_f.groupby("forecast_date", as_index=False)[["actual", "forecast", "p10", "p90"]].mean()
            fig = px.line(plot, x="forecast_date", y=["actual", "forecast", "p10", "p90"],
                          labels={"value": "units", "variable": "series"})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Actual = held-out TEST. Forecast = point prediction. P10/P90 = interval companions, not observations.")
        if not season.empty:
            monthly = season[season["pattern"] == "monthly"]
            st.subheader("Seasonality (TEST actuals)")
            if not monthly.empty:
                st.dataframe(monthly, use_container_width=True)
            summary = season[season["pattern"] == "monthly_summary"]
            if not summary.empty:
                st.dataframe(summary, use_container_width=True)
                st.caption("weak_or_uncertain=True when monthly CV < 0.10. Patterns are not manufactured.")
        st.subheader("Growth (documented rule)")
        show = demand_f.copy()
        keep = [c for c in ["sku_id", "store_id", "total_recent_units", "avg_daily_demand",
                            "growth_class", "growth_rate", "forecast_mean_h1", "stockout_risk_level"] if c in show.columns]
        st.dataframe(show[keep].head(200), use_container_width=True)
        st.caption("Growing / Stable / Declining / Insufficient Evidence. Extract-only SKUs without a TEST split stay Insufficient Evidence.")
        st.subheader("LOW DEMAND products")
        low = demand[demand["demand_rank_label"] == "LOW DEMAND"] if "demand_rank_label" in demand.columns else pd.DataFrame()
        st.dataframe(low, use_container_width=True)

    with forecast_tab:
        st.subheader("Forecast vs actual")
        cols = [c for c in ["forecast_date", "source_dataset", "horizon", "actual", "forecast",
                            "error", "absolute_error", "p10", "p90"] if c in perf_f.columns]
        st.dataframe(perf_f[cols].head(500), use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", _kpi_value(kpis, "forecast_mae"))
        m2.metric("RMSE", _kpi_value(kpis, "forecast_rmse"))
        m3.metric("WAPE", _kpi_value(kpis, "forecast_wape"))
        m4.metric("Bias", _kpi_value(kpis, "forecast_bias"))
        st.caption("Aggregate metrics come from the monitoring snapshot on held-out TEST actuals. Unknown future actuals are not scored.")
        st.write("Interval coverage:", _kpi_value(kpis, "forecast_interval_coverage"))

    with inv_tab:
        st.warning("On-disk risk matrix is a 1000-row reference extract — not the operational inventory universe.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Critical stockout", _kpi_value(kpis, "inventory_stockout_critical_high"))
        c2.metric("Severe overstock", _kpi_value(kpis, "inventory_overstock_severe"))
        c3.metric("Moderate overstock", _kpi_value(kpis, "inventory_overstock_moderate"))
        c4.metric("Reorder review", _kpi_value(kpis, "inventory_reorder_review_count"))
        c5.metric("No exceptional risk", _kpi_value(kpis, "inventory_no_exceptional_risk"))
        st.subheader("Risk matrix (median split on extract)")
        if "risk_matrix_cell" in risk_f.columns:
            mat = risk_f.groupby(["demand_high", "inventory_high", "risk_matrix_cell"], as_index=False).size() if "demand_high" in risk_f.columns else risk_f["risk_matrix_cell"].value_counts().reset_index()
            st.dataframe(mat, use_container_width=True)
            st.caption(
                "Low demand + low inventory → Normal. High demand + low inventory → Stockout Review. "
                "Low demand + high inventory → Overstock Review. High + high → Critical Review. "
                "Split is **strict greater than** the extract median so a zero-inflated inventory median "
                "(0 on this extract) does not mark every row as High. This overlay does not replace "
                "`stockout_risk_level` / `overstock_risk_level`."
            )
        st.dataframe(risk_f.head(200), use_container_width=True)

    with rec_tab:
        st.subheader("Decision-support reviews")
        st.caption("Not purchase orders. Not autonomous procurement.")
        show = recs_f if recs_f is not None and not recs_f.empty else recs
        keep = [c for c in ["priority", "sku_id", "store_id", "recommended_review",
                            "evidence", "reason", "confidence_limitation", "autonomous_decision"] if c in show.columns]
        st.dataframe(show[keep].sort_values("priority") if "priority" in show.columns else show[keep], use_container_width=True)

    with health_tab:
        st.subheader("System health (file snapshot)")
        st.dataframe(health, use_container_width=True)
        st.json(freshness())
        st.caption("API health and readiness are served by GET /health and GET /ready. This page does not call them live unless you run the API separately.")

    st.sidebar.info("Filters change the view only. Frozen models are unchanged.")


if os.environ.get("EXECUTIVE_DASHBOARD_SKIP_MAIN") != "1":
    main()
