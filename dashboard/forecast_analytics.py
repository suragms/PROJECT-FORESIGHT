"""
Read-only forecast analytics dashboard (Phase 12).

Does not retrain models and does not write to source datasets.
Run: streamlit run dashboard/forecast_analytics.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (  # noqa: E402
    FINAL_FORECASTS_PATH,
    PHASE11_META_PATH,
    PROJECT_ROOT,
    REGISTRY_PATH,
)

FINAL_DIR = PROJECT_ROOT / "data" / "processed" / "forecasts" / "final"
PHASE10_DIR = PROJECT_ROOT / "data" / "processed" / "forecasts" / "phase10"
MONITOR_DIR = PROJECT_ROOT / "outputs" / "monitoring"

st.set_page_config(page_title="FORESIGHT Forecast Analytics", layout="wide")


@st.cache_data(show_spinner=False)
def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_parquet(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def main() -> None:
    st.title("FORESIGHT — Forecast Analytics")
    st.caption("Read-only Phase 11/12 dashboard. Models are not retrained.")

    meta = load_json(str(PHASE11_META_PATH)) or {}
    registry = load_json(str(REGISTRY_PATH)) or []
    preds = load_parquet(str(FINAL_FORECASTS_PATH))
    if preds is None:
        st.error("Final forecasts not found. Run Phase 11 first.")
        return
    preds["forecast_date"] = pd.to_datetime(preds["forecast_date"])

    datasets = sorted(preds["source_dataset"].astype(str).unique())
    horizons = sorted(preds["horizon"].astype(int).unique())
    col_a, col_b = st.sidebar.columns(2)
    dataset = st.sidebar.selectbox("Dataset", datasets)
    horizon = st.sidebar.selectbox("Horizon", horizons, index=0)
    view = preds[(preds["source_dataset"] == dataset) & (preds["horizon"] == int(horizon))].copy()
    if view.empty:
        st.warning("No rows for this dataset/horizon combination.")
        return

    selected = [r for r in registry if r.get("status") == "selected"
                and r.get("dataset") == dataset and int(r.get("horizon")) == int(horizon)]
    model = selected[0] if selected else {}

    st.header("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected model", model.get("model_id", "n/a"))
    c2.metric("Version", str(model.get("code_version", meta.get("code_version", "")))[:10])
    c3.metric("Dataset / h", f"{dataset} / {horizon}")
    gen = str(view["generated_at"].iloc[0]) if "generated_at" in view.columns else meta.get("executed_at_utc", "")
    c4.metric("Generated", str(gen)[:19])

    st.header("Forecast summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total forecasts", f"{len(view):,}")
    s2.metric("Mean prediction", f"{view['prediction'].mean():.3f}")
    s3.metric("Min", f"{view['prediction'].min():.3f}")
    s4.metric("Max", f"{view['prediction'].max():.3f}")
    s5.metric("Zero-prediction %", f"{100 * (view['prediction'] == 0).mean():.2f}")

    st.header("Forecast by date")
    daily = view.groupby("forecast_date", as_index=False).agg(
        prediction=("prediction", "mean"),
        actual=("actual", "mean") if "actual" in view.columns else ("prediction", "mean"),
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["forecast_date"], y=daily["prediction"], name="Predicted", mode="lines"))
    if "actual" in view.columns and view["actual"].notna().any():
        fig.add_trace(go.Scatter(x=daily["forecast_date"], y=daily["actual"], name="Actual", mode="lines"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.header("Horizon analysis")
    hz = load_parquet(str(FINAL_DIR / "analysis_horizon.parquet"))
    if hz is not None:
        sub = hz[hz["dataset"] == dataset] if "dataset" in hz.columns else hz
        st.dataframe(sub, use_container_width=True)
        if "WAPE" in sub.columns and "horizon" in sub.columns:
            st.plotly_chart(
                px.line(sub, x="horizon", y="WAPE", markers=True, title=f"{dataset} WAPE by horizon"),
                use_container_width=True,
            )

    st.header("Store / entity analysis")
    store = load_parquet(str(FINAL_DIR / "analysis_store.parquet"))
    if store is not None:
        ss = store[store["dataset"] == dataset].sort_values("WAPE") if "dataset" in store.columns else store
        if ss.empty:
            st.info("No entity metrics for this dataset.")
        else:
            best = ss.head(5)
            worst = ss.tail(5)
            c1, c2 = st.columns(2)
            c1.subheader("Best WAPE")
            c1.dataframe(best, use_container_width=True)
            c2.subheader("Worst WAPE")
            c2.dataframe(worst, use_container_width=True)

    st.header("Zero-demand analysis")
    if dataset == "SYNTHETIC":
        zf = load_parquet(str(FINAL_DIR / "analysis_zero_final.parquet"))
        zp = load_parquet(str(FINAL_DIR / "analysis_zero_phase8.parquet"))
        if zf is not None:
            st.dataframe(zf, use_container_width=True)
        if zp is not None:
            st.caption("Phase 8 comparison")
            st.dataframe(zp, use_container_width=True)
        st.plotly_chart(
            px.histogram(view, x="prediction", nbins=40, title="SYNTHETIC prediction distribution"),
            use_container_width=True,
        )
    else:
        st.info("UCI TEST grain has no coded zeros; zero-demand analysis is SYNTHETIC-only.")

    st.header("Prediction intervals")
    if view["lower_bound"].notna().any() and view["upper_bound"].notna().any():
        sample = view.sort_values("forecast_date").head(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample["forecast_date"], y=sample["upper_bound"], name="P90", line=dict(width=0)))
        fig.add_trace(go.Scatter(
            x=sample["forecast_date"], y=sample["lower_bound"], name="P10",
            fill="tonexty", fillcolor="rgba(29,78,216,0.15)", line=dict(width=0),
        ))
        fig.add_trace(go.Scatter(x=sample["forecast_date"], y=sample["prediction"], name="Point / P50 companion"))
        fig.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        iv = load_parquet(str(FINAL_DIR / "analysis_intervals.parquet"))
        if iv is not None:
            st.dataframe(iv, use_container_width=True)
    else:
        st.info("Intervals are attached on h=1 only.")

    st.header("Model comparison")
    cand = load_parquet(str(FINAL_DIR / "candidate_matrix.parquet"))
    sel = load_parquet(str(FINAL_DIR / "selection_table.parquet"))
    if cand is not None:
        show = cand[cand["dataset"] == dataset].copy()
        if "eligible" in show.columns:
            show["label"] = show["eligible"].map({True: "candidate", False: "rejected"})
        st.dataframe(show, use_container_width=True)
        st.caption("Rejected Phase 10 experiments are labelled. Final selection is in the table below.")
    if sel is not None:
        st.subheader("Phase 11 selected")
        st.dataframe(sel[sel["dataset"] == dataset] if "dataset" in sel.columns else sel)

    mon = load_json(str(MONITOR_DIR / "monitoring_summary.json"))
    if mon:
        st.header("Monitoring snapshot")
        st.json(mon)

    st.sidebar.info("This dashboard never writes to data/ or models/.")


if os.environ.get("FORECAST_DASHBOARD_SKIP_MAIN") != "1":
    main()
