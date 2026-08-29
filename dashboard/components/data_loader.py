"""Phase 23 — Cached data loaders (reuses Phase 20–22 adapters)."""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P21_MON = os.path.join(BASE, "data", "phase21", "monitoring")
P19_FEAT = os.path.join(BASE, "data", "phase19", "features", "synthetic_weekly_features.parquet")
DOCS = os.path.join(BASE, "docs")


@st.cache_data(show_spinner=False)
def load_executive_data() -> dict:
    from src.phase22_executive_adapter import executive_bundle
    return executive_bundle()


@st.cache_data(show_spinner=False)
def load_production_bundle() -> dict:
    from src.phase20_dashboard_adapter import dashboard_bundle
    return dashboard_bundle()


@st.cache_data(show_spinner=False)
def load_features_df() -> pd.DataFrame:
    if not os.path.exists(P19_FEAT):
        return pd.DataFrame()
    df = pd.read_parquet(P19_FEAT)
    df["week"] = pd.to_datetime(df["week"])
    return df


@st.cache_data(show_spinner=False)
def load_monitoring_json(name: str) -> dict | None:
    path = os.path.join(P21_MON, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_contract() -> dict:
    path = os.path.join(DOCS, "phase20_feature_contract.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_forecast_performance() -> dict | None:
    return load_monitoring_json("forecast_performance_report.json")
