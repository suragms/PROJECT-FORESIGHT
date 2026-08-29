"""Phase 23 — Machine learning pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.data_loader import load_feature_contract, load_forecast_performance
from dashboard.components.status_cards import validation_disclaimer


def render(page: str) -> None:
    contract = load_feature_contract()
    perf = load_forecast_performance() or {}

    if page == "model_overview":
        st.title("Model Overview")
        validation_disclaimer()
        st.markdown("""
| Field | Value |
|-------|-------|
| Model | phase20_synthetic_lightgbm |
| Type | LightGBM |
| Dataset | Synthetic Retail |
| Grain | Weekly SKU-level |
| Features | 45 |
| Horizon | 6 Weeks |
| Live Performance | PENDING ACTUALS |
        """)
        st.warning("Known limitations: holiday bias partially unresolved; h7–h8 partial accuracy; UCI research only.")

    elif page == "feature_contract":
        st.title("Feature Contract")
        st.metric("Feature Count", contract.get("feature_count", 45))
        features = pd.DataFrame(contract.get("features", []))
        if len(features):
            st.dataframe(features[["feature_name", "dtype", "required", "leakage_status"]], use_container_width=True)

    elif page == "model_performance":
        st.title("Model Performance")
        validation_disclaimer()
        if perf:
            st.json({k: v for k, v in perf.items() if k != "horizon_performance"})
        else:
            st.info("Validation baselines: Overall WAPE 13.96%, h1–h6 WAPE 11.03%")

    elif page == "model_explainability":
        st.title("Model Explainability")
        st.info(
            "SHAP-based production explainability is **NOT AVAILABLE** in the current repository. "
            "Feature contract and leakage audit evidence are available under Feature Contract."
        )
        passed = sum(1 for f in contract.get("features", []) if f.get("leakage_status") == "PASS")
        st.metric("Features Passing Leakage Audit", f"{passed}/{contract.get('feature_count', 45)}")
