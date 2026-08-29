"""Phase 23 — Machine learning pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.data_loader import load_feature_contract, load_forecast_performance
from dashboard.components.status_cards import validation_disclaimer
from dashboard.components.ui import kv_table, page_header, safe_dataframe, show_empty


def render(page: str) -> None:
    contract = load_feature_contract()
    perf = load_forecast_performance() or {}

    if page == "model_overview":
        page_header(
            "Model Overview",
            "Production LightGBM weekly SKU forecaster (Phase 20).",
        )
        validation_disclaimer()
        kv_table(
            [
                ("Model", "phase20_synthetic_lightgbm"),
                ("Type", "LightGBM"),
                ("Dataset", "Synthetic Retail"),
                ("Grain", "Weekly SKU-level"),
                ("Features", str(contract.get("feature_count", 45))),
                ("Horizon", "6 Weeks"),
                ("Overall Validation WAPE", "13.96%"),
                ("Validated h1–h6 WAPE", "11.03%"),
                ("Live Performance", "PENDING ACTUALS"),
            ]
        )
        st.warning(
            "Known limitations: holiday bias partially unresolved; "
            "h7–h8 partial accuracy; UCI remains research-only."
        )

    elif page == "feature_contract":
        page_header("Feature Contract", "45-feature production contract with leakage audit status.")
        st.metric("Feature Count", contract.get("feature_count", 45))
        features = pd.DataFrame(contract.get("features", []))
        if len(features):
            cols = [c for c in ["feature_name", "dtype", "required", "leakage_status"] if c in features.columns]
            safe_dataframe(features[cols])
        else:
            show_empty("Feature contract not loaded.")

    elif page == "model_performance":
        page_header("Model Performance", "Validation / backtest reference — not live production WAPE.")
        validation_disclaimer()
        if perf:
            kv_table(
                [
                    ("Overall Validation WAPE %", perf.get("overall_wape_pct", "N/A")),
                    ("h1–h6 Validation WAPE %", perf.get("h1_h6_wape_pct", "N/A")),
                    ("Production Actuals", perf.get("production_actuals_status", "PENDING ACTUALS")),
                    ("Data Source", perf.get("data_source", "N/A")),
                ]
            )
            if perf.get("horizon_performance"):
                st.subheader("Horizon Performance")
                safe_dataframe(pd.DataFrame(perf["horizon_performance"]))
        else:
            st.info("Validation baselines: Overall WAPE 13.96%, h1–h6 WAPE 11.03%")

    elif page == "model_explainability":
        page_header("Model Explainability", "SHAP explainability is not shipped in this repository.")
        st.info(
            "SHAP-based production explainability is **not available**. "
            "Use Feature Contract and leakage audit evidence instead."
        )
        passed = sum(1 for f in contract.get("features", []) if f.get("leakage_status") == "PASS")
        st.metric("Features Passing Leakage Audit", f"{passed}/{contract.get('feature_count', 45)}")
