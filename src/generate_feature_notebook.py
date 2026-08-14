"""
Generate and execute notebooks/05_feature_engineering.ipynb (Phase 6).
"""

from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "05_feature_engineering.ipynb")


def md(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if not lines:
        lines = [""]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if not lines:
        lines = [""]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_notebook() -> dict:
    cells = []

    cells.append(md("""# Project FORESIGHT — Phase 6: Feature Engineering

**Input:** `data/processed/integrated/forecast_base.parquet` (CAM)  
**Output:** `data/processed/features/forecast_features.parquet`  
**Target:** `units_sold`  
**Grain:** `date + source_dataset + entity_id + product_key`  

Strict UCI / SYNTHETIC separation. Leakage-safe lags & rolling windows.  
**Do not proceed to Phase 7 until validation passes.**
"""))

    cells.append(md("## 1. Setup & schema validation"))

    cells.append(code("""import os, sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(".")
if os.path.basename(BASE_DIR) == "notebooks":
    BASE_DIR = os.path.abspath("..")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.feature_engineering import (
    load_features_input,
    validate_forecast_base_schema,
    run_feature_pipeline,
    write_feature_engineering_report,
    get_split_summary,
    get_ml_feature_compatibility,
)
from src.validate_features import run_validation
from src.feature_adapter import get_compatibility_summary

print("BASE_DIR:", BASE_DIR)
fb = load_features_input()
print("forecast_base shape:", fb.shape)
print("columns:", list(fb.columns))
print("sources:\\n", fb["source_dataset"].value_counts())
print("date range:", fb["date"].min(), "->", fb["date"].max())
print("Schema validation: PASS")
"""))

    cells.append(md("## 2. Run full Phase 6 feature pipeline"))

    cells.append(code("""df, meta = run_feature_pipeline(save=True)
print("output shape:", df.shape)
print("output columns:", list(df.columns))
print(get_split_summary(df).to_string(index=False))
"""))

    cells.append(md("## 3. Feature groups & sample rows"))

    cells.append(code("""calendar_cols = ["year","month","quarter","week_of_year","day_of_week","day_of_month","day_of_year","is_weekend"]
cyclical_cols = ["month_sin","month_cos","dow_sin","dow_cos"]
lag_cols = [c for c in df.columns if c.startswith("units_sold_lag_")]
rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
trend_cols = [c for c in df.columns if c.startswith("demand_")]
price_cols = ["average_unit_price","base_price","discount_pct","price_lag_1","price_change"]
promo_cols = ["promotion_flag","promotion_available","promo_rolling_7"]
product_cols = ["category","sub_category","brand"]
entity_cols = ["region","store_type","store_size_sqft"]
inv_cols = ["ending_inventory","on_order_qty","stockout_flag","historical_doi"]

for name, cols in [
    ("calendar", calendar_cols), ("cyclical", cyclical_cols), ("lag", lag_cols),
    ("rolling", rolling_cols), ("trend", trend_cols), ("price", price_cols),
    ("promo", promo_cols), ("product", product_cols), ("entity", entity_cols),
    ("inventory", inv_cols),
]:
    print(f"{name}: {len(cols)} -> {cols}")

print("\\nUCI sample:")
print(df[df.source_dataset=="UCI"].head(3).to_string())
print("SYNTHETIC sample:")
print(df[df.source_dataset=="SYNTHETIC"].head(3).to_string())
"""))

    cells.append(md("## 4. Missing-value strategy (no blind zero-fill)"))

    cells.append(code("""print("insufficient_history rate:", round(100*df['insufficient_history'].mean(), 2), "%")
lag_roll = [c for c in df.columns if "lag_" in c or c.startswith("rolling_") or c.startswith("demand_")]
miss = pd.DataFrame({
    "feature": lag_roll,
    "missing_pct": [round(100*df[c].isna().mean(), 2) for c in lag_roll],
}).sort_values("missing_pct", ascending=False)
print(miss.head(20).to_string(index=False))
print("Strategy: leave warm-up NaNs as NaN; flag with insufficient_history.")
"""))

    cells.append(md("## 5. Source separation checks"))

    cells.append(code("""uci = df[df.source_dataset=="UCI"]
syn = df[df.source_dataset=="SYNTHETIC"]
print("UCI rows:", len(uci), "entities:", uci.entity_id.nunique(), "products:", uci.product_key.nunique())
print("SYN rows:", len(syn), "entities:", syn.entity_id.nunique(), "products:", syn.product_key.nunique())
print("UCI promotion_flag all NaN:", bool(uci.promotion_flag.isna().all()))
print("UCI promotion_available all 0:", bool((uci.promotion_available==0).all()))
print("UCI category all NaN:", bool(uci.category.isna().all()))
print("UCI ending_inventory all NaN:", bool(uci.ending_inventory.isna().all()))
print("SYN ending_inventory non-null:", int(syn.ending_inventory.notna().sum()))
"""))

    cells.append(md("## 6. Leakage validation suite"))

    cells.append(code("""result = run_validation()
print("VALIDATION:", result.summary())
assert result.failed == 0, f"Validation failed: {result.failed} checks"
"""))

    cells.append(md("## 7. ML compatibility & final report"))

    cells.append(code("""compat = get_compatibility_summary()
print(compat)
report_path = write_feature_engineering_report(df, meta, result.summary())
print("Wrote report:", report_path)
print("Phase 6 COMPLETE — STOP before Phase 7.")
"""))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def main() -> int:
    nb = build_notebook()
    os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NOTEBOOK_PATH}")

    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    print(f"Executing {len(notebook.cells)} cells...")
    try:
        ep.preprocess(notebook, {"metadata": {"path": BASE_DIR}})
    except Exception as e:
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        print("NOTEBOOK EXECUTION ERROR:", e)
        return 1

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)
    print("Notebook executed with 0 errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
