"""
Generate and execute notebooks/06_baseline_forecasting.ipynb (Phase 7).
"""

from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "06_baseline_forecasting.ipynb")


def md(source: str) -> dict:
    lines = source.splitlines(keepends=True) or [""]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(source: str) -> dict:
    lines = source.splitlines(keepends=True) or [""]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_notebook() -> dict:
    cells = []

    cells.append(md("""# Project FORESIGHT — Phase 7: Baseline Demand Forecasting

**Input:** `data/processed/features/forecast_features.parquet` (Phase 6)  
**Target:** `units_sold`  
**Grain:** `date + source_dataset + entity_id + product_key`  

Baselines only: Naive, Seasonal Naive, Moving Average (7/14/30), Historical Mean.  
**No ML models in this phase.**
"""))

    cells.append(md("## 1–2. Load Phase 6 Features"))
    cells.append(code("""import os, sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(".")
if os.path.basename(BASE_DIR) == "notebooks":
    BASE_DIR = os.path.abspath("..")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.baseline_forecasting import (
    load_features, create_time_split, confirm_seasonal_period,
    generate_all_predictions, evaluate_baselines, best_baselines,
    high_value_sku_analysis, save_baseline_results, create_forecast_charts,
    create_comparison_chart, write_baseline_report, SEASONAL_PERIOD, MODEL_COLS,
)
from src.validate_baselines import run_validation

df = load_features()
print("rows", len(df), "cols", df.shape[1])
print("date", df.date.min().date(), "->", df.date.max().date())
print("sources", df.source_dataset.value_counts().to_dict())
print("target nulls", int(df.units_sold.isna().sum()))
print("columns sample:", list(df.columns)[:15], "...")
"""))

    cells.append(md("## 3–4. Validation of grain & target"))
    cells.append(code("""assert set(["date","source_dataset","entity_id","product_key","units_sold","split"]).issubset(df.columns)
assert df.duplicated(["date","source_dataset","entity_id","product_key"]).sum() == 0
print("Forecast grain OK; target=units_sold; duplicate keys=0")
print("UCI entities", df[df.source_dataset=="UCI"].entity_id.nunique(),
      "SYN entities", df[df.source_dataset=="SYNTHETIC"].entity_id.nunique())
"""))

    cells.append(md("## 5. Target Analysis"))
    cells.append(code("""for src in sorted(df.source_dataset.unique()):
    sub = df[df.source_dataset==src]
    print(src, "mean", round(sub.units_sold.mean(),3),
          "median", round(sub.units_sold.median(),3),
          "zero_pct", round(100*(sub.units_sold==0).mean(),2),
          "p95", round(sub.units_sold.quantile(0.95),2))
"""))

    cells.append(md("## 6. Chronological Split"))
    cells.append(code("""split_summary = create_time_split(df)
print(split_summary.to_string(index=False))
for src in sorted(df.source_dataset.unique()):
    for sp in ["train","validation","test"]:
        sub = df[(df.source_dataset==src)&(df.split==sp)]
        print(f"{src}/{sp}: {sub.date.min().date()} -> {sub.date.max().date()} rows={len(sub):,}")
"""))

    cells.append(md("## 7–10. Generate Baselines (Naive / Seasonal Naive / MA / Historical Mean)"))
    cells.append(code("""seasonality = confirm_seasonal_period(df)
print("Seasonal period:", seasonality["selected_period"])
print("Rationale:", seasonality["rationale"])
print("DOW CV:", {k: seasonality[k] for k in seasonality if k.endswith("_dow_cv")})

pred_df = generate_all_predictions(df)
print("Prediction columns:", list(MODEL_COLS.values()))
print(pred_df[["date","source_dataset","entity_id","product_key","units_sold","split"] + list(MODEL_COLS.values())].head(3).to_string(index=False))
"""))

    cells.append(md("## 11–15. Metric Evaluation (UCI / Synthetic / Product / Store)"))
    cells.append(code("""tables = evaluate_baselines(pred_df)
best = best_baselines(tables["comparison"])
best_models = {s: i["model"] for s,i in best.items()}
tables["high_value"] = high_value_sku_analysis(df, tables["by_product"], best_models)

print("=== TEST metrics by source ===")
test_m = tables["by_source"][tables["by_source"].split=="test"].sort_values(["source_dataset","WAPE"])
print(test_m[["source_dataset","model","MAE","RMSE","MAPE","sMAPE","WAPE","n"]].to_string(index=False))

print("\\n=== Best baselines ===")
for src, info in best.items():
    print(src, info)

print("\\n=== High-value SKU analysis ===")
print(tables["high_value"].to_string(index=False))

print("\\n=== Synthetic store WAPE (best model) ===")
be = tables["by_entity"]
syn_model = best["SYNTHETIC"]["model"]
print(be[(be.source_dataset=="SYNTHETIC")&(be.model==syn_model)][["entity_id","MAE","RMSE","WAPE"]].sort_values("WAPE").to_string(index=False))
"""))

    cells.append(md("## 16–17. Visualizations & Comparison"))
    cells.append(code("""paths = save_baseline_results(pred_df, tables)
chart_paths = create_forecast_charts(pred_df)
chart_paths.append(create_comparison_chart(tables["comparison"]))
print("Saved files:")
for k,v in paths.items():
    print(" ", k, "->", v)
print("Charts:")
for p in chart_paths:
    print(" ", p)
print("\\nComparison (TEST):")
print(tables["comparison"][["source_dataset","rank","model","MAE","RMSE","sMAPE","WAPE"]].to_string(index=False))
"""))

    cells.append(md("## 18–19. Business Interpretation & Phase 8 Recommendations"))
    cells.append(code("""for src, info in best.items():
    print(f"OBSERVATION: Best {src} baseline is {info['model']}.")
    print(f"EVIDENCE: TEST WAPE={info['WAPE']:.4f}, MAE={info['MAE']:.4f}, RMSE={info['RMSE']:.4f}, sMAPE={info['sMAPE']:.4f}")
    print("BUSINESS INTERPRETATION: Source-specific demand regimes require separate benchmarks.")
    print(f"IMPLICATION FOR ML: Phase 8 must beat {src} WAPE={info['WAPE']:.4f} on the same TEST window.\\n")
"""))

    cells.append(md("## 20. Validation"))
    cells.append(code("""result = run_validation()
print("VALIDATION:", result.summary())
assert result.failed == 0, f"Baseline validation failed: {result.failed}"
report_path = write_baseline_report(
    df, split_summary, seasonality, tables, best, chart_paths,
    validation_summary=result.summary(), paths=paths,
)
print("Report:", report_path)
print("Phase 7 COMPLETE — STOP before Phase 8.")
"""))

    cells.append(md("## 21. Summary"))
    cells.append(code("""print("Input rows:", len(df))
print("Models:", list(MODEL_COLS))
print("Seasonal period:", SEASONAL_PERIOD)
print("Best:", best)
print("Validation:", result.summary())
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
    from nbformat.validator import normalize
    from nbconvert.preprocessors import ExecutePreprocessor

    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    normalize(notebook)

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

    errs = 0
    for i, c in enumerate(notebook.cells):
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errs += 1
                print("ERR cell", i, o.get("ename"), o.get("evalue"))
    print(f"Notebook executed. code-cell errors={errs}")
    return 0 if errs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
