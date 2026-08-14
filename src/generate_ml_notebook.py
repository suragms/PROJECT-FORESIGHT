"""
Generate and execute notebooks/07_ml_forecasting.ipynb (Phase 8).
"""

from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "07_ml_forecasting.ipynb")


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
    cells.append(md("""# Project FORESIGHT — Phase 8: ML Demand Forecasting

**Input:** Phase 6 `forecast_features.parquet`  
**Benchmarks (Phase 7 TEST):** SYNTHETIC Naive WAPE=72.8181 | UCI MA-30 WAPE=86.3870  

Models: RandomForest, HistGradientBoosting, LightGBM, XGBoost  
Selection on VALIDATION only; final scores on untouched TEST.
"""))

    cells.append(md("## 1–3. Load Features & Validation"))
    cells.append(code("""import os, sys
import pandas as pd

BASE_DIR = os.path.abspath(".")
if os.path.basename(BASE_DIR) == "notebooks":
    BASE_DIR = os.path.abspath("..")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml_forecasting import (
    load_feature_dataset, EXCLUDED_FIELDS, BASELINE_BENCHMARKS, ML_DIR, MODELS_DIR,
    write_ml_report,
)
from src.validate_ml_forecasting import run_validation

df = load_feature_dataset()
print("shape", df.shape)
print("split", df.split.value_counts().to_dict())
print("baselines", BASELINE_BENCHMARKS)
print("excluded fields:", list(EXCLUDED_FIELDS))
assert len(df) == 1995496
assert set(df.split.unique()) == {"train","validation","test"}
print("Phase 6 input OK")
"""))

    cells.append(md("## 4–10. Load Saved Phase 8 Results (trained pipeline)"))
    cells.append(code("""metrics = pd.read_parquet(os.path.join(ML_DIR, "ml_model_metrics.parquet"))
preds = pd.read_parquet(os.path.join(ML_DIR, "ml_predictions.parquet"))
importance = pd.read_parquet(os.path.join(ML_DIR, "feature_importance.parquet"))
err = pd.read_parquet(os.path.join(ML_DIR, "ml_error_analysis.parquet"))

print("=== VALIDATION metrics ===")
print(metrics[metrics.split=="validation"][["source_dataset","model","MAE","RMSE","sMAPE","WAPE","training_time"]]
      .sort_values(["source_dataset","WAPE"]).to_string(index=False))

print("\\n=== TEST metrics (+ baseline comparison) ===")
print(metrics[metrics.split=="test"][["source_dataset","model","MAE","RMSE","sMAPE","WAPE",
      "baseline_WAPE","wape_improvement_pct","selected"]]
      .sort_values(["source_dataset","WAPE"]).to_string(index=False))
"""))

    cells.append(md("## 11. Baseline Comparison"))
    cells.append(code("""for src in ["UCI","SYNTHETIC"]:
    best = metrics[(metrics.source_dataset==src)&(metrics.split=="test")&(metrics.selected==True)].iloc[0]
    base = BASELINE_BENCHMARKS[src]
    beat = best.WAPE < base["WAPE"]
    print(f"{src}: best={best.model} WAPE={best.WAPE:.4f} vs baseline {base['model']} {base['WAPE']}")
    print(f"  improvement_pct={best.wape_improvement_pct:.4f} | beat_baseline={beat}")
"""))

    cells.append(md("## 12. Feature Importance"))
    cells.append(code("""for src in ["UCI","SYNTHETIC"]:
    sub = importance[(importance.source_dataset==src)&(importance.importance_type=="native")].sort_values("rank").head(20)
    print(f"\\n{src} top-20 native importance:")
    print(sub[["rank","feature","importance"]].to_string(index=False))
"""))

    cells.append(md("## 13–14. Product / Store / Error Analysis"))
    cells.append(code("""for src in ["UCI","SYNTHETIC"]:
    bp = pd.read_parquet(os.path.join(ML_DIR, f"ml_metrics_by_product_{src.lower()}.parquet"))
    print(f"\\n{src} best products:")
    print(bp.sort_values("WAPE").head(3)[["entity_id","product_key","WAPE","MAE"]].to_string(index=False))
    print(f"{src} worst products:")
    print(bp.sort_values("WAPE", ascending=False).head(3)[["entity_id","product_key","WAPE","MAE"]].to_string(index=False))
    hv_path = os.path.join(ML_DIR, f"ml_high_value_{src.lower()}.parquet")
    if os.path.exists(hv_path):
        print(pd.read_parquet(hv_path).to_string(index=False))

be = pd.read_parquet(os.path.join(ML_DIR, "ml_metrics_by_entity_synthetic.parquet"))
print("\\nSynthetic stores:")
print(be.sort_values("WAPE")[["entity_id","MAE","RMSE","WAPE"]].to_string(index=False))
print("\\nError analysis:")
print(err.to_string(index=False))
"""))

    cells.append(md("## 15–17. Saved Models & Predictions"))
    cells.append(code("""import joblib
for src in ["uci","synthetic"]:
    path = os.path.join(MODELS_DIR, f"{src}_best_model.joblib")
    obj = joblib.load(path)
    print(src, "->", obj["model_name"], "features", len(obj["feature_names"]))
print("predictions rows", len(preds), preds.source_dataset.value_counts().to_dict())
print(preds.head(3).to_string(index=False))
"""))

    cells.append(md("## 18. Validation"))
    cells.append(code("""result = run_validation()
print("VALIDATION:", result.summary())
assert result.failed == 0
print("Phase 8 COMPLETE — STOP before Phase 9.")
"""))

    cells.append(md("## 19. Summary"))
    cells.append(code("""print("Models evaluated: random_forest, hist_gradient_boosting, lightgbm, xgboost")
for src in ["UCI","SYNTHETIC"]:
    best = metrics[(metrics.source_dataset==src)&(metrics.split=="test")&(metrics.selected==True)].iloc[0]
    print(src, dict(best[["model","MAE","RMSE","sMAPE","WAPE","baseline_WAPE","wape_improvement_pct","training_time"]]))
print("Validation:", result.summary())
"""))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def main() -> int:
    nb = build_notebook()
    os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    import nbformat
    from nbformat.validator import normalize
    from nbconvert.preprocessors import ExecutePreprocessor

    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    normalize(notebook)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    print(f"Executing {len(notebook.cells)} cells...")
    try:
        ep.preprocess(notebook, {"metadata": {"path": BASE_DIR}})
    except Exception as e:
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        print("NOTEBOOK ERROR:", e)
        return 1

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)
    errs = sum(
        1 for c in notebook.cells if c.cell_type == "code"
        for o in c.get("outputs", []) if o.get("output_type") == "error"
    )
    print(f"Notebook done. errors={errs}")
    return 0 if errs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
