"""
Generate and execute notebooks/08_phase9_stability_residual_horizon.ipynb
"""

from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "08_phase9_stability_residual_horizon.ipynb")


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
    cells.append(md("""# Project FORESIGHT — Phase 9: Stability, Residual & Horizon Analysis

Phase 8 LightGBM is **frozen**. This notebook evaluates walk-forward stability,
residuals, recursive horizons, zero-demand behavior, and store variation.

Do not train production replacements here.
"""))
    cells.append(md("## 1. Load Phase 9 artifacts"))
    cells.append(code("""import os, sys, json
import pandas as pd
from IPython.display import Image, display, Markdown

BASE_DIR = os.path.abspath(".")
if os.path.basename(BASE_DIR) == "notebooks":
    BASE_DIR = os.path.abspath("..")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.phase9_common import PHASE9_DIR, FIGURES_DIR, METADATA_PATH, PHASE8_TEST
from src.validate_phase9 import run_validation

folds = pd.read_parquet(os.path.join(PHASE9_DIR, "walk_forward_folds.parquet"))
wf_sum = pd.read_parquet(os.path.join(PHASE9_DIR, "walk_forward_summary.parquet"))
overall = pd.read_parquet(os.path.join(PHASE9_DIR, "residual_overall.parquet"))
regimes = pd.read_parquet(os.path.join(PHASE9_DIR, "residual_by_regime.parquet"))
zero = pd.read_parquet(os.path.join(PHASE9_DIR, "zero_demand.parquet"))
entities = pd.read_parquet(os.path.join(PHASE9_DIR, "entity_metrics.parquet"))
hsum = pd.read_parquet(os.path.join(PHASE9_DIR, "horizon_summary.parquet"))
diag = pd.read_parquet(os.path.join(PHASE9_DIR, "residual_diagnostics.parquet"))
meta = json.load(open(METADATA_PATH, encoding="utf-8"))
print("Phase 8 frozen WAPE", PHASE8_TEST)
print("walk-forward summary:")
print(wf_sum.to_string(index=False))
print("phase8_unchanged", meta.get("phase8_unchanged"))
print("conclusion", meta["conclusion"]["option"], meta["conclusion"]["label"])
"""))
    cells.append(md("## 2. Walk-forward folds"))
    cells.append(code("""print(folds[[
    "source_dataset","fold","train_end","val_start","val_end",
    "train_rows","val_rows","MAE","RMSE","sMAPE","WAPE","bias",
    "overprediction_pct","underprediction_pct"
]].to_string(index=False))
assert folds.groupby("source_dataset")["fold"].nunique().min() == 5
assert (pd.to_datetime(folds.train_end_actual) < pd.to_datetime(folds.val_start_actual)).all()
print("Chronology OK")
display(Image(os.path.join(FIGURES_DIR, "walk_forward_wape.png")))
display(Image(os.path.join(FIGURES_DIR, "walk_forward_mae.png")))
display(Image(os.path.join(FIGURES_DIR, "fold_stability.png")))
"""))
    cells.append(md("## 3. Residual overview"))
    cells.append(code("""print(overall.to_string(index=False))
print("\\nregimes:")
print(regimes.to_string(index=False))
print("\\ndiagnostics:")
print(diag.to_string(index=False))
display(Image(os.path.join(FIGURES_DIR, "residual_distribution_uci.png")))
display(Image(os.path.join(FIGURES_DIR, "residual_distribution_synthetic.png")))
display(Image(os.path.join(FIGURES_DIR, "residual_vs_actual_uci.png")))
display(Image(os.path.join(FIGURES_DIR, "residual_vs_actual_synthetic.png")))
display(Image(os.path.join(FIGURES_DIR, "residual_over_time_uci.png")))
display(Image(os.path.join(FIGURES_DIR, "residual_over_time_synthetic.png")))
"""))
    cells.append(md("## 4. Zero-demand"))
    cells.append(code("""print(zero.to_string(index=False))
for _, r in zero.iterrows():
    assert int(r.n_zero) + int(r.n_nonzero) == int(r.n_total)
print("Zero-demand counts reconcile")
display(Image(os.path.join(FIGURES_DIR, "zero_demand_analysis.png")))
"""))
    cells.append(md("## 5. Horizon table"))
    cells.append(code("""print(hsum.to_string(index=False))
print("supported horizons", sorted(hsum.horizon.unique().tolist()))
display(Image(os.path.join(FIGURES_DIR, "horizon_performance_uci.png")))
display(Image(os.path.join(FIGURES_DIR, "horizon_performance_synthetic.png")))
"""))
    cells.append(md("## 6. Store stability"))
    cells.append(code("""syn = entities[entities.source_dataset=="SYNTHETIC"].sort_values("WAPE")
print(syn[["entity_id","n","MAE","RMSE","WAPE","bias"]].to_string(index=False))
print("best", syn.iloc[0].entity_id, syn.iloc[0].WAPE)
print("worst", syn.iloc[-1].entity_id, syn.iloc[-1].WAPE)
display(Image(os.path.join(FIGURES_DIR, "store_stability.png")))
"""))
    cells.append(md("## 7. Figures present"))
    cells.append(code("""figs = sorted(os.listdir(FIGURES_DIR))
print("\\n".join(figs))
assert "walk_forward_wape.png" in figs
assert "store_stability.png" in figs
assert "horizon_performance_uci.png" in figs
assert "horizon_performance_synthetic.png" in figs
"""))
    cells.append(md("## 8. Validation"))
    cells.append(code("""result = run_validation()
print("VALIDATION", result.summary())
assert result.failed == 0
print("Phase 9 COMPLETE — STOP before Phase 10.")
print("Conclusion:", meta["conclusion"]["option"], meta["conclusion"]["label"])
print(meta["conclusion"]["statement"])
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


def write_notebook() -> str:
    nb = build_notebook()
    os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    return NOTEBOOK_PATH


def execute_notebook() -> int:
    import nbformat
    from nbformat.validator import normalize
    from nbconvert.preprocessors import ExecutePreprocessor

    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    normalize(notebook)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
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


def main() -> int:
    path = write_notebook()
    print(f"Wrote {path}")
    return execute_notebook()


if __name__ == "__main__":
    sys.exit(main())
