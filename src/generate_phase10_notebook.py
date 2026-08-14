"""Generate notebooks/09_phase10_forecasting_improvements.ipynb"""

from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "09_phase10_forecasting_improvements.ipynb")


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
    cells.append(md("""# Project FORESIGHT — Phase 10: Forecasting Strategy Improvement

Experimental layer on top of **frozen** Phase 8 LightGBM and **frozen** Phase 9 stability analysis.

Priority: hurdle/zero-demand → direct multi-horizon → prediction intervals → small HPO.
"""))
    cells.append(md("## 1. Load artifacts"))
    cells.append(code("""import os, sys, json
import pandas as pd
from IPython.display import Image, display

BASE_DIR = os.path.abspath(".")
if os.path.basename(BASE_DIR) == "notebooks":
    BASE_DIR = os.path.abspath("..")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.phase10_common import PHASE10_DIR, FIGURES_DIR, METADATA_PATH, REGISTRY_PATH, PHASE8_TEST
from src.validate_phase10 import run_validation

meta = json.load(open(METADATA_PATH, encoding="utf-8"))
print("Phase 8 frozen", PHASE8_TEST)
print("phase8_unchanged", meta.get("phase8_unchanged"), "phase9_unchanged", meta.get("phase9_unchanged"))
print("decision", meta["decision"]["option"], meta["decision"]["label"])
print(meta["decision"]["statement"])
"""))
    cells.append(md("## 2. Hurdle vs Phase 8"))
    cells.append(code("""hv = pd.read_parquet(os.path.join(PHASE10_DIR, "hurdle_vs_phase8.parquet"))
print(hv.to_string(index=False))
thr = pd.read_parquet(os.path.join(PHASE10_DIR, "hurdle_threshold_table.parquet"))
print(thr.to_string(index=False))
display(Image(os.path.join(FIGURES_DIR, "zero_demand_threshold_analysis.png")))
display(Image(os.path.join(FIGURES_DIR, "classifier_confusion_matrix.png")))
display(Image(os.path.join(FIGURES_DIR, "hurdle_vs_lightgbm.png")))
"""))
    cells.append(md("## 3. Intermittent baselines"))
    cells.append(code("""p = os.path.join(PHASE10_DIR, "intermittent_summary.parquet")
if os.path.exists(p):
    print(pd.read_parquet(p).to_string(index=False))
    display(Image(os.path.join(FIGURES_DIR, "intermittent_baseline_comparison.png")))
else:
    print("No intermittent summary (unexpected for SYNTHETIC)")
"""))
    cells.append(md("## 4. Direct vs recursive"))
    cells.append(code("""print(pd.read_parquet(os.path.join(PHASE10_DIR, "direct_vs_recursive.parquet")).to_string(index=False))
display(Image(os.path.join(FIGURES_DIR, "direct_vs_recursive_horizon.png")))
display(Image(os.path.join(FIGURES_DIR, "horizon_improvement.png")))
"""))
    cells.append(md("## 5. Prediction intervals"))
    cells.append(code("""print(pd.read_parquet(os.path.join(PHASE10_DIR, "quantile_summary.parquet")).to_string(index=False))
display(Image(os.path.join(FIGURES_DIR, "prediction_interval_coverage.png")))
display(Image(os.path.join(FIGURES_DIR, "prediction_interval_width.png")))
display(Image(os.path.join(FIGURES_DIR, "quantile_calibration.png")))
"""))
    cells.append(md("## 6. HPO"))
    cells.append(code("""print(pd.read_parquet(os.path.join(PHASE10_DIR, "hpo_grid.parquet")).to_string(index=False))
print(pd.read_parquet(os.path.join(PHASE10_DIR, "hpo_summary.parquet")).to_string(index=False))
display(Image(os.path.join(FIGURES_DIR, "hyperparameter_comparison.png")))
"""))
    cells.append(md("## 7. Validation"))
    cells.append(code("""result = run_validation()
print("VALIDATION", result.summary())
assert result.failed == 0
print("Phase 10 COMPLETE — STOP before Phase 11.")
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
    print("Wrote", write_notebook())
    return execute_notebook()


if __name__ == "__main__":
    sys.exit(main())
