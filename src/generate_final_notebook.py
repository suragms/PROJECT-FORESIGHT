"""Generate notebooks/10_final_forecasting.ipynb"""

from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "10_final_forecasting.ipynb")


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
    cells.append(md("""# Project FORESIGHT — Phase 11: Final Forecasting Model Selection

Consolidation of Phases 8–10. Frozen Phase 8 LightGBM and Phase 9 stability remain immutable.
"""))
    cells.append(md("## 1. Load registry and metadata"))
    cells.append(code("""import os, sys, json
import pandas as pd
from IPython.display import Image, display

BASE_DIR = os.path.abspath(".")
if os.path.basename(BASE_DIR) == "notebooks":
    BASE_DIR = os.path.abspath("..")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.phase11_common import (
    CANDIDATE_PATH, FIGURES_FINAL_DIR, FINAL_PRED_PATH, FORECASTS_FINAL_DIR,
    METADATA_PATH, REGISTRY_PATH, REPORT_PATH,
)
from src.validate_final_forecasting import run_validation

meta = json.load(open(METADATA_PATH, encoding="utf-8"))
reg = json.load(open(REGISTRY_PATH, encoding="utf-8"))
print("readiness", meta["production_readiness"])
print("UCI", meta["final_uci_model"])
print("SYNTHETIC", meta["final_synthetic_model"])
print("horizon", meta["horizon_strategy"])
print("phase8_unchanged", meta.get("phase8_unchanged"), "phase9_unchanged", meta.get("phase9_unchanged"))
print("n_models", len(reg))
print(pd.DataFrame(reg)[["model_id", "dataset", "horizon", "model_type", "status"]].to_string(index=False))
"""))
    cells.append(md("## 2. Candidate matrix and selection"))
    cells.append(code("""cand = pd.read_parquet(CANDIDATE_PATH)
print(cand.to_string(index=False))
print("--- selected ---")
print(pd.read_parquet(os.path.join(FORECASTS_FINAL_DIR, "selection_table.parquet")).to_string(index=False))
"""))
    cells.append(md("## 3. Final performance"))
    cells.append(code("""print(pd.read_parquet(os.path.join(FORECASTS_FINAL_DIR, "analysis_baseline.parquet")).to_string(index=False))
print(pd.read_parquet(os.path.join(FORECASTS_FINAL_DIR, "analysis_overall_h1.parquet")).to_string(index=False))
print(pd.read_parquet(os.path.join(FORECASTS_FINAL_DIR, "analysis_horizon.parquet")).to_string(index=False))
display(Image(os.path.join(FIGURES_FINAL_DIR, "final_model_comparison.png")))
display(Image(os.path.join(FIGURES_FINAL_DIR, "final_horizon_comparison.png")))
"""))
    cells.append(md("## 4. Error analysis"))
    cells.append(code("""print(pd.read_parquet(os.path.join(FORECASTS_FINAL_DIR, "analysis_bias.parquet")).to_string(index=False))
print(pd.read_parquet(os.path.join(FORECASTS_FINAL_DIR, "analysis_regime.parquet")).to_string(index=False))
print(pd.read_parquet(os.path.join(FORECASTS_FINAL_DIR, "analysis_store_summary.parquet")).to_string(index=False))
display(Image(os.path.join(FIGURES_FINAL_DIR, "final_residual_analysis.png")))
display(Image(os.path.join(FIGURES_FINAL_DIR, "final_store_stability.png")))
"""))
    cells.append(md("## 5. Zero-demand and intervals"))
    cells.append(code("""print(pd.read_parquet(os.path.join(FORECASTS_FINAL_DIR, "analysis_zero_final.parquet")).to_string(index=False))
print(pd.read_parquet(os.path.join(FORECASTS_FINAL_DIR, "analysis_zero_phase8.parquet")).to_string(index=False))
print(pd.read_parquet(os.path.join(FORECASTS_FINAL_DIR, "analysis_intervals.parquet")).to_string(index=False))
display(Image(os.path.join(FIGURES_FINAL_DIR, "final_zero_demand_comparison.png")))
display(Image(os.path.join(FIGURES_FINAL_DIR, "final_prediction_intervals.png")))
"""))
    cells.append(md("## 6. Validation"))
    cells.append(code("""result = run_validation()
print("VALIDATION", result.summary())
assert result.failed == 0
print("Phase 11 COMPLETE — STOP. Do not start another modeling phase.")
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
