"""
Phase 11 — Shared paths, freeze contract, selection rules, hashing.

Consolidation layer. Does not overwrite Phase 8, 9, or 10 artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.phase10_common import (  # noqa: E402
    FEATURES_PATH,
    FIGURES_DIR as PHASE10_FIGURES,
    GRAIN,
    HORIZONS,
    LGB_POINT_PARAMS,
    METADATA_PATH as PHASE10_META,
    ML_METRICS_PATH,
    ML_PRED_PATH,
    PHASE8_FREEZE_FILES,
    PHASE8_TEST,
    PHASE9_DIR,
    PHASE9_FREEZE_FILES,
    PHASE9_HORIZON,
    PHASE9_META,
    PHASE9_REPORT,
    PHASE9_ZERO,
    PHASE10_DIR,
    RANDOM_STATE,
    REGISTRY_PATH as PHASE10_REGISTRY,
    REPORT_PATH as PHASE10_REPORT,
    SYN_MODEL_PATH,
    TARGET,
    UCI_MODEL_PATH,
    file_md5,
    forecast_metrics,
    hashes_unchanged,
    pinball_loss,
    snapshot_hashes,
)
from src.phase9_common import STABILITY_RULES, classify_stability  # noqa: E402

DOCS_DIR = os.path.join(BASE_DIR, "docs")
MODELS_FINAL_DIR = os.path.join(BASE_DIR, "models", "final")
FORECASTS_FINAL_DIR = os.path.join(
    BASE_DIR, "data", "processed", "forecasts", "final"
)
FIGURES_FINAL_DIR = os.path.join(
    BASE_DIR, "outputs", "figures", "forecasting", "final"
)
REGISTRY_PATH = os.path.join(DOCS_DIR, "final_model_registry.json")
REPORT_PATH = os.path.join(DOCS_DIR, "final_forecasting_report.md")
MONITOR_PATH = os.path.join(DOCS_DIR, "forecast_monitoring_plan.md")
METADATA_PATH = os.path.join(DOCS_DIR, "phase11_metadata.json")
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "10_final_forecasting.ipynb")
CANDIDATE_PATH = os.path.join(FORECASTS_FINAL_DIR, "candidate_matrix.parquet")
SELECTION_PATH = os.path.join(FORECASTS_FINAL_DIR, "selection_table.parquet")
FINAL_PRED_PATH = os.path.join(FORECASTS_FINAL_DIR, "final_predictions.parquet")

FEATURE_VERSION = "phase6_forecast_features"
MODEL_STATUS_SELECTED = "selected"
MODEL_STATUS_INTERVAL = "interval_companion"

# Relative WAPE gain required to displace a walk-forward-validated model.
STABILITY_OVERRIDE_REL_WAPE = 0.03

PHASE7_BASELINE = {
    "UCI": {"model": "moving_average_30", "WAPE": 86.3870, "MAE": 18.8542},
    "SYNTHETIC": {"model": "naive", "WAPE": 72.8181, "MAE": 5.2717},
}

OUTPUT_SCHEMA = [
    "forecast_date",
    "source_dataset",
    "entity_id",
    "product_key",
    "horizon",
    "actual",
    "prediction",
    "lower_bound",
    "upper_bound",
    "model_name",
    "model_version",
]

PROHIBITED_NEGATIVE = [
    "units_sold_lag_1", "units_sold_lag_2", "units_sold_lag_3",
    "units_sold_lag_7", "units_sold_lag_14", "units_sold_lag_21",
    "units_sold_lag_28", "units_sold_lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "average_unit_price", "price_lag_1", "base_price",
]

LEAKAGE_FORBIDDEN = {
    "units_sold", "revenue", "transaction_count", "unique_customers",
}

PHASE10_REQUIRED_PARQUET = [
    "hurdle_vs_phase8.parquet",
    "hurdle_threshold_table.parquet",
    "intermittent_summary.parquet",
    "direct_horizon_summary.parquet",
    "direct_vs_recursive.parquet",
    "direct_horizon_predictions.parquet",
    "quantile_summary.parquet",
    "quantile_predictions.parquet",
    "hpo_grid.parquet",
    "hpo_summary.parquet",
]

SELECTION_LOGIC = """
Primary (in order):
  1. WAPE on held-out TEST (lower is better).
  2. MAE if WAPE is within 2% relative.
  3. Stability: if relative WAPE gain vs a walk-forward-validated model
     is below 3%, keep the walk-forward-validated model.
  4. Horizon: select independently per horizon. Operational h=1 uses the
     Phase 8 same-row contract; h in {3,7,14,30} uses leakage-safe direct models.

Secondary:
  5. Bias closer to 0.
  6. Zero-demand MAE and false-positive demand rate (SYNTHETIC).
  7. Prediction-interval coverage near 80% (diagnostic, not a point-forecast rule).
  8. Training / inference cost.
  9. Model complexity (prefer one-stage unless zeros require a hurdle).
 10. Reproducibility (frozen serialized artifacts preferred when quality is comparable).

Gates:
  - Exclude models with WAPE worse than the Phase 7 baseline.
  - Exclude hurdle on UCI (train zero share 0%; not identified).
  - Exclude Croston / SBA / TSB (worse than Naive on SYNTHETIC).
  - Do not select a model solely for lowest RMSE.
  - Quantile P50 is a point-forecast candidate only if it beats the selected
    point model on WAPE; otherwise P10/P90 remain interval companions.
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def file_sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_phase11_dirs() -> None:
    os.makedirs(MODELS_FINAL_DIR, exist_ok=True)
    os.makedirs(FORECASTS_FINAL_DIR, exist_ok=True)
    os.makedirs(FIGURES_FINAL_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)


def apply_mpl_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (10, 5.2),
        "figure.dpi": 120,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def relpath(path: str) -> str:
    return os.path.relpath(path, BASE_DIR)


def write_json(path: str, obj: Any) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def md_table(df, index: bool = False) -> str:
    if df is None or len(df) == 0:
        return "_(empty)_"
    try:
        return df.to_markdown(index=index)
    except Exception:
        return "```\n" + df.to_string(index=index) + "\n```"


def verify_phase10_ready() -> dict[str, Any]:
    """Stop-condition helper: Phase 10 must be complete before Phase 11."""
    issues = []
    if not os.path.exists(PHASE10_META):
        issues.append("missing docs/phase10_metadata.json")
        return {"ready": False, "issues": issues}
    with open(PHASE10_META, encoding="utf-8") as f:
        meta = json.load(f)
    val = meta.get("validation") or {}
    if val.get("passed") != val.get("total") or not val.get("total"):
        issues.append(f"Phase 10 validation incomplete: {val}")
    if not meta.get("phase8_unchanged"):
        issues.append("metadata.phase8_unchanged is not true")
    if not meta.get("phase9_unchanged"):
        issues.append("metadata.phase9_unchanged is not true")
    for name in PHASE10_REQUIRED_PARQUET:
        p = os.path.join(PHASE10_DIR, name)
        if not os.path.exists(p):
            issues.append(f"missing Phase 10 parquet: {name}")
    if not os.path.exists(PHASE10_REPORT):
        issues.append("missing docs/phase10_analysis_report.md")
    if not os.path.exists(PHASE10_REGISTRY):
        issues.append("missing docs/phase10_experiment_registry.json")
    cur8 = snapshot_hashes(PHASE8_FREEZE_FILES)
    ok8, ch8 = hashes_unchanged(meta.get("phase8_hashes_after") or {}, cur8)
    if not ok8:
        issues.append(f"Phase 8 hashes changed: {ch8}")
    cur9 = snapshot_hashes(PHASE9_FREEZE_FILES)
    ok9, ch9 = hashes_unchanged(meta.get("phase9_hashes_after") or {}, cur9)
    if not ok9:
        issues.append(f"Phase 9 hashes changed: {ch9}")
    return {
        "ready": len(issues) == 0,
        "issues": issues,
        "validation": val,
        "decision": meta.get("decision"),
        "phase8_unchanged": ok8,
        "phase9_unchanged": ok9,
    }
