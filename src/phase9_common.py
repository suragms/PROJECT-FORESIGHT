"""
Phase 9 — Shared paths, metric helpers, and Phase 8 freeze contract.

Does not retrain or overwrite Phase 8 artifacts.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.baseline_forecasting import calculate_metrics  # Phase 7/8 metric parity

FEATURES_PATH = os.path.join(
    BASE_DIR, "data", "processed", "features", "forecast_features.parquet"
)
ML_PRED_PATH = os.path.join(
    BASE_DIR, "data", "processed", "forecasts", "ml", "ml_predictions.parquet"
)
ML_METRICS_PATH = os.path.join(
    BASE_DIR, "data", "processed", "forecasts", "ml", "ml_model_metrics.parquet"
)
ML_ENTITY_SYN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "forecasts", "ml",
    "ml_metrics_by_entity_synthetic.parquet",
)
UCI_MODEL_PATH = os.path.join(BASE_DIR, "models", "uci_best_model.joblib")
SYN_MODEL_PATH = os.path.join(BASE_DIR, "models", "synthetic_best_model.joblib")
LEGACY_LGB_PATH = os.path.join(BASE_DIR, "models", "lightgbm_forecaster.joblib")

PHASE9_DIR = os.path.join(BASE_DIR, "data", "processed", "forecasts", "phase9")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures", "forecasting", "phase9")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
REPORT_PATH = os.path.join(DOCS_DIR, "phase9_analysis_report.md")
METADATA_PATH = os.path.join(DOCS_DIR, "phase9_metadata.json")
NOTEBOOK_PATH = os.path.join(
    BASE_DIR, "notebooks", "08_phase9_stability_residual_horizon.ipynb"
)

GRAIN = ["date", "source_dataset", "entity_id", "product_key"]
TARGET = "units_sold"
RANDOM_STATE = 42

PHASE8_TEST = {
    "UCI": {"model": "lightgbm", "MAE": 17.3447, "RMSE": 70.8952, "sMAPE": 82.8734, "WAPE": 79.4710},
    "SYNTHETIC": {"model": "lightgbm", "MAE": 2.8156, "RMSE": 5.1469, "sMAPE": 113.6813, "WAPE": 38.8923},
}

# Expanding-window folds. Train always precedes validation; no overlap.
FOLDS = {
    "UCI": [
        {"fold": 1, "train_end": "2010-07-13", "val_start": "2010-07-14", "val_end": "2010-12-31"},
        {"fold": 2, "train_end": "2010-12-31", "val_start": "2011-01-01", "val_end": "2011-04-30"},
        {"fold": 3, "train_end": "2011-04-30", "val_start": "2011-05-01", "val_end": "2011-07-13"},
        {"fold": 4, "train_end": "2011-07-13", "val_start": "2011-07-14", "val_end": "2011-09-25"},
        {"fold": 5, "train_end": "2011-09-25", "val_start": "2011-09-26", "val_end": "2011-12-09"},
    ],
    "SYNTHETIC": [
        {"fold": 1, "train_end": "2023-12-31", "val_start": "2024-01-01", "val_end": "2024-06-30"},
        {"fold": 2, "train_end": "2024-06-30", "val_start": "2024-07-01", "val_end": "2024-12-31"},
        {"fold": 3, "train_end": "2024-12-31", "val_start": "2025-01-01", "val_end": "2025-03-13"},
        {"fold": 4, "train_end": "2025-03-13", "val_start": "2025-03-14", "val_end": "2025-08-06"},
        {"fold": 5, "train_end": "2025-08-06", "val_start": "2025-08-07", "val_end": "2025-12-31"},
    ],
}

# Stability rule (documented, not arbitrary silent cuts):
#   CV_WAPE = std(fold WAPE) / mean(fold WAPE)
#   range_ratio = max(WAPE) / min(WAPE)
# Stable if CV < 0.15 AND range_ratio < 1.50
# Moderately Stable if CV < 0.35 AND range_ratio < 2.00
# else Unstable
STABILITY_RULES = {
    "stable_cv_max": 0.15,
    "stable_range_ratio_max": 1.50,
    "moderate_cv_max": 0.35,
    "moderate_range_ratio_max": 2.00,
}

HORIZONS = (1, 3, 7, 14, 30)

LAG_PERIODS = (1, 2, 3, 7, 14, 21, 28, 30)
ROLL_WINDOWS = (7, 14, 30)

PHASE8_FREEZE_FILES = [
    ML_PRED_PATH,
    ML_METRICS_PATH,
    UCI_MODEL_PATH,
    SYN_MODEL_PATH,
    LEGACY_LGB_PATH,
]


def file_md5(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_phase8_hashes() -> dict[str, dict]:
    out = {}
    for p in PHASE8_FREEZE_FILES:
        out[os.path.relpath(p, BASE_DIR)] = {
            "exists": os.path.exists(p),
            "size": os.path.getsize(p) if os.path.exists(p) else None,
            "md5": file_md5(p),
        }
    return out


def hashes_unchanged(before: dict, after: dict) -> tuple[bool, list[str]]:
    changed = []
    for k, b in before.items():
        a = after.get(k, {})
        if b.get("md5") != a.get("md5") or b.get("size") != a.get("size"):
            changed.append(k)
    return len(changed) == 0, changed


def evaluate_extended(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Phase 7/8 metrics plus bias / over-under rates.

    residual (analysis) = actual - prediction
    bias (Phase 8 convention) = mean(prediction - actual)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    base = calculate_metrics(y_true, y_pred, model_name=model_name)
    if len(y_true) == 0:
        base.update({
            "bias": np.nan,
            "mean_residual": np.nan,
            "median_residual": np.nan,
            "residual_std": np.nan,
            "overprediction_pct": np.nan,
            "underprediction_pct": np.nan,
        })
        return base
    residual = y_true - y_pred
    bias = float(np.mean(y_pred - y_true))
    base.update({
        "bias": round(bias, 4),
        "mean_residual": round(float(np.mean(residual)), 4),
        "median_residual": round(float(np.median(residual)), 4),
        "residual_std": round(float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0, 4),
        "overprediction_pct": round(100.0 * float(np.mean(y_pred > y_true)), 2),
        "underprediction_pct": round(100.0 * float(np.mean(y_pred < y_true)), 2),
    })
    return base


def classify_stability(wape_values: list[float]) -> dict:
    vals = np.asarray([v for v in wape_values if np.isfinite(v)], dtype=float)
    if len(vals) < 2:
        return {
            "label": "Unstable",
            "cv_wape": np.nan,
            "range_ratio": np.nan,
            "mean_wape": float(vals[0]) if len(vals) else np.nan,
            "reason": "Fewer than 2 finite fold WAPE values.",
        }
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    cv = std / mean if mean > 0 else np.inf
    rmin, rmax = float(np.min(vals)), float(np.max(vals))
    ratio = rmax / rmin if rmin > 0 else np.inf
    rules = STABILITY_RULES
    if cv < rules["stable_cv_max"] and ratio < rules["stable_range_ratio_max"]:
        label = "Stable"
        reason = (
            f"CV_WAPE={cv:.3f} < {rules['stable_cv_max']} and "
            f"max/min={ratio:.3f} < {rules['stable_range_ratio_max']}"
        )
    elif cv < rules["moderate_cv_max"] and ratio < rules["moderate_range_ratio_max"]:
        label = "Moderately Stable"
        reason = (
            f"CV_WAPE={cv:.3f} < {rules['moderate_cv_max']} and "
            f"max/min={ratio:.3f} < {rules['moderate_range_ratio_max']}"
        )
    else:
        label = "Unstable"
        reason = (
            f"CV_WAPE={cv:.3f} or max/min={ratio:.3f} exceeded moderate thresholds "
            f"(CV<{rules['moderate_cv_max']}, ratio<{rules['moderate_range_ratio_max']})"
        )
    return {
        "label": label,
        "cv_wape": round(cv, 4),
        "range_ratio": round(ratio, 4) if np.isfinite(ratio) else np.nan,
        "mean_wape": round(mean, 4),
        "std_wape": round(std, 4),
        "min_wape": round(rmin, 4),
        "max_wape": round(rmax, 4),
        "reason": reason,
    }


def fold_summary(df: pd.DataFrame, value_col: str) -> dict:
    s = df[value_col].dropna().astype(float)
    if s.empty:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "std": round(float(s.std(ddof=1)), 4) if len(s) > 1 else 0.0,
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
    }


def ensure_dirs() -> None:
    os.makedirs(PHASE9_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
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
