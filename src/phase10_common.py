"""
Phase 10 — Shared paths, freeze contract, metrics, experiment registry.

Experimental layer. Does not overwrite Phase 8 or Phase 9 artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.phase9_common import evaluate_extended  # metric parity with Phase 8/9

FEATURES_PATH = os.path.join(
    BASE_DIR, "data", "processed", "features", "forecast_features.parquet"
)
ML_PRED_PATH = os.path.join(
    BASE_DIR, "data", "processed", "forecasts", "ml", "ml_predictions.parquet"
)
ML_METRICS_PATH = os.path.join(
    BASE_DIR, "data", "processed", "forecasts", "ml", "ml_model_metrics.parquet"
)
UCI_MODEL_PATH = os.path.join(BASE_DIR, "models", "uci_best_model.joblib")
SYN_MODEL_PATH = os.path.join(BASE_DIR, "models", "synthetic_best_model.joblib")
LEGACY_LGB_PATH = os.path.join(BASE_DIR, "models", "lightgbm_forecaster.joblib")

PHASE9_DIR = os.path.join(BASE_DIR, "data", "processed", "forecasts", "phase9")
PHASE9_REPORT = os.path.join(BASE_DIR, "docs", "phase9_analysis_report.md")
PHASE9_META = os.path.join(BASE_DIR, "docs", "phase9_metadata.json")
PHASE9_HORIZON = os.path.join(PHASE9_DIR, "horizon_summary.parquet")
PHASE9_HORIZON_DETAIL = os.path.join(PHASE9_DIR, "horizon_detail.parquet")
PHASE9_ZERO = os.path.join(PHASE9_DIR, "zero_demand.parquet")

PHASE10_DIR = os.path.join(BASE_DIR, "data", "processed", "forecasts", "phase10")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures", "forecasting", "phase10")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
REPORT_PATH = os.path.join(DOCS_DIR, "phase10_analysis_report.md")
METADATA_PATH = os.path.join(DOCS_DIR, "phase10_metadata.json")
REGISTRY_PATH = os.path.join(DOCS_DIR, "phase10_experiment_registry.json")
NOTEBOOK_PATH = os.path.join(
    BASE_DIR, "notebooks", "09_phase10_forecasting_improvements.ipynb"
)

GRAIN = ["date", "source_dataset", "entity_id", "product_key"]
SERIES_KEYS = ["entity_id", "product_key"]
TARGET = "units_sold"
RANDOM_STATE = 42
HORIZONS = (1, 3, 7, 14, 30)
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
QUANTILES = (0.10, 0.50, 0.90)

PHASE8_TEST = {
    "UCI": {"model": "lightgbm", "MAE": 17.3447, "RMSE": 70.8952, "sMAPE": 82.8734, "WAPE": 79.4710},
    "SYNTHETIC": {"model": "lightgbm", "MAE": 2.8156, "RMSE": 5.1469, "sMAPE": 113.6813, "WAPE": 38.8923},
}

CALENDAR_COLS = [
    "year", "month", "quarter", "week_of_year", "day_of_week",
    "day_of_month", "day_of_year", "is_weekend",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_holiday", "season",
]

LGB_POINT_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

PHASE8_FREEZE_FILES = [
    ML_PRED_PATH, ML_METRICS_PATH, UCI_MODEL_PATH, SYN_MODEL_PATH, LEGACY_LGB_PATH,
]
PHASE9_FREEZE_FILES = [
    PHASE9_REPORT, PHASE9_META, PHASE9_HORIZON, PHASE9_HORIZON_DETAIL, PHASE9_ZERO,
    os.path.join(PHASE9_DIR, "walk_forward_folds.parquet"),
    os.path.join(PHASE9_DIR, "residual_overall.parquet"),
]


def file_md5(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_hashes(paths: list[str]) -> dict[str, dict]:
    out = {}
    for p in paths:
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


def ensure_dirs() -> None:
    os.makedirs(PHASE10_DIR, exist_ok=True)
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


def zero_slice_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    z = y_true == 0
    nz = ~z
    n = len(y_true)
    n_z = int(z.sum())
    n_nz = int(nz.sum())
    z_mae = float(np.mean(np.abs(y_pred[z] - y_true[z]))) if n_z else np.nan
    nz_mae = float(np.mean(np.abs(y_pred[nz] - y_true[nz]))) if n_nz else np.nan
    z_pos = float(np.mean(y_pred[z] > 0) * 100.0) if n_z else np.nan
    return {
        "n": n,
        "n_zero": n_z,
        "n_nonzero": n_nz,
        "zero_share_pct": round(100.0 * n_z / n, 2) if n else np.nan,
        "zero_mae": round(z_mae, 4) if z_mae == z_mae else np.nan,
        "nonzero_mae": round(nz_mae, 4) if nz_mae == nz_mae else np.nan,
        "zero_positive_prediction_rate": round(z_pos, 2) if z_pos == z_pos else np.nan,
        "predicted_zero_rate": round(100.0 * float(np.mean(y_pred == 0)), 2) if n else np.nan,
        "actual_zero_rate": round(100.0 * n_z / n, 2) if n else np.nan,
        "counts_reconcile": int(n_z + n_nz == n),
    }


def forecast_metrics(y_true, y_pred, model_name: str = "model") -> dict:
    m = evaluate_extended(y_true, y_pred, model_name=model_name)
    m.update(zero_slice_metrics(y_true, y_pred))
    return m


def pinball_loss(y_true: np.ndarray, y_q: np.ndarray, tau: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_q = np.asarray(y_q, dtype=float)
    e = y_true - y_q
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1.0) * e)))


class ExperimentRegistry:
    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def add(self, **kwargs) -> None:
        rec = {
            "experiment_id": kwargs.get("experiment_id") or f"exp_{len(self.records)+1:03d}",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": kwargs.get("random_seed", RANDOM_STATE),
            "status": kwargs.get("status", "completed"),
        }
        rec.update(kwargs)
        self.records.append(rec)

    def save(self, path: str = REGISTRY_PATH) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, default=str)
        return path


def split_end_dates(df_src: pd.DataFrame) -> dict[str, pd.Timestamp]:
    out = {}
    for sp in ["train", "validation", "test"]:
        part = df_src[df_src["split"] == sp]
        out[sp] = part["date"].max() if len(part) else pd.NaT
    return out
