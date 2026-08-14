"""
Central configuration for Phase 12 production packaging.

Paths are relative to the repository root. Override with environment
variables; never hard-code machine-specific Windows paths.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("FORESIGHT_PROJECT_ROOT")
    or Path(__file__).resolve().parent.parent
)

APP_NAME = "Demand-Inventory-Intelligence"
APP_VERSION = os.environ.get("FORESIGHT_APP_VERSION", "0.12.0")
LOG_LEVEL = os.environ.get("FORESIGHT_LOG_LEVEL", "INFO")

REGISTRY_PATH = PROJECT_ROOT / os.environ.get(
    "FORESIGHT_REGISTRY_PATH", os.path.join("docs", "final_model_registry.json")
)
MODELS_FINAL_DIR = PROJECT_ROOT / "models" / "final"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "forecast_features.parquet"
FINAL_FORECASTS_PATH = PROJECT_ROOT / "data" / "processed" / "forecasts" / "final" / "final_predictions.parquet"
PHASE11_META_PATH = PROJECT_ROOT / "docs" / "phase11_metadata.json"
PHASE10_META_PATH = PROJECT_ROOT / "docs" / "phase10_metadata.json"
PHASE9_META_PATH = PROJECT_ROOT / "docs" / "phase9_metadata.json"
MONITORING_PLAN_PATH = PROJECT_ROOT / "docs" / "forecast_monitoring_plan.md"

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
OUTPUTS_FORECASTS_DIR = PROJECT_ROOT / "outputs" / "forecasts"
OUTPUTS_MONITORING_DIR = PROJECT_ROOT / "outputs" / "monitoring"
OUTPUTS_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

SUPPORTED_DATASETS = ("UCI", "SYNTHETIC")
SUPPORTED_HORIZONS = (1, 3, 7, 14, 30)

API_HOST = os.environ.get("FORESIGHT_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("FORESIGHT_API_PORT", "8000"))
API_MAX_BATCH = int(os.environ.get("FORESIGHT_API_MAX_BATCH", "500"))
API_MAX_PAYLOAD_BYTES = int(os.environ.get("FORESIGHT_API_MAX_PAYLOAD_BYTES", str(2_000_000)))

# Evidence-based monitoring thresholds (docs/forecast_monitoring_plan.md).
MONITORING = {
    "unseen_category_rate_warn": 5.0,
    "feature_zscore_warn": 3.0,
    "pred_mean_rel_change_warn": 0.25,
    "synthetic_zero_pred_rate_min": 50.0,
    "synthetic_zero_pred_rate_max": 75.0,
    "synthetic_zero_fp_warn": 10.0,
    "volume_rel_warn": 0.02,
    "uci_h1_wape_fold2": 105.0,
    "uci_h1_wape_1p5x": 119.2,
    "synthetic_h1_wape_1p5x": 39.4,
    "interval_coverage_min": 70.0,
    "interval_coverage_max": 95.0,
    "psi_warn": 0.20,
    "ks_warn": 0.20,
}

PHASE8_FREEZE = [
    PROJECT_ROOT / "data" / "processed" / "forecasts" / "ml" / "ml_predictions.parquet",
    PROJECT_ROOT / "data" / "processed" / "forecasts" / "ml" / "ml_model_metrics.parquet",
    PROJECT_ROOT / "models" / "uci_best_model.joblib",
    PROJECT_ROOT / "models" / "synthetic_best_model.joblib",
    PROJECT_ROOT / "models" / "lightgbm_forecaster.joblib",
]
PHASE9_FREEZE = [
    PROJECT_ROOT / "docs" / "phase9_analysis_report.md",
    PROJECT_ROOT / "docs" / "phase9_metadata.json",
    PROJECT_ROOT / "data" / "processed" / "forecasts" / "phase9" / "horizon_summary.parquet",
    PROJECT_ROOT / "data" / "processed" / "forecasts" / "phase9" / "horizon_detail.parquet",
    PROJECT_ROOT / "data" / "processed" / "forecasts" / "phase9" / "zero_demand.parquet",
    PROJECT_ROOT / "data" / "processed" / "forecasts" / "phase9" / "walk_forward_folds.parquet",
    PROJECT_ROOT / "data" / "processed" / "forecasts" / "phase9" / "residual_overall.parquet",
]
PHASE10_FREEZE = [
    PROJECT_ROOT / "docs" / "phase10_analysis_report.md",
    PROJECT_ROOT / "docs" / "phase10_metadata.json",
    PROJECT_ROOT / "docs" / "phase10_experiment_registry.json",
    PROJECT_ROOT / "data" / "processed" / "forecasts" / "phase10" / "hurdle_vs_phase8.parquet",
]
PHASE11_FREEZE = [
    PROJECT_ROOT / "docs" / "final_model_registry.json",
    PROJECT_ROOT / "docs" / "final_forecasting_report.md",
    PROJECT_ROOT / "docs" / "forecast_monitoring_plan.md",
    PROJECT_ROOT / "docs" / "phase11_metadata.json",
    PROJECT_ROOT / "data" / "processed" / "forecasts" / "final" / "final_predictions.parquet",
]
