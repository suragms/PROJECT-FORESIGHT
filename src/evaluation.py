"""
Phase 9 — Model Evaluation & Benchmarking
===========================================
Project FORESIGHT: Demand & Inventory Intelligence

Provides standardized metrics for time-series forecasting models:
MAE, RMSE, MAPE, WAPE, R2, Forecast Bias, and Tracking Signal.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    Calculate comprehensive forecasting evaluation metrics.
    Handles zero values gracefully for percentage metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Ensure non-negative predictions for demand
    y_pred = np.maximum(0, y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.var(y_true) > 0 else 0.0

    # WAPE (Weighted Absolute Percentage Error) - industry standard for intermittent/retail demand
    sum_actual = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / sum_actual * 100) if sum_actual > 0 else 0.0

    # MAPE (with small epsilon to avoid divide-by-zero)
    eps = 1e-5
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)

    # Forecast Bias (Mean Forecast Error)
    bias = float(np.mean(y_pred - y_true))

    return {
        "model": model_name,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "wape_pct": round(wape, 2),
        "mape_pct": round(min(mape, 500.0), 2),  # cap extreme MAPE outliers
        "r2": round(r2, 4),
        "bias": round(bias, 3),
        "sample_count": len(y_true),
    }


def compare_models(predictions_dict: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """
    Generate comparison leaderboard table across multiple models.
    `predictions_dict` format: { "Model Name": (y_true, y_pred) }
    """
    records = []
    for name, (y_true, y_pred) in predictions_dict.items():
        metrics = calculate_metrics(y_true, y_pred, model_name=name)
        records.append(metrics)
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="wape_pct", ascending=True).reset_index(drop=True)
    return df
