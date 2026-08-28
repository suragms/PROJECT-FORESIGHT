"""
Phase 17 — Seasonal-Naive Baseline, Rolling-Origin Backtest, Candidate Models
==============================================================================
Forecasting at weekly SKU-level with WAPE as primary metric.
Horizon: 8 weeks (matching Zidio 6-8 week requirement).
"""

import os
import sys
import json
import hashlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

P17_DIR = os.path.join(BASE_DIR, "data", "phase17")
P17_FEAT = os.path.join(P17_DIR, "features")
P17_FCST = os.path.join(P17_DIR, "forecasts")
P17_BT = os.path.join(P17_DIR, "backtests")
MODELS_DIR = os.path.join(BASE_DIR, "models", "phase17")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

for d in [P17_FCST, P17_BT, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

HORIZON = 8  # weeks
GRAIN = ["source_dataset", "product_key"]
TARGET = "units_sold"
MIN_HISTORY_WEEKS = 52  # require at least 1 year of history


def wape(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual - forecast)) / denom)


def bias(actual, forecast):
    return float(np.mean(np.asarray(forecast) - np.asarray(actual)))


# =====================================================================
# SEASONAL-NAIVE BASELINE
# =====================================================================

def seasonal_naive_forecast(history: pd.Series, horizon: int, season_length: int = 52) -> np.ndarray:
    """Predict using the value from the same week last year."""
    vals = history.values
    preds = np.zeros(horizon)
    for h in range(horizon):
        idx = len(vals) - season_length + h
        if idx >= 0:
            preds[h] = max(0, vals[idx])
        else:
            preds[h] = max(0, vals[-1]) if len(vals) > 0 else 0
    return preds


# =====================================================================
# ROLLING-ORIGIN BACKTEST
# =====================================================================

def rolling_origin_backtest(df: pd.DataFrame, source: str, n_folds: int = 5):
    """
    Rolling-origin backtest for a single source_dataset.
    Each fold trains on all data up to origin, forecasts HORIZON weeks ahead.
    """
    src_df = df[df["source_dataset"] == source].copy()
    weeks = sorted(src_df["week"].unique())

    if len(weeks) < MIN_HISTORY_WEEKS + HORIZON + n_folds:
        print(f"  {source}: insufficient weeks ({len(weeks)}) for {n_folds}-fold backtest")
        n_folds = max(1, len(weeks) - MIN_HISTORY_WEEKS - HORIZON)
        if n_folds < 1:
            return pd.DataFrame(), {}

    # Define fold origins
    last_trainable = len(weeks) - HORIZON
    fold_origins = []
    for i in range(n_folds):
        idx = last_trainable - i
        if idx >= MIN_HISTORY_WEEKS:
            fold_origins.append(idx)
    fold_origins = sorted(fold_origins)

    if not fold_origins:
        return pd.DataFrame(), {}

    skus = sorted(src_df["product_key"].unique())
    all_results = []

    # Feature columns for ML
    feature_cols = [c for c in src_df.columns if c.startswith(("lag_", "rolling_", "ewm_",
                    "sin_", "cos_", "week_of_year", "month", "quarter", "year", "price_lag", "promo_lag"))]

    for fold_idx, origin_idx in enumerate(fold_origins):
        origin_week = weeks[origin_idx]
        test_weeks = weeks[origin_idx + 1: origin_idx + 1 + HORIZON]

        if len(test_weeks) == 0:
            continue

        train_mask = src_df["week"] <= origin_week
        test_mask = src_df["week"].isin(test_weeks)

        train_df = src_df[train_mask]
        test_df = src_df[test_mask]

        for pk in skus:
            pk_train = train_df[train_df["product_key"] == pk].sort_values("week")
            pk_test = test_df[test_df["product_key"] == pk].sort_values("week")

            if len(pk_train) < 13 or len(pk_test) == 0:
                continue

            # Seasonal naive
            sn_pred = seasonal_naive_forecast(pk_train[TARGET], len(pk_test))

            for i, (_, row) in enumerate(pk_test.iterrows()):
                if i < len(sn_pred):
                    all_results.append({
                        "source_dataset": source,
                        "product_key": pk,
                        "fold": fold_idx,
                        "forecast_origin": str(origin_week),
                        "forecast_week": str(row["week"]),
                        "horizon_step": i + 1,
                        "actual": float(row[TARGET]),
                        "seasonal_naive_forecast": float(sn_pred[i]),
                    })

    results_df = pd.DataFrame(all_results)

    if len(results_df) == 0:
        return results_df, {}

    # Compute metrics
    sn_wape = wape(results_df["actual"], results_df["seasonal_naive_forecast"])
    sn_bias = bias(results_df["actual"], results_df["seasonal_naive_forecast"])

    metrics = {
        "source": source,
        "n_folds": len(fold_origins),
        "n_skus": len(skus),
        "n_predictions": len(results_df),
        "seasonal_naive_wape": round(sn_wape * 100, 4) if not np.isnan(sn_wape) else None,
        "seasonal_naive_bias": round(sn_bias, 4),
        "horizon": HORIZON,
    }

    return results_df, metrics


def train_candidate_lightgbm(train_df: pd.DataFrame, feature_cols: list,
                             source: str) -> object:
    """Train a LightGBM candidate model."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not available, skipping candidate")
        return None

    valid_train = train_df.dropna(subset=feature_cols + [TARGET])
    if len(valid_train) < 100:
        return None

    X = valid_train[feature_cols].values
    y = valid_train[TARGET].values

    model = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbose=-1,
    )
    model.fit(X, y)
    return model


def run_forecasting():
    print("=" * 60)
    print("PHASE 17 — FORECASTING & BACKTEST")
    print("=" * 60)
    print(f"Horizon: {HORIZON} weeks")

    feat_path = os.path.join(P17_FEAT, "weekly_features.parquet")
    if not os.path.exists(feat_path):
        print("ERROR: weekly_features.parquet not found. Run feature engineering first.")
        return None

    df = pd.read_parquet(feat_path)
    df["week"] = pd.to_datetime(df["week"])
    sources = sorted(df["source_dataset"].unique())
    print(f"Sources: {sources}")

    feature_cols = [c for c in df.columns if c.startswith(("lag_", "rolling_", "ewm_",
                    "sin_", "cos_")) or c in ("week_of_year", "month", "quarter", "year",
                    "price_lag1", "promo_lag1")]

    all_results = []
    all_metrics = {}
    candidate_registry = []

    for source in sources:
        print(f"\n--- {source} ---")
        src_df = df[df["source_dataset"] == source].copy()

        # Rolling-origin backtest with seasonal naive
        results_df, sn_metrics = rolling_origin_backtest(df, source)

        if len(results_df) == 0:
            print(f"  No backtest results for {source}")
            all_metrics[source] = {"status": "INSUFFICIENT_DATA"}
            continue

        print(f"  Seasonal-naive WAPE: {sn_metrics.get('seasonal_naive_wape')}%")
        print(f"  Seasonal-naive bias: {sn_metrics.get('seasonal_naive_bias')}")

        # Train candidate LightGBM using rolling-origin
        weeks = sorted(src_df["week"].unique())
        n_folds = min(5, max(1, len(weeks) - MIN_HISTORY_WEEKS - HORIZON))
        last_trainable = len(weeks) - HORIZON

        fold_origins = []
        for i in range(n_folds):
            idx = last_trainable - i
            if idx >= MIN_HISTORY_WEEKS:
                fold_origins.append(idx)
        fold_origins = sorted(fold_origins)

        candidate_preds = []
        for fold_idx, origin_idx in enumerate(fold_origins):
            origin_week = weeks[origin_idx]
            test_weeks = weeks[origin_idx + 1: origin_idx + 1 + HORIZON]
            if not test_weeks:
                continue

            train_data = src_df[src_df["week"] <= origin_week].dropna(subset=feature_cols + [TARGET])
            test_data = src_df[src_df["week"].isin(test_weeks)]

            if len(train_data) < 100:
                continue

            model = train_candidate_lightgbm(train_data, feature_cols, source)
            if model is None:
                continue

            for _, row in test_data.iterrows():
                feat_vals = row[feature_cols].values.reshape(1, -1)
                try:
                    pred = max(0, float(model.predict(feat_vals)[0]))
                except Exception:
                    pred = 0.0
                candidate_preds.append({
                    "source_dataset": source,
                    "product_key": row["product_key"],
                    "fold": fold_idx,
                    "forecast_origin": str(origin_week),
                    "forecast_week": str(row["week"]),
                    "actual": float(row[TARGET]),
                    "candidate_forecast": pred,
                })

        # Save final model on full training data
        full_train = src_df.dropna(subset=feature_cols + [TARGET])
        final_model = train_candidate_lightgbm(full_train, feature_cols, source)

        if final_model is not None:
            import joblib
            model_subdir = "uci" if source == "UCI" else "synthetic"
            model_path = os.path.join(MODELS_DIR, model_subdir, f"phase17_{source.lower()}_lightgbm.joblib")
            joblib.dump(final_model, model_path)
            model_hash = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
            print(f"  Candidate model saved: {model_path}")
            print(f"  SHA-256: {model_hash[:20]}...")
        else:
            model_path = None
            model_hash = None

        # Merge candidate predictions with baseline
        if candidate_preds:
            cand_df = pd.DataFrame(candidate_preds)
            merged = results_df.merge(
                cand_df[["source_dataset", "product_key", "fold", "forecast_week", "candidate_forecast"]],
                on=["source_dataset", "product_key", "fold", "forecast_week"],
                how="left"
            )
            cand_wape_val = wape(
                cand_df["actual"], cand_df["candidate_forecast"]
            )
            cand_bias_val = bias(
                cand_df["actual"], cand_df["candidate_forecast"]
            )
        else:
            merged = results_df.copy()
            merged["candidate_forecast"] = np.nan
            cand_wape_val = np.nan
            cand_bias_val = np.nan

        # Select best
        sn_wape_val = sn_metrics.get("seasonal_naive_wape")
        cand_wape_pct = round(cand_wape_val * 100, 4) if not np.isnan(cand_wape_val) else None

        if cand_wape_pct is not None and sn_wape_val is not None and cand_wape_pct < sn_wape_val:
            champion = "LightGBM"
            merged["selected_forecast"] = merged["candidate_forecast"].fillna(merged["seasonal_naive_forecast"])
            improvement = round(sn_wape_val - cand_wape_pct, 4)
        else:
            champion = "SEASONAL_NAIVE"
            merged["selected_forecast"] = merged["seasonal_naive_forecast"]
            improvement = 0.0

        print(f"  Candidate LightGBM WAPE: {cand_wape_pct}%")
        print(f"  Champion: {champion}")

        all_results.append(merged)

        source_metrics = {
            **sn_metrics,
            "candidate_model": "LightGBM",
            "candidate_wape": cand_wape_pct,
            "candidate_bias": round(cand_bias_val, 4) if not np.isnan(cand_bias_val) else None,
            "champion": champion,
            "improvement_pct": improvement,
        }
        all_metrics[source] = source_metrics

        # Registry entry
        candidate_registry.append({
            "model_id": f"phase17_{source.lower()}_lightgbm",
            "source_dataset": source,
            "training_data": f"data/phase17/features/weekly_features.parquet (source={source})",
            "feature_version": "phase17_weekly",
            "forecast_horizon": f"{HORIZON} weeks",
            "validation_method": "rolling_origin",
            "wape": cand_wape_pct,
            "bias": round(cand_bias_val, 4) if not np.isnan(cand_bias_val) else None,
            "baseline_wape": sn_wape_val,
            "improvement_vs_baseline": improvement,
            "artifact_path": model_path,
            "sha256": model_hash,
            "status": "candidate",
            "champion": champion,
        })

    # Save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_parquet(os.path.join(P17_BT, "backtest_results.parquet"), index=False)

    # Save baseline report
    baseline_report_path = os.path.join(P17_BT, "seasonal_naive_results.parquet")
    if all_results:
        sn_cols = ["source_dataset", "product_key", "fold", "forecast_origin",
                   "forecast_week", "horizon_step", "actual", "seasonal_naive_forecast"]
        sn_only = pd.concat(all_results, ignore_index=True)
        available = [c for c in sn_cols if c in sn_only.columns]
        sn_only[available].to_parquet(baseline_report_path, index=False)

    # Save metrics
    metrics_path = os.path.join(P17_FCST, "backtest_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Save candidate registry
    registry_path = os.path.join(DOCS_DIR, "phase17_candidate_model_registry.json")
    with open(registry_path, "w") as f:
        json.dump(candidate_registry, f, indent=2, default=str)

    return all_metrics, candidate_registry


if __name__ == "__main__":
    run_forecasting()
