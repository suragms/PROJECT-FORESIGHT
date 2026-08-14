"""
Phase 10.4 — Small time-aware LightGBM hyperparameter search.

Selection: chronological VALIDATION WAPE (then MAE). TEST is evaluated once
after selection. Never shuffled K-fold. Does not overwrite Phase 8 models.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml_forecasting import (
    FeaturePreprocessor,
    _predict_nonneg,
    load_feature_dataset,
    prepare_features,
)
from src.phase10_common import (
    FIGURES_DIR,
    GRAIN,
    LGB_POINT_PARAMS,
    PHASE10_DIR,
    RANDOM_STATE,
    TARGET,
    apply_mpl_style,
    ensure_dirs,
    forecast_metrics,
)

# Frozen Phase 8 config is candidate 0. Remaining candidates vary one/two knobs.
SEARCH_SPACE = [
    dict(LGB_POINT_PARAMS),
    {**LGB_POINT_PARAMS, "num_leaves": 15},
    {**LGB_POINT_PARAMS, "num_leaves": 63},
    {**LGB_POINT_PARAMS, "learning_rate": 0.03, "n_estimators": 250},
    {**LGB_POINT_PARAMS, "learning_rate": 0.10, "n_estimators": 100},
    {**LGB_POINT_PARAMS, "min_child_samples": 50},
    {**LGB_POINT_PARAMS, "reg_lambda": 1.0, "reg_alpha": 0.1},
    {**LGB_POINT_PARAMS, "subsample": 0.7, "colsample_bytree": 0.7, "num_leaves": 63},
]


def _usable(df_src: pd.DataFrame) -> pd.DataFrame:
    if "units_sold_lag_1" in df_src.columns:
        return df_src[df_src["units_sold_lag_1"].notna()].copy()
    return df_src


def run_hpo_source(df: pd.DataFrame, source: str) -> dict[str, Any]:
    print(f"[Phase 10.4] HPO {source} ({len(SEARCH_SPACE)} configs, val WAPE)...")
    df_src, numeric, cats = prepare_features(df, source)
    df_src = _usable(df_src)
    feat = numeric + cats
    train = df_src[df_src["split"] == "train"]
    val = df_src[df_src["split"] == "validation"]
    test = df_src[df_src["split"] == "test"]
    pre = FeaturePreprocessor(numeric, cats, impute=False)
    X_train = pre.fit_transform(train[feat])
    y_train = train[TARGET].astype(float).to_numpy()
    X_val = pre.transform(val[feat])
    y_val = val[TARGET].astype(float).to_numpy()
    X_test = pre.transform(test[feat])
    y_test = test[TARGET].astype(float).to_numpy()

    rows = []
    best_idx, best_wape, best_mae = 0, np.inf, np.inf
    best_model = None
    for i, params in enumerate(SEARCH_SPACE):
        t0 = time.perf_counter()
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, feature_name=pre.feature_names_)
        dt = time.perf_counter() - t0
        pred_val = _predict_nonneg(model, X_val)
        vm = forecast_metrics(y_val, pred_val, f"hpo_{i}")
        is_frozen = params.get("num_leaves") == 31 and params.get("learning_rate") == 0.05 \
            and params.get("n_estimators") == 150 and params.get("min_child_samples", 20) == 20 \
            and params.get("reg_lambda", 0) in (0, 0.0) and params.get("reg_alpha", 0) in (0, 0.0) \
            and params.get("subsample") == 0.8
        rec = {
            "source_dataset": source,
            "config_id": i,
            "is_phase8_config": bool(is_frozen),
            "num_leaves": params.get("num_leaves"),
            "learning_rate": params.get("learning_rate"),
            "n_estimators": params.get("n_estimators"),
            "min_child_samples": params.get("min_child_samples", 20),
            "subsample": params.get("subsample"),
            "colsample_bytree": params.get("colsample_bytree"),
            "reg_alpha": params.get("reg_alpha", 0.0),
            "reg_lambda": params.get("reg_lambda", 0.0),
            "val_WAPE": vm["WAPE"],
            "val_MAE": vm["MAE"],
            "val_RMSE": vm["RMSE"],
            "val_sMAPE": vm["sMAPE"],
            "training_time_sec": round(dt, 3),
        }
        rows.append(rec)
        print(f"    cfg {i}: val WAPE={vm['WAPE']:.4f} MAE={vm['MAE']:.4f} frozen={is_frozen}")
        if (vm["WAPE"] < best_wape) or (vm["WAPE"] == best_wape and vm["MAE"] < best_mae):
            best_wape, best_mae, best_idx, best_model = vm["WAPE"], vm["MAE"], i, model

    grid = pd.DataFrame(rows)
    pred_test = _predict_nonneg(best_model, X_test)
    tm = forecast_metrics(y_test, pred_test, "hpo_best")
    pred_df = test[GRAIN].copy()
    pred_df["actual_units_sold"] = y_test
    pred_df["predicted_units_sold"] = pred_test
    pred_df["model"] = "lightgbm_hpo"
    pred_df["config_id"] = best_idx
    print(f"  selected cfg {best_idx} TEST WAPE={tm['WAPE']:.4f}")
    return {
        "source_dataset": source,
        "grid": grid,
        "best_config_id": int(best_idx),
        "best_is_phase8_config": bool(grid.loc[grid["config_id"] == best_idx, "is_phase8_config"].iloc[0]),
        "test_metrics": tm,
        "predictions": pred_df,
        "random_state": RANDOM_STATE,
    }


def create_hpo_chart(results: dict[str, dict]) -> list[str]:
    import matplotlib.pyplot as plt

    apply_mpl_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, src in zip(axes, ["UCI", "SYNTHETIC"]):
        g = results[src]["grid"].sort_values("config_id")
        colors = ["#059669" if bool(f) else "#1d4ed8" for f in g["is_phase8_config"]]
        ax.bar(g["config_id"].astype(str), g["val_WAPE"], color=colors)
        ax.axhline(
            g.loc[g["config_id"] == results[src]["best_config_id"], "val_WAPE"].iloc[0],
            color="#d97706", ls="--", lw=1.2, label="selected",
        )
        ax.set_xlabel("Config id")
        ax.set_ylabel("Validation WAPE (%)")
        ax.set_title(f"{src} (green = Phase 8 config)")
        ax.legend()
    fig.suptitle("LightGBM HPO on chronological validation WAPE")
    fig.tight_layout()
    p = os.path.join(FIGURES_DIR, "hyperparameter_comparison.png")
    fig.savefig(p)
    plt.close(fig)
    return [p]


def run_hyperparameter_tuning(df: pd.DataFrame | None = None, save: bool = True) -> dict[str, Any]:
    ensure_dirs()
    if df is None:
        df = load_feature_dataset()
    results = {}
    grids = []
    test_rows = []
    for src in ["UCI", "SYNTHETIC"]:
        results[src] = run_hpo_source(df, src)
        grids.append(results[src]["grid"])
        tm = results[src]["test_metrics"]
        test_rows.append({
            "source_dataset": src,
            "best_config_id": results[src]["best_config_id"],
            "best_is_phase8_config": results[src]["best_is_phase8_config"],
            **{k: tm[k] for k in ["MAE", "RMSE", "sMAPE", "WAPE", "bias", "n"]},
        })
        if save:
            results[src]["predictions"].to_parquet(
                os.path.join(PHASE10_DIR, f"hpo_test_predictions_{src.lower()}.parquet"),
                index=False,
            )
    grid = pd.concat(grids, ignore_index=True)
    summary = pd.DataFrame(test_rows)
    charts = create_hpo_chart(results) if save else []
    if save:
        grid.to_parquet(os.path.join(PHASE10_DIR, "hpo_grid.parquet"), index=False)
        summary.to_parquet(os.path.join(PHASE10_DIR, "hpo_summary.parquet"), index=False)
    return {"results": results, "grid": grid, "summary": summary, "charts": charts}


if __name__ == "__main__":
    run_hyperparameter_tuning(save=True)
