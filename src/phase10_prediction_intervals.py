"""
Phase 10.3 — Quantile LightGBM prediction intervals (P10 / P50 / P90).

1-step, same Phase 8 feature contract and chronological splits.
Non-negative post-process: clip quantiles at 0 (demand cannot be negative);
documented, not silent. If clipping induces P10>P50 or P50>P90, reorder
quantiles after clip (isotonic on the three points) and record the count.
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
    feature_lists_for_source,
    load_feature_dataset,
    prepare_features,
)
from src.phase10_common import (
    FIGURES_DIR,
    GRAIN,
    LGB_POINT_PARAMS,
    PHASE10_DIR,
    QUANTILES,
    TARGET,
    apply_mpl_style,
    ensure_dirs,
    pinball_loss,
)


def _usable(df_src: pd.DataFrame) -> pd.DataFrame:
    if "units_sold_lag_1" in df_src.columns:
        df_src = df_src[df_src["units_sold_lag_1"].notna()].copy()
    return df_src


def _enforce_nonneg_and_order(p10, p50, p90) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    p10 = np.maximum(0.0, np.asarray(p10, dtype=float))
    p50 = np.maximum(0.0, np.asarray(p50, dtype=float))
    p90 = np.maximum(0.0, np.asarray(p90, dtype=float))
    n_neg_clipped = None  # computed by caller from raw
    stacked = np.vstack([p10, p50, p90])
    ordered = np.sort(stacked, axis=0)
    n_cross = int(np.sum((p10 > p50) | (p50 > p90) | (p10 > p90)))
    return ordered[0], ordered[1], ordered[2], {"n_crossed_before_reorder": n_cross}


def run_intervals_source(df: pd.DataFrame, source: str) -> dict[str, Any]:
    print(f"[Phase 10.3] Quantile intervals {source}...")
    df_src, numeric, cats = prepare_features(df, source)
    df_src = _usable(df_src)
    feat = numeric + cats
    train = df_src[df_src["split"] == "train"]
    val = df_src[df_src["split"] == "validation"]
    test = df_src[df_src["split"] == "test"]
    pre = FeaturePreprocessor(numeric, cats, impute=False)
    X_train = pre.fit_transform(train[feat])
    y_train = train[TARGET].astype(float).to_numpy()
    models = {}
    t0 = time.perf_counter()
    for tau in QUANTILES:
        params = dict(LGB_POINT_PARAMS)
        params["objective"] = "quantile"
        params["alpha"] = tau
        m = lgb.LGBMRegressor(**params)
        m.fit(X_train, y_train, feature_name=pre.feature_names_)
        models[tau] = m
        print(f"    fitted q={tau}")
    train_s = time.perf_counter() - t0

    def predict_raw(part: pd.DataFrame) -> dict[str, np.ndarray]:
        X = pre.transform(part[feat])
        return {tau: np.asarray(models[tau].predict(X), dtype=float) for tau in QUANTILES}

    raw_test = predict_raw(test)
    n_neg = {
        f"p{int(tau*100)}": int(np.sum(raw_test[tau] < 0)) for tau in QUANTILES
    }
    p10, p50, p90, extra = _enforce_nonneg_and_order(
        raw_test[0.10], raw_test[0.50], raw_test[0.90]
    )
    y = test[TARGET].astype(float).to_numpy()
    inside = (y >= p10) & (y <= p90)
    coverage = float(np.mean(inside) * 100.0)
    width = float(np.mean(p90 - p10))
    pin = {tau: round(pinball_loss(y, q, tau), 4)
           for tau, q in [(0.10, p10), (0.50, p50), (0.90, p90)]}
    # calibration: share of actuals below each quantile (should ~ tau)
    cal = {
        f"p{int(tau*100)}_below_pct": round(100.0 * float(np.mean(y <= q)), 2)
        for tau, q in [(0.10, p10), (0.50, p50), (0.90, p90)]
    }
    pred_df = test[GRAIN].copy()
    pred_df["actual_units_sold"] = y
    pred_df["p10"] = p10
    pred_df["p50"] = p50
    pred_df["p90"] = p90
    pred_df["interval_width"] = p90 - p10
    pred_df["in_interval"] = inside
    pred_df["nonneg_clip"] = True
    pred_df["quantile_reordered"] = extra["n_crossed_before_reorder"] > 0
    print(
        f"  {source} coverage={coverage:.2f}% width={width:.3f} "
        f"pinball={pin} crossed={extra['n_crossed_before_reorder']}"
    )
    return {
        "source_dataset": source,
        "training_time_sec": round(train_s, 3),
        "n_test": int(len(test)),
        "coverage_pct": round(coverage, 4),
        "mean_width": round(width, 4),
        "pinball": pin,
        "calibration": cal,
        "n_negative_raw": n_neg,
        "n_crossed_before_reorder": extra["n_crossed_before_reorder"],
        "nonneg_clip": True,
        "predictions": pred_df,
        "val_n": int(len(val)),
        "train_n": int(len(train)),
    }


def create_interval_charts(results: dict[str, dict]) -> list[str]:
    import matplotlib.pyplot as plt

    apply_mpl_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = []
    fig, ax = plt.subplots()
    srcs, cov = [], []
    for src in ["UCI", "SYNTHETIC"]:
        if src not in results:
            continue
        srcs.append(src)
        cov.append(results[src]["coverage_pct"])
    ax.bar(srcs, cov, color=["#1d4ed8", "#059669"][:len(srcs)])
    ax.axhline(80.0, color="#111827", ls="--", lw=1.2, label="nominal 80% (P10-P90)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("P10-P90 interval coverage on TEST")
    ax.legend()
    p = os.path.join(FIGURES_DIR, "prediction_interval_coverage.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots()
    widths = [results[s]["mean_width"] for s in srcs]
    ax.bar(srcs, widths, color=["#1d4ed8", "#059669"][:len(srcs)])
    ax.set_ylabel("Mean (P90 - P10) in units_sold")
    ax.set_title("Mean prediction-interval width on TEST")
    p = os.path.join(FIGURES_DIR, "prediction_interval_width.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots()
    x = np.arange(3)
    w = 0.35
    taus = [10, 50, 90]
    for i, src in enumerate(srcs):
        cal = results[src]["calibration"]
        vals = [cal[f"p{t}_below_pct"] for t in taus]
        ax.bar(x + (i - 0.5) * w, vals, width=w, label=src)
    ax.plot(x, taus, "k--", lw=1.2, marker="o", label="nominal tau %")
    ax.set_xticks(x, ["P10", "P50", "P90"])
    ax.set_ylabel("% of actuals <= quantile")
    ax.set_title("Quantile calibration on TEST")
    ax.legend()
    p = os.path.join(FIGURES_DIR, "quantile_calibration.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)
    return paths


def run_prediction_intervals(df: pd.DataFrame | None = None, save: bool = True) -> dict[str, Any]:
    ensure_dirs()
    if df is None:
        df = load_feature_dataset()
    results = {}
    parts = []
    rows = []
    for src in ["UCI", "SYNTHETIC"]:
        results[src] = run_intervals_source(df, src)
        parts.append(results[src]["predictions"])
        r = results[src]
        rows.append({
            "source_dataset": src,
            "n_test": r["n_test"],
            "coverage_pct": r["coverage_pct"],
            "mean_width": r["mean_width"],
            "pinball_p10": r["pinball"][0.10],
            "pinball_p50": r["pinball"][0.50],
            "pinball_p90": r["pinball"][0.90],
            **r["calibration"],
            "n_crossed_before_reorder": r["n_crossed_before_reorder"],
            "n_negative_raw_p10": r["n_negative_raw"]["p10"],
            "n_negative_raw_p50": r["n_negative_raw"]["p50"],
            "n_negative_raw_p90": r["n_negative_raw"]["p90"],
            "nonneg_clip": True,
            "training_time_sec": r["training_time_sec"],
        })
    summary = pd.DataFrame(rows)
    preds = pd.concat(parts, ignore_index=True)
    charts = create_interval_charts(results) if save else []
    if save:
        preds.to_parquet(os.path.join(PHASE10_DIR, "quantile_predictions.parquet"), index=False)
        summary.to_parquet(os.path.join(PHASE10_DIR, "quantile_summary.parquet"), index=False)
    return {"results": results, "summary": summary, "predictions": preds, "charts": charts}


if __name__ == "__main__":
    run_prediction_intervals(save=True)
