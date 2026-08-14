"""
Phase 9.1 — Walk-forward / expanding-window temporal validation.

Retrains the Phase 8 LightGBM *configuration* on chronological folds.
Does NOT modify saved Phase 8 models or prediction files.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml_forecasting import (
    FeaturePreprocessor,
    _predict_nonneg,
    feature_lists_for_source,
    load_feature_dataset,
    train_lightgbm,
)
from src.phase9_common import (
    FIGURES_DIR,
    FOLDS,
    PHASE9_DIR,
    RANDOM_STATE,
    TARGET,
    apply_mpl_style,
    classify_stability,
    ensure_dirs,
    evaluate_extended,
    fold_summary,
)


def _split_fold(df_src: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(spec["train_end"])
    val_start = pd.Timestamp(spec["val_start"])
    val_end = pd.Timestamp(spec["val_end"])
    train = df_src[df_src["date"] <= train_end]
    val = df_src[(df_src["date"] >= val_start) & (df_src["date"] <= val_end)]
    return train, val


def run_walk_forward_source(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict]:
    """Expanding-window LightGBM evaluation for one source."""
    print(f"[Phase 9.1] Walk-forward {source}...")
    df_src = df[df["source_dataset"] == source].copy()
    if "units_sold_lag_1" in df_src.columns:
        df_src = df_src[df_src["units_sold_lag_1"].notna()].copy()
    lists = feature_lists_for_source(source)
    numeric = [c for c in lists["numeric"] if c in df_src.columns]
    cats = [c for c in lists["categorical"] if c in df_src.columns]
    feat_cols = numeric + cats

    rows = []
    for spec in FOLDS[source]:
        train, val = _split_fold(df_src, spec)
        leak_ok = (
            len(train) > 0
            and len(val) > 0
            and train["date"].max() < val["date"].min()
            and set(train.index).isdisjoint(set(val.index))
        )
        if not leak_ok:
            raise RuntimeError(
                f"{source} fold {spec['fold']} failed chronology/leakage checks"
            )
        pre = FeaturePreprocessor(numeric, cats, impute=False)
        t0 = time.perf_counter()
        X_train = pre.fit_transform(train[feat_cols])
        y_train = train[TARGET].astype(float).to_numpy()
        model = train_lightgbm(X_train, y_train, pre.feature_names_)
        train_s = time.perf_counter() - t0
        pred = _predict_nonneg(model, pre.transform(val[feat_cols]))
        y_val = val[TARGET].astype(float).to_numpy()
        m = evaluate_extended(y_val, pred, model_name="lightgbm")
        rec = {
            "source_dataset": source,
            "fold": spec["fold"],
            "train_end": spec["train_end"],
            "val_start": spec["val_start"],
            "val_end": spec["val_end"],
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "train_start_actual": str(train["date"].min().date()),
            "train_end_actual": str(train["date"].max().date()),
            "val_start_actual": str(val["date"].min().date()),
            "val_end_actual": str(val["date"].max().date()),
            "train_precedes_val": bool(train["date"].max() < val["date"].min()),
            "training_time_sec": round(train_s, 3),
            "random_state": RANDOM_STATE,
            **{k: m[k] for k in [
                "MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "n",
                "bias", "overprediction_pct", "underprediction_pct",
            ]},
        }
        rows.append(rec)
        print(
            f"  fold {spec['fold']}: train {rec['train_rows']:,} -> val {rec['val_rows']:,} "
            f"WAPE={rec['WAPE']:.4f} MAE={rec['MAE']:.4f} bias={rec['bias']:.4f}"
        )

    fold_df = pd.DataFrame(rows)
    stability = classify_stability(fold_df["WAPE"].tolist())
    summary = {
        "source_dataset": source,
        "n_folds": int(len(fold_df)),
        "stability": stability,
        "MAE": fold_summary(fold_df, "MAE"),
        "RMSE": fold_summary(fold_df, "RMSE"),
        "sMAPE": fold_summary(fold_df, "sMAPE"),
        "WAPE": fold_summary(fold_df, "WAPE"),
        "bias": fold_summary(fold_df, "bias"),
        "overprediction_pct": fold_summary(fold_df, "overprediction_pct"),
        "underprediction_pct": fold_summary(fold_df, "underprediction_pct"),
    }
    print(f"  {source} stability: {stability['label']} ({stability['reason']})")
    return fold_df, summary


def create_walk_forward_charts(fold_df: pd.DataFrame, out_dir: str = FIGURES_DIR) -> list[str]:
    import matplotlib.pyplot as plt

    apply_mpl_style()
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    fig, ax = plt.subplots()
    for src, color in [("UCI", "#1d4ed8"), ("SYNTHETIC", "#059669")]:
        sub = fold_df[fold_df["source_dataset"] == src].sort_values("fold")
        ax.plot(sub["fold"], sub["WAPE"], marker="o", lw=2, color=color, label=src)
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Walk-forward WAPE by fold")
    ax.set_xticks(sorted(fold_df["fold"].unique()))
    ax.legend()
    p = os.path.join(out_dir, "walk_forward_wape.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, src, color in zip(axes, ["UCI", "SYNTHETIC"], ["#1d4ed8", "#059669"]):
        sub = fold_df[fold_df["source_dataset"] == src].sort_values("fold")
        ax.plot(sub["fold"], sub["MAE"], marker="o", lw=2, color=color)
        ax.set_xlabel("Walk-forward fold")
        ax.set_ylabel("MAE (units_sold)")
        ax.set_title(f"{src} walk-forward MAE")
        ax.set_xticks(sorted(fold_df["fold"].unique()))
    fig.suptitle("Walk-forward MAE by fold (source-specific scale)")
    fig.tight_layout()
    p = os.path.join(out_dir, "walk_forward_mae.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for col, src in enumerate(["UCI", "SYNTHETIC"]):
        sub = fold_df[fold_df["source_dataset"] == src]
        axes[0, col].boxplot(
            sub["WAPE"].to_numpy(), tick_labels=["WAPE"], patch_artist=True,
            boxprops=dict(facecolor="#bfdbfe"),
        )
        axes[0, col].set_title(f"{src} fold WAPE (%)")
        axes[0, col].set_ylabel("WAPE (%)")
        axes[1, col].boxplot(
            sub["MAE"].to_numpy(), tick_labels=["MAE"], patch_artist=True,
            boxprops=dict(facecolor="#fde68a"),
        )
        axes[1, col].set_title(f"{src} fold MAE (units_sold)")
        axes[1, col].set_ylabel("MAE (units_sold)")
    fig.suptitle("Fold stability (distribution across walk-forward folds)")
    fig.tight_layout()
    p = os.path.join(out_dir, "fold_stability.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)
    return paths


def summaries_from_folds(fold_df: pd.DataFrame) -> dict[str, Any]:
    summaries = {}
    for src in ["UCI", "SYNTHETIC"]:
        sub = fold_df[fold_df["source_dataset"] == src]
        stability = classify_stability(sub["WAPE"].tolist())
        summaries[src] = {
            "source_dataset": src,
            "n_folds": int(len(sub)),
            "stability": stability,
            "MAE": fold_summary(sub, "MAE"),
            "RMSE": fold_summary(sub, "RMSE"),
            "sMAPE": fold_summary(sub, "sMAPE"),
            "WAPE": fold_summary(sub, "WAPE"),
            "bias": fold_summary(sub, "bias"),
            "overprediction_pct": fold_summary(sub, "overprediction_pct"),
            "underprediction_pct": fold_summary(sub, "underprediction_pct"),
        }
    return summaries


def load_walk_forward(refresh_charts: bool = True) -> dict[str, Any]:
    path = os.path.join(PHASE9_DIR, "walk_forward_folds.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    fold_df = pd.read_parquet(path)
    summaries = summaries_from_folds(fold_df)
    charts = create_walk_forward_charts(fold_df) if refresh_charts else []
    return {"folds": fold_df, "summaries": summaries, "charts": charts}


def run_walk_forward(df: pd.DataFrame | None = None, save: bool = True) -> dict[str, Any]:
    ensure_dirs()
    if df is None:
        df = load_feature_dataset()
    parts, summaries = [], {}
    for src in ["UCI", "SYNTHETIC"]:
        fold_df, summary = run_walk_forward_source(df, src)
        parts.append(fold_df)
        summaries[src] = summary
    all_folds = pd.concat(parts, ignore_index=True)
    charts = []
    if save:
        path = os.path.join(PHASE9_DIR, "walk_forward_folds.parquet")
        all_folds.to_parquet(path, index=False)
        pd.DataFrame([
            {
                "source_dataset": src,
                "stability_label": s["stability"]["label"],
                "cv_wape": s["stability"]["cv_wape"],
                "range_ratio": s["stability"]["range_ratio"],
                "mean_wape": s["WAPE"]["mean"],
                "std_wape": s["WAPE"]["std"],
                "min_wape": s["WAPE"]["min"],
                "max_wape": s["WAPE"]["max"],
                "mean_mae": s["MAE"]["mean"],
                "mean_rmse": s["RMSE"]["mean"],
                "mean_smape": s["sMAPE"]["mean"],
                "mean_bias": s["bias"]["mean"],
                "reason": s["stability"]["reason"],
            }
            for src, s in summaries.items()
        ]).to_parquet(os.path.join(PHASE9_DIR, "walk_forward_summary.parquet"), index=False)
        charts = create_walk_forward_charts(all_folds)
        print(f"[Phase 9.1] Saved {path}")
    return {"folds": all_folds, "summaries": summaries, "charts": charts}


if __name__ == "__main__":
    run_walk_forward(save=True)
