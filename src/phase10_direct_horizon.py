"""
Phase 10.2 — Direct multi-horizon LightGBM.

For horizon h, origin-row features at time t predict units_sold at t+h
(observation steps within entity_id + product_key).

Known-in-advance target calendar fields (date, holiday, season) are taken
from the target row. Demand lags/rolling/price/inventory remain at origin t.

Leakage control:
  train: origin in train AND target_date <= train_end
  validation: origin in validation AND target_date <= validation_end
  test: origin in test AND target exists

Does not overwrite Phase 8/9 artifacts.
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
    feature_lists_for_source,
    load_feature_dataset,
)
from src.phase10_common import (
    CALENDAR_COLS,
    FIGURES_DIR,
    GRAIN,
    HORIZONS,
    LGB_POINT_PARAMS,
    PHASE10_DIR,
    PHASE9_HORIZON,
    TARGET,
    apply_mpl_style,
    ensure_dirs,
    forecast_metrics,
    split_end_dates,
)

HCAL_PREFIX = "hcal_"


def add_horizon_target(df_src: pd.DataFrame, h: int) -> pd.DataFrame:
    df = df_src.sort_values(["entity_id", "product_key", "date"]).copy()
    g = df.groupby(["entity_id", "product_key"], sort=False, observed=True)
    df["target"] = g[TARGET].shift(-h)
    df["target_date"] = g["date"].shift(-h)
    for c in CALENDAR_COLS:
        if c in df.columns:
            df[f"{HCAL_PREFIX}{c}"] = g[c].shift(-h)
    df["horizon"] = h
    return df


def _masks(df_h: pd.DataFrame, ends: dict) -> dict[str, pd.Series]:
    has = df_h["target"].notna() & df_h["target_date"].notna()
    train_end = ends["train"]
    val_end = ends["validation"]
    return {
        "train": has & (df_h["split"] == "train") & (df_h["target_date"] <= train_end),
        "validation": has & (df_h["split"] == "validation") & (df_h["target_date"] <= val_end),
        "test": has & (df_h["split"] == "test"),
    }


def _feature_lists(df_h: pd.DataFrame, source: str) -> tuple[list[str], list[str]]:
    lists = feature_lists_for_source(source)
    numeric = [c for c in lists["numeric"] if c in df_h.columns]
    cats = [c for c in lists["categorical"] if c in df_h.columns]
    # replace origin calendar with target calendar where present
    numeric = [c for c in numeric if c not in CALENDAR_COLS]
    cats = [c for c in cats if c not in CALENDAR_COLS]
    for c in CALENDAR_COLS:
        hc = f"{HCAL_PREFIX}{c}"
        if hc not in df_h.columns:
            continue
        if c == "season":
            cats.append(hc)
        else:
            numeric.append(hc)
    return numeric, cats


def train_direct_horizon(df_src: pd.DataFrame, source: str, h: int) -> dict[str, Any]:
    ends = split_end_dates(df_src)
    df_h = add_horizon_target(df_src, h)
    masks = _masks(df_h, ends)
    numeric, cats = _feature_lists(df_h, source)
    feat = numeric + cats
    train = df_h.loc[masks["train"]]
    val = df_h.loc[masks["validation"]]
    test = df_h.loc[masks["test"]]
    print(
        f"    {source} h={h}: train={len(train):,} val={len(val):,} test={len(test):,}"
    )
    if len(train) < 500 or len(test) < 50:
        return {
            "source_dataset": source, "horizon": h, "skipped": True,
            "reason": f"insufficient rows train={len(train)} test={len(test)}",
        }
    pre = FeaturePreprocessor(numeric, cats, impute=False)
    t0 = time.perf_counter()
    X_train = pre.fit_transform(train[feat])
    y_train = train["target"].astype(float).to_numpy()
    model = lgb.LGBMRegressor(**LGB_POINT_PARAMS)
    model.fit(X_train, y_train, feature_name=pre.feature_names_)
    train_s = time.perf_counter() - t0
    val_pred = _predict_nonneg(model, pre.transform(val[feat])) if len(val) else np.array([])
    test_pred = _predict_nonneg(model, pre.transform(test[feat]))
    y_val = val["target"].astype(float).to_numpy() if len(val) else np.array([])
    y_test = test["target"].astype(float).to_numpy()
    val_m = forecast_metrics(y_val, val_pred, f"direct_h{h}") if len(val) else {}
    test_m = forecast_metrics(y_test, test_pred, f"direct_h{h}")
    pred_df = test[GRAIN].copy()
    pred_df["origin_date"] = test["date"].values
    pred_df["target_date"] = test["target_date"].values
    pred_df["horizon"] = h
    pred_df["actual_units_sold"] = y_test
    pred_df["predicted_units_sold"] = test_pred
    pred_df["model"] = "lightgbm_direct"
    # leakage flags for validation
    leak_train = bool((train["target_date"] > ends["train"]).any()) if len(train) else False
    leak_val = bool((val["target_date"] > ends["validation"]).any()) if len(val) else False
    return {
        "source_dataset": source,
        "horizon": h,
        "skipped": False,
        "train_n": int(len(train)),
        "val_n": int(len(val)),
        "test_n": int(len(test)),
        "training_time_sec": round(train_s, 3),
        "val_metrics": val_m,
        "test_metrics": test_m,
        "predictions": pred_df,
        "train_target_within_train": not leak_train,
        "val_target_within_val": not leak_val,
        "origin_precedes_target": bool((test["date"] < test["target_date"]).all()) if len(test) else True,
        "numeric": numeric,
        "categorical": cats,
    }


def run_direct_source(df: pd.DataFrame, source: str) -> dict[str, Any]:
    print(f"[Phase 10.2] Direct horizons {source}...")
    src = df[df["source_dataset"] == source].copy()
    if "units_sold_lag_1" in src.columns:
        src = src[src["units_sold_lag_1"].notna()].copy()
    by_h = {}
    for h in HORIZONS:
        by_h[h] = train_direct_horizon(src, source, h)
        if not by_h[h].get("skipped"):
            tm = by_h[h]["test_metrics"]
            print(f"      TEST WAPE={tm['WAPE']:.4f} MAE={tm['MAE']:.4f}")
    return by_h


def compare_recursive(direct_all: dict) -> pd.DataFrame:
    if not os.path.exists(PHASE9_HORIZON):
        return pd.DataFrame()
    rec = pd.read_parquet(PHASE9_HORIZON)
    rows = []
    for src in ["UCI", "SYNTHETIC"]:
        for h in HORIZONS:
            d = direct_all.get(src, {}).get(h, {})
            if d.get("skipped") or not d:
                continue
            r = rec[(rec["source_dataset"] == src) & (rec["horizon"] == h)]
            rec_wape = float(r["WAPE"].iloc[0]) if len(r) else np.nan
            rec_mae = float(r["MAE"].iloc[0]) if len(r) else np.nan
            dw = d["test_metrics"]["WAPE"]
            dm = d["test_metrics"]["MAE"]
            rows.append({
                "source_dataset": src,
                "horizon": h,
                "recursive_WAPE": rec_wape,
                "direct_WAPE": dw,
                "wape_improvement_pct": round((rec_wape - dw) / rec_wape * 100.0, 4)
                if rec_wape and rec_wape == rec_wape else np.nan,
                "recursive_MAE": rec_mae,
                "direct_MAE": dm,
                "recursive_sMAPE": float(r["sMAPE"].iloc[0]) if len(r) else np.nan,
                "direct_sMAPE": d["test_metrics"]["sMAPE"],
                "recursive_bias": float(r["bias"].iloc[0]) if len(r) else np.nan,
                "direct_bias": d["test_metrics"]["bias"],
                "direct_n": d["test_n"],
                "recursive_n": int(r["n"].iloc[0]) if len(r) else np.nan,
                "note": "Populations differ: Phase 9 recursive used strided origins "
                        "(UCI 400-series cap); direct uses all valid test origins.",
            })
    return pd.DataFrame(rows)


def create_horizon_charts(cmp_df: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt

    apply_mpl_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = []
    if cmp_df.empty:
        return paths
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, src in zip(axes, ["UCI", "SYNTHETIC"]):
        sub = cmp_df[cmp_df["source_dataset"] == src].sort_values("horizon")
        if sub.empty:
            continue
        ax.plot(sub["horizon"], sub["recursive_WAPE"], marker="o", lw=2, color="#94a3b8", label="Phase 9 recursive")
        ax.plot(sub["horizon"], sub["direct_WAPE"], marker="s", lw=2, color="#1d4ed8", label="Phase 10 direct")
        ax.set_xlabel("Horizon (observation steps)")
        ax.set_ylabel("WAPE (%)")
        ax.set_title(f"{src} direct vs recursive WAPE")
        ax.set_xticks(list(HORIZONS))
        ax.legend()
    fig.suptitle("Direct multi-horizon vs Phase 9 recursive (see n differences in table)")
    fig.tight_layout()
    p = os.path.join(FIGURES_DIR, "direct_vs_recursive_horizon.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots()
    for src, color in [("UCI", "#1d4ed8"), ("SYNTHETIC", "#059669")]:
        sub = cmp_df[cmp_df["source_dataset"] == src].sort_values("horizon")
        if sub.empty:
            continue
        ax.plot(sub["horizon"], sub["wape_improvement_pct"], marker="o", lw=2, color=color, label=src)
    ax.axhline(0, color="#111827", ls="--", lw=1)
    ax.set_xlabel("Horizon (observation steps)")
    ax.set_ylabel("WAPE improvement vs recursive (%)")
    ax.set_title("Direct-model WAPE improvement (positive = direct better)")
    ax.set_xticks(list(HORIZONS))
    ax.legend()
    p = os.path.join(FIGURES_DIR, "horizon_improvement.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)
    return paths


def run_direct_horizon(df: pd.DataFrame | None = None, save: bool = True) -> dict[str, Any]:
    ensure_dirs()
    if df is None:
        df = load_feature_dataset()
    all_h = {}
    pred_parts = []
    summary_rows = []
    for src in ["UCI", "SYNTHETIC"]:
        all_h[src] = run_direct_source(df, src)
        for h, rec in all_h[src].items():
            if rec.get("skipped"):
                continue
            pred_parts.append(rec["predictions"])
            tm = rec["test_metrics"]
            summary_rows.append({
                "source_dataset": src,
                "horizon": h,
                "train_n": rec["train_n"],
                "val_n": rec["val_n"],
                "test_n": rec["test_n"],
                "training_time_sec": rec["training_time_sec"],
                "train_target_within_train": rec["train_target_within_train"],
                "val_target_within_val": rec["val_target_within_val"],
                "origin_precedes_target": rec["origin_precedes_target"],
                **{k: tm[k] for k in ["MAE", "RMSE", "sMAPE", "WAPE", "bias"]},
            })
    summary = pd.DataFrame(summary_rows)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    cmp_df = compare_recursive(all_h)
    print(cmp_df.to_string(index=False) if not cmp_df.empty else "no recursive comparison")
    charts = create_horizon_charts(cmp_df) if save else []
    if save:
        if not preds.empty:
            preds.to_parquet(os.path.join(PHASE10_DIR, "direct_horizon_predictions.parquet"), index=False)
        summary.to_parquet(os.path.join(PHASE10_DIR, "direct_horizon_summary.parquet"), index=False)
        if not cmp_df.empty:
            cmp_df.to_parquet(os.path.join(PHASE10_DIR, "direct_vs_recursive.parquet"), index=False)
    return {"by_source": all_h, "summary": summary, "predictions": preds, "comparison": cmp_df, "charts": charts}


if __name__ == "__main__":
    run_direct_horizon(save=True)
