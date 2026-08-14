"""
Phase 9.3 — Forecast horizon analysis.

Phase 8 LightGBM is a 1-observation-ahead tabular model. It does not emit
direct 3/7/14/30-step forecasts. Multi-step evaluation therefore uses
*iterated recursive* forecasts from the frozen Phase 8 model:

  origin t  ->  predict t+1  ->  feed prediction into lag/rolling buffer
            ->  predict t+2  ...  up to horizon h

This is the same recursive idea as ``src/forecasting.py:generate_multi_step_forecast``,
adapted to Phase 6 feature names. It does NOT copy a 1-day prediction h times.

Horizon unit = observation step at the forecasting grain
(calendar day for SYNTHETIC; next observed date for gappy UCI series).

Exogenous operational fields (price, promo, inventory) are held at origin
values to avoid using future realized demand-correlated fields.
Calendar features of the target date are used (known in advance).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml_forecasting import _predict_nonneg, load_feature_dataset
from src.phase9_common import (
    FIGURES_DIR,
    HORIZONS,
    LAG_PERIODS,
    PHASE9_DIR,
    ROLL_WINDOWS,
    SYN_MODEL_PATH,
    TARGET,
    UCI_MODEL_PATH,
    apply_mpl_style,
    ensure_dirs,
    evaluate_extended,
)

MODEL_PATHS = {"UCI": UCI_MODEL_PATH, "SYNTHETIC": SYN_MODEL_PATH}
ORIGIN_STRIDE = {"UCI": 5, "SYNTHETIC": 7}
MAX_H = max(HORIZONS)


def _load_artifact(source: str) -> dict:
    path = MODEL_PATHS[source]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Phase 8 model missing: {path}")
    obj = joblib.load(path)
    if obj.get("model_name") != "lightgbm":
        raise RuntimeError(f"Expected lightgbm artifact for {source}, got {obj.get('model_name')}")
    return obj


def _lags_from_buffers(buffers: list[list[float]], n: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for lag in LAG_PERIODS:
        arr = np.full(n, np.nan, dtype=float)
        for i, buf in enumerate(buffers):
            if len(buf) >= lag:
                arr[i] = buf[-lag]
        out[f"units_sold_lag_{lag}"] = arr
    for w in ROLL_WINDOWS:
        mean = np.full(n, np.nan, dtype=float)
        std = np.full(n, np.nan, dtype=float)
        for i, buf in enumerate(buffers):
            if not buf:
                continue
            window = buf[-w:]
            mean[i] = float(np.mean(window))
            if len(window) >= 2:
                std[i] = float(np.std(window, ddof=1))
        out[f"rolling_mean_{w}"] = mean
        out[f"rolling_std_{w}"] = std
    lag1 = out.get("units_sold_lag_1")
    lag2 = out.get("units_sold_lag_2")
    lag7 = out.get("units_sold_lag_7")
    lag30 = out.get("units_sold_lag_30")
    lag8 = np.full(n, np.nan, dtype=float)
    for i, buf in enumerate(buffers):
        if len(buf) >= 8:
            lag8[i] = buf[-8]
    if lag1 is not None and lag2 is not None:
        dc1 = np.full(n, np.nan, dtype=float)
        ok = np.isfinite(lag1) & np.isfinite(lag2)
        dc1[ok] = lag1[ok] - lag2[ok]
        out["demand_change_1"] = dc1
    if lag1 is not None:
        dc7 = np.full(n, np.nan, dtype=float)
        ok = np.isfinite(lag1) & np.isfinite(lag8)
        dc7[ok] = lag1[ok] - lag8[ok]
        out["demand_change_7"] = dc7
    rm7 = out.get("rolling_mean_7")
    if rm7 is not None and lag7 is not None:
        g7 = np.full(n, np.nan, dtype=float)
        ok = np.isfinite(rm7) & np.isfinite(lag7)
        g7[ok] = (rm7[ok] - lag7[ok]) / (np.abs(lag7[ok]) + 1.0)
        out["demand_growth_7"] = g7
    rm30 = out.get("rolling_mean_30")
    if rm30 is not None and lag30 is not None:
        g30 = np.full(n, np.nan, dtype=float)
        ok = np.isfinite(rm30) & np.isfinite(lag30)
        g30[ok] = (rm30[ok] - lag30[ok]) / (np.abs(lag30[ok]) + 1.0)
        out["demand_growth_30"] = g30
    return out


def _gather_base_frame(
    live: list[dict],
    series_cols: list[dict[str, np.ndarray]],
    feat_cols: list[str],
    h: int,
) -> pd.DataFrame:
    data: dict[str, list] = {c: [] for c in feat_cols}
    for st in live:
        t_idx = st["origin_idx"] + h
        cols = series_cols[st["series_i"]]
        for c in feat_cols:
            data[c].append(cols[c][t_idx])
    return pd.DataFrame(data)


def run_horizon_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    print(f"[Phase 9.3] Recursive horizons for {source}...")
    art = _load_artifact(source)
    model = art["model"]
    pre = art["preprocessor"]
    numeric = list(art["numeric_features"])
    cats = list(art["categorical_features"])
    feat_cols = numeric + cats

    hold_cols = [
        c for c in feat_cols
        if c in {
            "average_unit_price", "price_lag_1", "price_change", "base_price",
            "discount_pct", "promotion_flag", "promotion_available", "promo_rolling_7",
            "ending_inventory", "on_order_qty", "stockout_flag", "historical_doi",
            "store_size_sqft",
        }
    ]

    src = df[df["source_dataset"] == source].copy()
    src = src.sort_values(["entity_id", "product_key", "date"]).reset_index(drop=True)
    stride = ORIGIN_STRIDE[source]

    series_cols: list[dict[str, np.ndarray]] = []
    series_y: list[np.ndarray] = []
    series_dates: list[np.ndarray] = []
    states: list[dict] = []
    n_eligible = 0
    n_used = 0

    packs = []
    for (ent, pk), g in src.groupby(["entity_id", "product_key"], observed=True, sort=False):
        g = g.reset_index(drop=True)
        test_idx = np.flatnonzero(g["split"].to_numpy() == "test")
        if len(test_idx) <= MAX_H:
            continue
        n_eligible += 1
        packs.append((len(test_idx), ent, pk, g, test_idx))

    packs.sort(key=lambda x: (-x[0], str(x[1]), str(x[2])))
    max_series = 400 if source == "UCI" else None
    if max_series is not None and len(packs) > max_series:
        print(
            f"  {source}: using longest {max_series} of {len(packs)} eligible series "
            f"(deterministic compute cap)"
        )
        packs = packs[:max_series]

    for n_test, ent, pk, g, test_idx in packs:
        si = len(series_y)
        y = g[TARGET].astype(float).to_numpy()
        colmap = {c: g[c].to_numpy() for c in feat_cols}
        series_cols.append(colmap)
        series_y.append(y)
        series_dates.append(pd.to_datetime(g["date"]).to_numpy())
        n_used += 1
        origins = test_idx[::stride]
        origins = origins[origins + 1 < len(g)]
        for oi in origins:
            if oi < 1:
                continue
            exo = {c: colmap[c][oi - 1] if c in colmap else np.nan for c in hold_cols}
            states.append({
                "series_i": si,
                "entity_id": ent,
                "product_key": pk,
                "origin_idx": int(oi - 1),
                "origin_date": series_dates[si][oi - 1],
                "buffer": list(y[:oi]),
                "exo": exo,
            })

    records: list[dict] = []
    for h in range(1, MAX_H + 1):
        live = [st for st in states if st["origin_idx"] + h < len(series_y[st["series_i"]])]
        if not live:
            break
        Xdf = _gather_base_frame(live, series_cols, feat_cols, h)
        dyn = _lags_from_buffers([st["buffer"] for st in live], len(live))
        for col, arr in dyn.items():
            if col in Xdf.columns:
                Xdf[col] = arr
        for c in hold_cols:
            if c in Xdf.columns:
                Xdf[c] = [st["exo"][c] for st in live]
        preds = _predict_nonneg(model, pre.transform(Xdf))
        keep_h = h in HORIZONS
        for j, st in enumerate(live):
            pred = float(preds[j])
            st["buffer"].append(pred)
            if keep_h:
                t_idx = st["origin_idx"] + h
                records.append({
                    "source_dataset": source,
                    "entity_id": st["entity_id"],
                    "product_key": st["product_key"],
                    "origin_date": pd.Timestamp(st["origin_date"]),
                    "target_date": pd.Timestamp(series_dates[st["series_i"]][t_idx]),
                    "horizon": h,
                    "actual_units_sold": float(series_y[st["series_i"]][t_idx]),
                    "predicted_units_sold": pred,
                    "model": "lightgbm_recursive",
                })
        print(f"    {source} h={h}: {len(live):,} origins")

    print(
        f"  {source}: eligible_series={n_eligible}, used_series={n_used}, "
        f"origins={len(states):,}, rows={len(records):,}"
    )
    return pd.DataFrame(records)


def summarize_horizons(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (src, h), g in detail.groupby(["source_dataset", "horizon"], observed=True):
        m = evaluate_extended(
            g["actual_units_sold"].to_numpy(),
            g["predicted_units_sold"].to_numpy(),
            "lightgbm_recursive",
        )
        rows.append({
            "source_dataset": src,
            "horizon": int(h),
            "n": int(len(g)),
            **{k: m[k] for k in ["MAE", "RMSE", "sMAPE", "WAPE", "bias"]},
        })
    return pd.DataFrame(rows).sort_values(["source_dataset", "horizon"])


def create_horizon_charts(summary: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt

    apply_mpl_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = []
    for src in ["UCI", "SYNTHETIC"]:
        sub = summary[summary["source_dataset"] == src].sort_values("horizon")
        if sub.empty:
            continue
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        l1, = ax1.plot(
            sub["horizon"], sub["WAPE"], marker="o", lw=2, color="#1d4ed8", label="WAPE (%)"
        )
        l2, = ax2.plot(
            sub["horizon"], sub["MAE"], marker="s", lw=2, color="#d97706", label="MAE (units)"
        )
        ax1.set_xlabel("Forecast horizon (observation steps)")
        ax1.set_ylabel("WAPE (%)")
        ax2.set_ylabel("MAE (units_sold)")
        ax1.set_title(f"{src} recursive horizon performance (frozen LightGBM)")
        ax1.set_xticks(list(HORIZONS))
        ax2.spines["right"].set_visible(True)
        ax1.legend([l1, l2], ["WAPE (%)", "MAE (units)"], loc="best")
        p = os.path.join(FIGURES_DIR, f"horizon_performance_{src.lower()}.png")
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def run_horizon_analysis(df: pd.DataFrame | None = None, save: bool = True) -> dict[str, Any]:
    ensure_dirs()
    if df is None:
        df = load_feature_dataset()
    parts = []
    for src in ["UCI", "SYNTHETIC"]:
        parts.append(run_horizon_source(df, src))
    detail = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summary = summarize_horizons(detail) if not detail.empty else pd.DataFrame()
    if not summary.empty:
        print(summary.to_string(index=False))
    charts = []
    if save and not detail.empty:
        detail.to_parquet(os.path.join(PHASE9_DIR, "horizon_detail.parquet"), index=False)
        summary.to_parquet(os.path.join(PHASE9_DIR, "horizon_summary.parquet"), index=False)
        charts = create_horizon_charts(summary)
    return {"detail": detail, "summary": summary, "charts": charts}


if __name__ == "__main__":
    run_horizon_analysis(save=True)
