"""
Phase 10.1b — Intermittent-demand baselines (1-step, per series).

Implemented (SYNTHETIC, where zeros are observed as y=0):
  - Naive: last observed demand
  - Croston (1972): size / interval, common alpha
  - SBA (Syntetos-Boylan): Croston * (1 - alpha/2)
  - TSB (Teunter-Syntetos-Babai): size * occurrence probability

Not implemented for UCI:
  UCI forecasting rows are positive-demand invoice days. Missing calendar days
  are absent rather than coded zero, so Croston-style interval updates are
  not identified from this table.

Evaluation: 1-step on TEST using only history strictly before each origin,
updating after each forecast with the realized actual (standard rolling 1-step).
Alpha/beta = 0.1 (documented default; not test-tuned).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml_forecasting import load_feature_dataset
from src.phase10_common import (
    FIGURES_DIR,
    GRAIN,
    PHASE10_DIR,
    TARGET,
    apply_mpl_style,
    ensure_dirs,
    forecast_metrics,
)

ALPHA = 0.1
BETA = 0.1


def _series_forecasts(y: np.ndarray, split: np.ndarray) -> dict[str, np.ndarray]:
    """y chronological; split is object array of train/validation/test."""
    n = len(y)
    test = split == "test"
    naive = np.full(n, np.nan)
    croston = np.full(n, np.nan)
    sba = np.full(n, np.nan)
    tsb = np.full(n, np.nan)
    last = 0.0
    z, p, pi, q = 0.0, 1.0, 0.0, 0
    for i in range(n):
        if test[i]:
            naive[i] = last
            croston[i] = 0.0 if p <= 0 else z / p
            sba[i] = croston[i] * (1.0 - ALPHA / 2.0)
            tsb[i] = pi * z
        yi = float(y[i])
        last = yi
        if yi > 0:
            q_obs = q + 1
            if z == 0.0 and p == 1.0 and q == 0:
                z = yi
                p = float(max(q_obs, 1))
            else:
                z = z + ALPHA * (yi - z)
                p = p + ALPHA * (q_obs - p)
            p = max(p, 1.0)
            pi = pi + BETA * (1.0 - pi)
            q = 0
        else:
            q = q + 1
            pi = pi + BETA * (0.0 - pi)
    return {"naive": naive, "croston": croston, "sba": sba, "tsb": tsb}


def run_intermittent_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    print(f"[Phase 10.1b] Intermittent baselines {source}...")
    src = df[df["source_dataset"] == source].copy()
    if "units_sold_lag_1" in src.columns:
        src = src[src["units_sold_lag_1"].notna()].copy()
    src = src.sort_values(["entity_id", "product_key", "date"]).reset_index(drop=True)
    recs = []
    n_series = 0
    for (ent, pk), g in src.groupby(["entity_id", "product_key"], observed=True, sort=False):
        g = g.reset_index(drop=True)
        if not (g["split"] == "test").any():
            continue
        n_series += 1
        y = g[TARGET].astype(float).to_numpy()
        sp = g["split"].astype(str).to_numpy()
        fc = _series_forecasts(y, sp)
        test_idx = np.flatnonzero(sp == "test")
        for i in test_idx:
            recs.append({
                "source_dataset": source,
                "entity_id": ent,
                "product_key": pk,
                "date": g.loc[i, "date"],
                "actual_units_sold": y[i],
                "naive": fc["naive"][i],
                "croston": fc["croston"][i],
                "sba": fc["sba"][i],
                "tsb": fc["tsb"][i],
            })
    print(f"  {source}: {n_series} series, {len(recs):,} test rows")
    return pd.DataFrame(recs)


def summarize_baselines(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, g in detail.groupby("source_dataset", observed=True):
        y = g["actual_units_sold"].to_numpy()
        for name in ["naive", "croston", "sba", "tsb"]:
            m = forecast_metrics(y, g[name].to_numpy(), name)
            rows.append({
                "source_dataset": src,
                "model": name,
                "alpha": ALPHA,
                "beta": BETA if name == "tsb" else np.nan,
                **{k: m[k] for k in [
                    "MAE", "RMSE", "sMAPE", "WAPE", "bias", "n",
                    "zero_mae", "nonzero_mae", "zero_positive_prediction_rate",
                ]},
            })
    return pd.DataFrame(rows)


def create_baseline_chart(summary: pd.DataFrame, hurdle_wape: float | None) -> list[str]:
    import matplotlib.pyplot as plt

    apply_mpl_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    syn = summary[summary["source_dataset"] == "SYNTHETIC"]
    if syn.empty:
        return []
    fig, ax = plt.subplots()
    models = syn["model"].tolist()
    wape = syn["WAPE"].tolist()
    colors = ["#94a3b8"] * len(models)
    ax.bar(models, wape, color=colors)
    if hurdle_wape is not None and np.isfinite(hurdle_wape):
        ax.axhline(hurdle_wape, color="#1d4ed8", ls="--", lw=1.4, label=f"Hurdle WAPE={hurdle_wape:.2f}")
        ax.legend()
    ax.set_ylabel("TEST WAPE (%)")
    ax.set_xlabel("Intermittent baseline")
    ax.set_title("SYNTHETIC intermittent baselines vs hurdle (1-step TEST)")
    p = os.path.join(FIGURES_DIR, "intermittent_baseline_comparison.png")
    fig.savefig(p)
    plt.close(fig)
    return [p]


def run_intermittent_baselines(
    df: pd.DataFrame | None = None,
    save: bool = True,
    hurdle_wape: float | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    if df is None:
        df = load_feature_dataset()
    # Only SYNTHETIC has coded zeros; still run UCI to document naive.
    parts = []
    skip_notes = {}
    src_syn = df[df["source_dataset"] == "SYNTHETIC"]
    zshare = float((src_syn[TARGET] == 0).mean()) if len(src_syn) else 0.0
    if zshare >= 0.05:
        parts.append(run_intermittent_source(df, "SYNTHETIC"))
    else:
        skip_notes["SYNTHETIC"] = "Insufficient zeros"
    src_uci = df[df["source_dataset"] == "UCI"]
    z_uci = float((src_uci[TARGET] == 0).mean()) if len(src_uci) else 0.0
    skip_notes["UCI"] = (
        f"UCI zero-row share={100*z_uci:.2f}%. Croston/SBA/TSB require observed zero "
        "demand intervals; UCI grain omits no-sale days, so intermittent baselines "
        "are not identified. Naive last-observation remains available via Phase 7."
    )
    print(f"  SKIP UCI intermittent: {skip_notes['UCI']}")
    detail = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summary = summarize_baselines(detail) if not detail.empty else pd.DataFrame()
    if not summary.empty:
        print(summary.to_string(index=False))
    charts = create_baseline_chart(summary, hurdle_wape) if save else []
    if save and not detail.empty:
        detail.to_parquet(os.path.join(PHASE10_DIR, "intermittent_detail.parquet"), index=False)
        summary.to_parquet(os.path.join(PHASE10_DIR, "intermittent_summary.parquet"), index=False)
    return {"detail": detail, "summary": summary, "charts": charts, "skip_notes": skip_notes}


if __name__ == "__main__":
    run_intermittent_baselines(save=True)
