"""Accuracy metrics used by monitoring. Same definitions as Phases 8–11."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.phase10_common import forecast_metrics
from src.phase9_residual_analysis import demand_regime


def accuracy_table(df: pd.DataFrame, actual_col: str = "actual", pred_col: str = "prediction") -> dict:
    mask = df[actual_col].notna() & df[pred_col].notna()
    sub = df.loc[mask]
    if sub.empty:
        return {"n_with_actuals": 0, "metrics": None, "note": "No actuals available"}
    m = forecast_metrics(sub[actual_col].to_numpy(), sub[pred_col].to_numpy(), "monitor")
    return {"n_with_actuals": int(len(sub)), "metrics": m}


def by_group(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    rows = []
    if "actual" not in df.columns:
        return rows
    work = df[df["actual"].notna()].copy()
    if work.empty:
        return rows
    for keys, g in work.groupby(cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(cols, keys))
        rec.update(forecast_metrics(g["actual"].to_numpy(), g["prediction"].to_numpy(), "g"))
        rows.append(rec)
    return rows


def by_regime(df: pd.DataFrame) -> list[dict]:
    if "actual" not in df.columns or df["actual"].notna().sum() == 0:
        return []
    g = df[df["actual"].notna()].copy()
    g["regime"] = demand_regime(g["actual"])
    return by_group(g, ["source_dataset", "regime"] if "source_dataset" in g.columns else ["regime"])
