"""Data-quality checks for forecast feature batches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.forecasting.schemas import GRAIN, LEAKAGE_FORBIDDEN


def data_quality_report(
    df: pd.DataFrame,
    *,
    required: list[str] | None = None,
    reference_categories: dict[str, set] | None = None,
) -> dict[str, Any]:
    n = int(len(df))
    req = required or []
    missing_cols = [c for c in req if c not in df.columns]
    miss_rates = {}
    for c in df.columns:
        if df[c].isna().any():
            miss_rates[c] = round(100.0 * float(df[c].isna().mean()), 4)
    dups = 0
    grain = [c for c in GRAIN if c in df.columns]
    if len(grain) == len(GRAIN):
        dups = int(df.duplicated(grain).sum())
    invalid = {}
    for c in ["units_sold_lag_1", "average_unit_price"]:
        if c in df.columns:
            n_neg = int((pd.to_numeric(df[c], errors="coerce") < 0).sum())
            if n_neg:
                invalid[c] = n_neg
    cat_changes = {}
    if reference_categories:
        for col, known in reference_categories.items():
            if col not in df.columns:
                continue
            vals = set(df[col].dropna().astype(str))
            unseen = sorted(vals - set(map(str, known)))
            if unseen:
                cat_changes[col] = {
                    "n_unseen_levels": len(unseen),
                    "unseen_rate_pct": round(100.0 * float(df[col].astype(str).isin(unseen).mean()), 4),
                }
    date_gaps = None
    if "date" in df.columns and {"entity_id", "product_key"}.issubset(df.columns):
        s = df.copy()
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        gaps = 0
        checked = 0
        for _, g in s.dropna(subset=["date"]).groupby(["entity_id", "product_key"], observed=True):
            d = g["date"].sort_values().drop_duplicates()
            if len(d) < 2:
                continue
            checked += 1
            delta = d.diff().dt.days.dropna()
            gaps += int((delta > 1).sum())
        date_gaps = {"series_checked": checked, "gaps_gt_1_day": gaps}

    leak_present = [c for c in LEAKAGE_FORBIDDEN if c in df.columns and c != "units_sold"]
    return {
        "n_rows": n,
        "n_columns": int(df.shape[1]),
        "missing_required_columns": missing_cols,
        "missing_value_rate_pct": miss_rates,
        "duplicate_rate_pct": round(100.0 * dups / n, 4) if n else 0.0,
        "n_duplicates": dups,
        "invalid_negative_counts": invalid,
        "category_changes": cat_changes,
        "date_gaps": date_gaps,
        "leakage_columns_present": leak_present,
    }
