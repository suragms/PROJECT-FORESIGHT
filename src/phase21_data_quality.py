"""Phase 21 — Data quality monitoring for Synthetic weekly production inputs."""

from __future__ import annotations

import pandas as pd

from src.phase21_common import P19_FEAT, now_iso, classify_status


REQUIRED_COLS = ["week", "product_key", "units_sold", "source_dataset"]
EXPECTED_SOURCE = "SYNTHETIC"


def run_data_quality_monitoring(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_parquet(P19_FEAT)
    df = df.copy()
    df["week"] = pd.to_datetime(df["week"])

    checks = {}

    # Schema
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    checks["schema_validity"] = "FAIL" if missing_cols else "PASS"
    checks["missing_critical_columns"] = missing_cols

    # Source
    sources = set(df["source_dataset"].unique()) if "source_dataset" in df.columns else set()
    checks["unexpected_source"] = "FAIL" if sources - {EXPECTED_SOURCE} else "PASS"
    checks["sources_found"] = list(sources)

    # Nulls on critical
    null_rates = {c: round(float(df[c].isna().mean()), 4) for c in REQUIRED_COLS if c in df.columns}
    checks["null_rates"] = null_rates
    checks["missing_values"] = "WARNING" if any(v > 0.05 for v in null_rates.values()) else "PASS"

    # Duplicates at grain
    dup = int(df.duplicated(subset=["week", "product_key"]).sum()) if "product_key" in df.columns else 0
    checks["duplicate_records"] = "FAIL" if dup > 0 else "PASS"
    checks["duplicate_count"] = dup

    # Invalid dates
    invalid_dates = int(df["week"].isna().sum()) if "week" in df.columns else 0
    checks["invalid_dates"] = "FAIL" if invalid_dates > 0 else "PASS"

    # Negative quantities
    neg_qty = int((df["units_sold"] < 0).sum()) if "units_sold" in df.columns else 0
    checks["negative_quantities"] = "FAIL" if neg_qty > 0 else "PASS"

    # SKU coverage
    n_skus = int(df["product_key"].nunique()) if "product_key" in df.columns else 0
    checks["sku_count"] = n_skus
    checks["sku_coverage"] = "WARNING" if n_skus < 90 else "PASS"  # baseline ~100 SKUs

    # Row count anomaly (compare to historical median)
    weekly_rows = len(df[df["week"] == df["week"].max()]) if len(df) else 0
    median_weekly = df.groupby("week").size().median() if len(df) else 0
    row_anomaly = abs(weekly_rows - median_weekly) / max(median_weekly, 1)
    checks["latest_week_rows"] = int(weekly_rows)
    checks["median_weekly_rows"] = int(median_weekly)
    checks["row_count_anomaly"] = "WARNING" if row_anomaly > 0.2 else "PASS"

    check_vals = [v for k, v in checks.items() if k.endswith(("validity", "source", "values", "records", "dates", "quantities", "coverage", "anomaly")) and isinstance(v, str)]
    checks["overall_status"] = classify_status(check_vals)
    checks["timestamp"] = now_iso()
    return checks
