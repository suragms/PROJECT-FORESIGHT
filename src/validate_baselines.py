"""
Phase 7 — Baseline Forecast Validation
======================================
Validates baseline predictions and metrics produced by
``src/baseline_forecasting.py``.

Run:  python src/validate_baselines.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.baseline_forecasting import (
    FEATURES_PATH,
    FORECAST_DIR,
    FULL_GRAIN,
    GRAIN_COLS,
    MODEL_COLS,
    SEASONAL_PERIOD,
    TARGET,
    calculate_metrics,
)

PRED_PATH = os.path.join(FORECAST_DIR, "baseline_predictions.parquet")
METRICS_PATH = os.path.join(FORECAST_DIR, "baseline_metrics.parquet")
BY_SOURCE_PATH = os.path.join(FORECAST_DIR, "baseline_metrics_by_source.parquet")
COMPARISON_PATH = os.path.join(FORECAST_DIR, "baseline_comparison.parquet")


class ValidationResult:
    def __init__(self):
        self.results = []

    def check(self, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.results.append({"name": name, "status": status, "detail": detail})
        marker = "+" if passed else "X"
        print(f"  [{marker}] {name}" + (f" -- {detail}" if detail else ""))

    @property
    def total(self):
        return len(self.results)

    @property
    def passed(self):
        return sum(1 for r in self.results if r["status"] == "PASS")

    @property
    def failed(self):
        return sum(1 for r in self.results if r["status"] == "FAIL")

    def summary(self) -> str:
        return f"{self.passed}/{self.total} PASS"


def run_validation() -> ValidationResult:
    v = ValidationResult()
    print("=" * 60)
    print("PHASE 7 BASELINE VALIDATION")
    print("=" * 60)

    # 1. Phase 6 prerequisite
    print("\n[1] Phase 6 Prerequisite")
    v.check("Phase 6 forecast_features exists", os.path.exists(FEATURES_PATH))

    # 2. Output files
    print("\n[2] Output Files")
    for label, path in [
        ("predictions", PRED_PATH),
        ("metrics", METRICS_PATH),
        ("by_source", BY_SOURCE_PATH),
        ("comparison", COMPARISON_PATH),
    ]:
        v.check(f"{label} file exists", os.path.exists(path), path)

    try:
        pred = pd.read_parquet(PRED_PATH)
        pred["date"] = pd.to_datetime(pred["date"])
        v.check("Predictions readable", True, f"{len(pred):,} rows x {pred.shape[1]} cols")
    except Exception as e:
        v.check("Predictions readable", False, str(e))
        print(v.summary())
        return v

    features = pd.read_parquet(FEATURES_PATH)
    features["date"] = pd.to_datetime(features["date"])

    # 3. Grain
    print("\n[3] Forecast Grain")
    for col in FULL_GRAIN:
        v.check(f"Grain column '{col}' present", col in pred.columns)
        v.check(f"No null in '{col}'", int(pred[col].isna().sum()) == 0)

    dup = int(pred.duplicated(subset=FULL_GRAIN).sum())
    v.check("No duplicate forecast keys", dup == 0, f"duplicates={dup}")

    v.check(
        "Row count matches Phase 6 features",
        len(pred) == len(features),
        f"pred={len(pred):,} features={len(features):,}",
    )

    # 4. Prediction validity
    print("\n[4] Prediction Validity")
    for model, col in MODEL_COLS.items():
        v.check(f"Prediction column '{col}' exists", col in pred.columns)
        if col not in pred.columns:
            continue
        inf_count = int(np.isinf(pred[col].dropna()).sum())
        v.check(f"{model}: no infinite predictions", inf_count == 0, f"inf={inf_count}")
        # Negative demand predictions are allowed mathematically for baselines
        # that can be negative only if source data negative — units_sold >= 0
        # so lags/means should be >= 0 when defined
        neg = int((pred[col].dropna() < 0).sum())
        v.check(f"{model}: no negative predictions", neg == 0, f"neg={neg}")

    # 5. Leakage — first row NaN, and naive == lag_1
    print("\n[5] Leakage Checks")
    first = pred.sort_values(FULL_GRAIN).groupby(GRAIN_COLS, observed=True).nth(0)
    for model, col in MODEL_COLS.items():
        v.check(
            f"{model}: first-row prediction is NaN (no prior history)",
            first[col].isna().all(),
            f"non-null={int(first[col].notna().sum())}",
        )

    # Spot-check one series: naive(t) == units_sold(t-1)
    src = pred["source_dataset"].iloc[0]
    ent = pred.loc[pred["source_dataset"] == src, "entity_id"].iloc[0]
    pk = pred.loc[
        (pred["source_dataset"] == src) & (pred["entity_id"] == ent), "product_key"
    ].iloc[0]
    ts = pred[
        (pred["source_dataset"] == src)
        & (pred["entity_id"] == ent)
        & (pred["product_key"] == pk)
    ].sort_values("date").reset_index(drop=True)

    if len(ts) > 20:
        i = 15
        ok_naive = (
            pd.isna(ts.loc[i, "pred_naive"]) and pd.isna(ts.loc[i - 1, TARGET])
        ) or abs(float(ts.loc[i, "pred_naive"]) - float(ts.loc[i - 1, TARGET])) < 1e-9
        v.check("Naive equals Actual(t-1)", ok_naive)

        ok_seas = (
            pd.isna(ts.loc[i, "pred_seasonal_naive"])
            and pd.isna(ts.loc[i - SEASONAL_PERIOD, TARGET])
        ) or abs(
            float(ts.loc[i, "pred_seasonal_naive"])
            - float(ts.loc[i - SEASONAL_PERIOD, TARGET])
        ) < 1e-9
        v.check(f"Seasonal naive equals Actual(t-{SEASONAL_PERIOD})", ok_seas)

        # MA-7 = mean of t-7..t-1
        expected_ma = float(ts.loc[i - 7: i - 1, TARGET].mean())
        actual_ma = float(ts.loc[i, "pred_ma_7"])
        v.check(
            "MA-7 excludes current target",
            abs(expected_ma - actual_ma) < 1e-6,
            f"expected={expected_ma:.6f} actual={actual_ma:.6f}",
        )

        # Historical mean = mean of 0..i-1
        expected_hm = float(ts.loc[: i - 1, TARGET].mean())
        actual_hm = float(ts.loc[i, "pred_historical_mean"])
        v.check(
            "Historical mean uses only past observations",
            abs(expected_hm - actual_hm) < 1e-6,
            f"expected={expected_hm:.6f} actual={actual_hm:.6f}",
        )

    # 6. Source separation
    print("\n[6] Source Separation")
    sources = sorted(pred["source_dataset"].unique())
    v.check("Two sources present", sources == ["SYNTHETIC", "UCI"] or set(sources) == {"SYNTHETIC", "UCI"},
            f"sources={sources}")
    uci = pred[pred["source_dataset"] == "UCI"]
    syn = pred[pred["source_dataset"] == "SYNTHETIC"]
    shared_e = set(uci["entity_id"].unique()) & set(syn["entity_id"].unique())
    shared_p = set(uci["product_key"].unique()) & set(syn["product_key"].unique())
    v.check("No shared entity_id across sources", len(shared_e) == 0)
    v.check("No shared product_key across sources", len(shared_p) == 0)

    # Cross-boundary: first row of each grain NaN already checked

    # 7. Chronological split
    print("\n[7] Chronological Split")
    v.check("split column present", "split" in pred.columns)
    for src in sources:
        src_df = pred[pred["source_dataset"] == src]
        train_max = src_df.loc[src_df["split"] == "train", "date"].max()
        val_min = src_df.loc[src_df["split"] == "validation", "date"].min()
        val_max = src_df.loc[src_df["split"] == "validation", "date"].max()
        test_min = src_df.loc[src_df["split"] == "test", "date"].min()
        v.check(f"{src}: train_end < val_start", train_max < val_min,
                f"{train_max.date()} < {val_min.date()}")
        v.check(f"{src}: val_end < test_start", val_max < test_min,
                f"{val_max.date()} < {test_min.date()}")

    # 8. Metrics validity
    print("\n[8] Metrics Validity")
    metrics = pd.read_parquet(BY_SOURCE_PATH)
    v.check("by_source metrics readable", True, f"{len(metrics)} rows")
    for col in ["MAE", "RMSE", "sMAPE", "WAPE"]:
        v.check(f"Metric column {col} present", col in metrics.columns)
        finite = np.isfinite(metrics[col].dropna()).all()
        v.check(f"{col} finite", bool(finite))
        nonneg = (metrics[col].dropna() >= 0).all()
        v.check(f"{col} non-negative", bool(nonneg))

    # Recalculate one metric cell to confirm
    test_uci = pred[(pred["source_dataset"] == "UCI") & (pred["split"] == "test")]
    recalc = calculate_metrics(
        test_uci[TARGET].to_numpy(),
        test_uci["pred_naive"].to_numpy(),
        "naive",
    )
    stored = metrics[
        (metrics["source_dataset"] == "UCI")
        & (metrics["split"] == "test")
        & (metrics["model"] == "naive")
    ].iloc[0]
    v.check(
        "Stored UCI naive TEST MAE matches recalculation",
        abs(float(stored["MAE"]) - float(recalc["MAE"])) < 1e-3,
        f"stored={stored['MAE']} recalc={recalc['MAE']}",
    )

    comparison = pd.read_parquet(COMPARISON_PATH)
    v.check("Comparison file has both sources",
            set(comparison["source_dataset"].unique()) == {"UCI", "SYNTHETIC"})
    for src in ["UCI", "SYNTHETIC"]:
        sub = comparison[comparison["source_dataset"] == src]
        v.check(f"{src}: comparison has ranks", "rank" in sub.columns and sub["rank"].min() == 1)

    print("\n" + "=" * 60)
    print(f"VALIDATION RESULT: {v.summary()}")
    if v.failed:
        print("  FAILED:")
        for r in self_results_failed(v):
            print(f"    - {r['name']}: {r['detail']}")
    print("=" * 60)
    return v


def self_results_failed(v: ValidationResult):
    return [r for r in v.results if r["status"] == "FAIL"]


if __name__ == "__main__":
    result = run_validation()
    sys.exit(0 if result.failed == 0 else 1)
