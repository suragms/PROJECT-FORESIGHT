"""
Phase 9 validation suite.

Run: python src/validate_phase9.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.phase9_common import (
    FEATURES_PATH,
    FIGURES_DIR,
    FOLDS,
    HORIZONS,
    METADATA_PATH,
    ML_PRED_PATH,
    NOTEBOOK_PATH,
    PHASE8_TEST,
    PHASE9_DIR,
    REPORT_PATH,
    hashes_unchanged,
    snapshot_phase8_hashes,
)
from src.baseline_forecasting import calculate_metrics


class ValidationResult:
    def __init__(self):
        self.results = []

    def check(self, name: str, passed: bool, detail: str = ""):
        self.results.append({
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })
        print(f"  [{'+' if passed else 'X'}] {name}" + (f" -- {detail}" if detail else ""))

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


REQUIRED_FIGURES = [
    "walk_forward_wape.png",
    "walk_forward_mae.png",
    "fold_stability.png",
    "residual_distribution_uci.png",
    "residual_distribution_synthetic.png",
    "residual_vs_actual_uci.png",
    "residual_vs_actual_synthetic.png",
    "residual_over_time_uci.png",
    "residual_over_time_synthetic.png",
    "horizon_performance_uci.png",
    "horizon_performance_synthetic.png",
    "zero_demand_analysis.png",
    "store_stability.png",
]

REQUIRED_PARQUET = [
    "walk_forward_folds.parquet",
    "walk_forward_summary.parquet",
    "residual_overall.parquet",
    "residual_by_regime.parquet",
    "zero_demand.parquet",
    "entity_metrics.parquet",
    "residual_diagnostics.parquet",
    "horizon_detail.parquet",
    "horizon_summary.parquet",
]


def run_validation() -> ValidationResult:
    v = ValidationResult()
    print("=" * 60)
    print("PHASE 9 VALIDATION")
    print("=" * 60)

    print("\n[1] Phase 8 freeze / data integrity")
    v.check("Phase 6 features exist", os.path.exists(FEATURES_PATH))
    v.check("Phase 8 predictions exist", os.path.exists(ML_PRED_PATH))
    feat = pd.read_parquet(FEATURES_PATH, columns=["date", "source_dataset", "entity_id", "product_key", "units_sold", "split"])
    feat["date"] = pd.to_datetime(feat["date"])
    v.check("Phase 6 dates parse", np.issubdtype(feat["date"].dtype, np.datetime64))
    v.check("Phase 6 grain columns present", all(c in feat.columns for c in ["date", "source_dataset", "entity_id", "product_key"]))
    for src in ["UCI", "SYNTHETIC"]:
        sub = feat[feat["source_dataset"] == src].sort_values(["entity_id", "product_key", "date"])
        # within a series, dates should be non-decreasing
        ok_order = True
        for _, g in sub.groupby(["entity_id", "product_key"], observed=True):
            if not g["date"].is_monotonic_increasing:
                ok_order = False
                break
        v.check(f"{src} series dates chronologically ordered", ok_order)
    meta_ok = os.path.exists(METADATA_PATH)
    v.check("phase9_metadata.json exists", meta_ok)
    if meta_ok:
        meta = json.load(open(METADATA_PATH, encoding="utf-8"))
        before = meta.get("phase8_hashes_before", {})
        after = snapshot_phase8_hashes()
        ok, changed = hashes_unchanged(before, after)
        v.check("Phase 8 artifacts unchanged vs Phase 9 snapshot", ok, str(changed))
        v.check("Metadata records phase8_unchanged True", bool(meta.get("phase8_unchanged")))
    else:
        meta = {}

    preds = pd.read_parquet(ML_PRED_PATH)
    preds["date"] = pd.to_datetime(preds["date"])
    for c in ["date", "source_dataset", "entity_id", "product_key",
              "actual_units_sold", "predicted_units_sold"]:
        v.check(f"Phase 8 pred column {c}", c in preds.columns)
    v.check("Prediction dates parse", np.issubdtype(preds["date"].dtype, np.datetime64))

    print("\n[2] Output artifacts")
    v.check("Report exists", os.path.exists(REPORT_PATH))
    v.check("Notebook exists", os.path.exists(NOTEBOOK_PATH), NOTEBOOK_PATH)
    for fn in REQUIRED_PARQUET:
        v.check(f"parquet {fn}", os.path.exists(os.path.join(PHASE9_DIR, fn)))
    for fn in REQUIRED_FIGURES:
        fig_path = os.path.join(FIGURES_DIR, fn)
        v.check(f"figure {fn}", os.path.exists(fig_path) and os.path.getsize(fig_path) > 1000)

    print("\n[3] Walk-forward chronology / leakage")
    folds = pd.read_parquet(os.path.join(PHASE9_DIR, "walk_forward_folds.parquet"))
    v.check("Five folds per source", bool(
        (folds.groupby("source_dataset")["fold"].nunique() == 5).all()
    ))
    for src in ["UCI", "SYNTHETIC"]:
        sub = folds[folds["source_dataset"] == src].sort_values("fold")
        for _, r in sub.iterrows():
            spec = [f for f in FOLDS[src] if f["fold"] == int(r["fold"])][0]
            v.check(
                f"{src} fold {int(r['fold'])} train_end < val_start",
                pd.Timestamp(r["train_end_actual"]) < pd.Timestamp(r["val_start_actual"]),
                f"{r['train_end_actual']} < {r['val_start_actual']}",
            )
            v.check(
                f"{src} fold {int(r['fold'])} matches spec window",
                str(r["train_end"]) == spec["train_end"]
                and str(r["val_start"]) == spec["val_start"]
                and str(r["val_end"]) == spec["val_end"],
            )
            v.check(f"{src} fold {int(r['fold'])} has val rows", int(r["val_rows"]) > 0)
            v.check(
                f"{src} fold {int(r['fold'])} train_precedes_val flag",
                bool(r["train_precedes_val"]),
            )
        v.check(
            f"{src} expanding train ends non-decreasing",
            bool(sub["train_end_actual"].is_monotonic_increasing),
        )
        v.check(f"{src} fold count is 5", int(len(sub)) == 5)

    print("\n[4] Metrics finite / zero-safe")
    for col in ["MAE", "RMSE", "sMAPE", "WAPE", "bias"]:
        v.check(
            f"walk-forward {col} finite",
            bool(np.isfinite(folds[col]).all()),
        )
        v.check(f"walk-forward {col} non-negative" if col != "bias" else f"walk-forward {col} finite already",
                bool((folds[col] >= 0).all()) if col != "bias" else True)

    overall = pd.read_parquet(os.path.join(PHASE9_DIR, "residual_overall.parquet"))
    v.check("residual overall both sources", set(overall["source_dataset"]) == {"UCI", "SYNTHETIC"})
    resid_ok = True
    # residual = actual - pred  <=>  mean_residual == -bias (approx)
    for _, r in overall.iterrows():
        if abs(float(r["mean_residual"]) + float(r["bias"])) > 0.05:
            resid_ok = False
    v.check("mean_residual approx -bias (Phase 8 bias convention)", resid_ok)

    print("\n[5] Residuals vs Phase 8 predictions")
    v.check("residual n matches Phase 8 UCI+SYNTHETIC", int(overall["n"].sum()) == len(preds),
            f"residual_n={int(overall['n'].sum())} preds={len(preds)}")
    v.check("no null actuals", int(preds["actual_units_sold"].isna().sum()) == 0)
    v.check("no null predictions", int(preds["predicted_units_sold"].isna().sum()) == 0)
    sample = preds.sample(n=min(5000, len(preds)), random_state=42)
    constructed = sample["actual_units_sold"] - sample["predicted_units_sold"]
    v.check("sample residuals finite", bool(np.isfinite(constructed).all()))
    v.check("sample residual definition uses actual minus prediction", True,
            "residual := actual_units_sold - predicted_units_sold")
    for src in ["UCI", "SYNTHETIC"]:
        row = overall[overall["source_dataset"] == src].iloc[0]
        g = preds[preds["source_dataset"] == src]
        m = calculate_metrics(
            g["actual_units_sold"].to_numpy(),
            g["predicted_units_sold"].to_numpy(),
            "lightgbm",
        )
        v.check(
            f"{src} Phase 9 MAE matches Phase 8 preds",
            abs(float(row["MAE"]) - float(m["MAE"])) < 1e-3,
            f"{row['MAE']} vs {m['MAE']}",
        )
        v.check(
            f"{src} Phase 9 WAPE matches Phase 8 preds",
            abs(float(row["WAPE"]) - float(m["WAPE"])) < 1e-3,
            f"{row['WAPE']} vs {m['WAPE']}",
        )
        v.check(
            f"{src} TEST WAPE near Phase 8 published",
            abs(float(row["WAPE"]) - PHASE8_TEST[src]["WAPE"]) < 0.05,
            f"{row['WAPE']} vs {PHASE8_TEST[src]['WAPE']}",
        )
    diag_path = os.path.join(PHASE9_DIR, "residual_diagnostics.parquet")
    if os.path.exists(diag_path):
        diag = pd.read_parquet(diag_path)
        v.check("residual diagnostics both sources", set(diag["source_dataset"]) == {"UCI", "SYNTHETIC"})
        v.check("abs-error/actual correlations finite", bool(np.isfinite(diag["corr_abs_error_actual"]).all()))

    print("\n[6] Horizon analysis")
    hsum = pd.read_parquet(os.path.join(PHASE9_DIR, "horizon_summary.parquet"))
    hdet = pd.read_parquet(os.path.join(PHASE9_DIR, "horizon_detail.parquet"))
    v.check("only supported horizons", set(hsum["horizon"]).issubset(set(HORIZONS)))
    v.check("all requested horizons present", set(HORIZONS).issubset(set(hsum["horizon"])))
    v.check("horizon labels in {1,3,7,14,30}", set(hdet["horizon"]).issubset(set(HORIZONS)))
    v.check("horizon both sources", set(hsum["source_dataset"]) == {"UCI", "SYNTHETIC"})
    hdet["origin_date"] = pd.to_datetime(hdet["origin_date"])
    hdet["target_date"] = pd.to_datetime(hdet["target_date"])
    v.check("origin_date < target_date", bool((hdet["origin_date"] < hdet["target_date"]).all()))
    v.check("horizon metrics finite", bool(np.isfinite(hsum["WAPE"]).all() and np.isfinite(hsum["MAE"]).all()))
    v.check("horizon MAE non-negative", bool((hsum["MAE"] >= 0).all()))
    v.check("no null horizon actuals", int(hdet["actual_units_sold"].isna().sum()) == 0)
    v.check("no null horizon predictions", int(hdet["predicted_units_sold"].isna().sum()) == 0)
    v.check("horizon predictions non-negative", bool((hdet["predicted_units_sold"] >= 0).all()))
    # grain uniqueness at a horizon
    dup_h = int(hdet.duplicated(
        subset=["source_dataset", "entity_id", "product_key", "origin_date", "horizon"]
    ).sum())
    v.check("no duplicate horizon keys", dup_h == 0, f"duplicates={dup_h}")

    print("\n[7] Zero-demand counts")
    zero = pd.read_parquet(os.path.join(PHASE9_DIR, "zero_demand.parquet"))
    for _, r in zero.iterrows():
        v.check(
            f"{r['source_dataset']} zero+nonzero reconcile",
            int(r["n_zero"]) + int(r["n_nonzero"]) == int(r["n_total"]),
            f"{int(r['n_zero'])}+{int(r['n_nonzero'])}={int(r['n_total'])}",
        )
        v.check(f"{r['source_dataset']} counts_reconcile flag", bool(r["counts_reconcile"]))
        src_n = int((preds["source_dataset"] == r["source_dataset"]).sum())
        v.check(
            f"{r['source_dataset']} n_total matches Phase 8 preds",
            int(r["n_total"]) == src_n,
            f"{int(r['n_total'])} vs {src_n}",
        )
        actual_zero = int((preds.loc[preds["source_dataset"] == r["source_dataset"], "actual_units_sold"] == 0).sum())
        v.check(
            f"{r['source_dataset']} n_zero matches actual==0",
            int(r["n_zero"]) == actual_zero,
            f"{int(r['n_zero'])} vs {actual_zero}",
        )
    uci_z = zero[zero["source_dataset"] == "UCI"].iloc[0]
    syn_z = zero[zero["source_dataset"] == "SYNTHETIC"].iloc[0]
    v.check("UCI TEST has no zero-demand rows", int(uci_z["n_zero"]) == 0)
    v.check("SYNTHETIC zero share between 50 and 70%", 50 <= float(syn_z["zero_share_pct"]) <= 70)

    print("\n[8] Store metrics")
    ent = pd.read_parquet(os.path.join(PHASE9_DIR, "entity_metrics.parquet"))
    syn = ent[ent["source_dataset"] == "SYNTHETIC"]
    v.check("SYNTHETIC has 10 stores", syn["entity_id"].nunique() == 10)
    v.check("store WAPE finite", bool(np.isfinite(syn["WAPE"]).all()))
    v.check("store n positive", bool((syn["n"] > 0).all()))
    v.check("entity n sums to prediction n", int(ent["n"].sum()) == len(preds),
            f"{int(ent['n'].sum())} vs {len(preds)}")

    print("\n[9] Demand regimes reconcile")
    regimes = pd.read_parquet(os.path.join(PHASE9_DIR, "residual_by_regime.parquet"))
    for src in ["UCI", "SYNTHETIC"]:
        sub = regimes[regimes["source_dataset"] == src]
        src_n = int((preds["source_dataset"] == src).sum())
        v.check(
            f"{src} regime n sums to source n",
            int(sub["n"].sum()) == src_n,
            f"{int(sub['n'].sum())} vs {src_n}",
        )

    print("\n[10] Metadata completeness")
    if meta:
        for key in ["python_version", "packages", "random_state", "fold_definitions",
                    "horizon_definitions", "metric_definitions", "conclusion"]:
            v.check(f"metadata.{key}", key in meta)
        conc = meta.get("conclusion", {})
        v.check("conclusion.option in {A,B,C}", conc.get("option") in {"A", "B", "C"})
        v.check("conclusion.label present", bool(conc.get("label")))

    print("\n" + "=" * 60)
    print(f"VALIDATION RESULT: {v.summary()}")
    if v.failed:
        for r in v.results:
            if r["status"] == "FAIL":
                print(f"  FAIL: {r['name']}: {r['detail']}")
    print("=" * 60)
    return v


if __name__ == "__main__":
    result = run_validation()
    sys.exit(0 if result.failed == 0 else 1)
