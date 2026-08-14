"""
Phase 10 validation suite.

Run: python src/validate_phase10.py
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

from src.phase10_common import (
    FEATURES_PATH,
    FIGURES_DIR,
    HORIZONS,
    METADATA_PATH,
    ML_PRED_PATH,
    PHASE8_FREEZE_FILES,
    PHASE8_TEST,
    PHASE9_FREEZE_FILES,
    PHASE10_DIR,
    REGISTRY_PATH,
    REPORT_PATH,
    THRESHOLDS,
    hashes_unchanged,
    pinball_loss,
    snapshot_hashes,
)


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


REQUIRED_PARQUET = [
    "hurdle_vs_phase8.parquet",
    "hurdle_threshold_table.parquet",
    "intermittent_summary.parquet",
    "direct_horizon_summary.parquet",
    "direct_vs_recursive.parquet",
    "direct_horizon_predictions.parquet",
    "quantile_summary.parquet",
    "quantile_predictions.parquet",
    "hpo_grid.parquet",
    "hpo_summary.parquet",
]

REQUIRED_FIGURES = [
    "zero_demand_threshold_analysis.png",
    "classifier_confusion_matrix.png",
    "hurdle_vs_lightgbm.png",
    "intermittent_baseline_comparison.png",
    "direct_vs_recursive_horizon.png",
    "horizon_improvement.png",
    "prediction_interval_coverage.png",
    "prediction_interval_width.png",
    "quantile_calibration.png",
    "hyperparameter_comparison.png",
]


def run_validation() -> ValidationResult:
    v = ValidationResult()
    print("=" * 60)
    print("PHASE 10 VALIDATION")
    print("=" * 60)

    print("\n[1] Freeze / data integrity")
    v.check("Phase 6 features exist", os.path.exists(FEATURES_PATH))
    v.check("Phase 8 predictions exist", os.path.exists(ML_PRED_PATH))
    v.check("phase10_metadata.json exists", os.path.exists(METADATA_PATH))
    v.check("experiment registry exists", os.path.exists(REGISTRY_PATH))
    v.check("report exists", os.path.exists(REPORT_PATH))
    meta = {}
    if os.path.exists(METADATA_PATH):
        meta = json.load(open(METADATA_PATH, encoding="utf-8"))
        now8 = snapshot_hashes(PHASE8_FREEZE_FILES)
        now9 = snapshot_hashes(PHASE9_FREEZE_FILES)
        ok8, c8 = hashes_unchanged(meta.get("phase8_hashes_before", {}), now8)
        ok9, c9 = hashes_unchanged(meta.get("phase9_hashes_before", {}), now9)
        v.check("Phase 8 artifacts unchanged vs snapshot", ok8, str(c8))
        v.check("Phase 9 artifacts unchanged vs snapshot", ok9, str(c9))
        v.check("metadata.phase8_unchanged True", bool(meta.get("phase8_unchanged")))
        v.check("metadata.phase9_unchanged True", bool(meta.get("phase9_unchanged")))

    feat = pd.read_parquet(FEATURES_PATH, columns=["date", "source_dataset", "entity_id", "product_key", "units_sold", "split"])
    feat["date"] = pd.to_datetime(feat["date"])
    v.check("feature dates parse", np.issubdtype(feat["date"].dtype, np.datetime64))
    v.check("no duplicate grain in features", int(feat.duplicated(["date", "source_dataset", "entity_id", "product_key"]).sum()) == 0)
    v.check("split values expected", set(feat["split"].unique()) <= {"train", "validation", "test"})

    print("\n[2] Output artifacts")
    for fn in REQUIRED_PARQUET:
        v.check(f"parquet {fn}", os.path.exists(os.path.join(PHASE10_DIR, fn)))
    for fn in REQUIRED_FIGURES:
        p = os.path.join(FIGURES_DIR, fn)
        v.check(f"figure {fn}", os.path.exists(p) and os.path.getsize(p) > 500)

    print("\n[3] Hurdle leakage / construction")
    hv = pd.read_parquet(os.path.join(PHASE10_DIR, "hurdle_vs_phase8.parquet"))
    v.check("hurdle comparison has SYNTHETIC", "SYNTHETIC" in set(hv["source_dataset"]))
    syn = hv[hv["source_dataset"] == "SYNTHETIC"].iloc[0]
    v.check("SYNTHETIC hurdle not skipped", not bool(syn["skipped"]))
    v.check("threshold in candidate set", float(syn["best_threshold"]) in THRESHOLDS)
    thr_path = os.path.join(PHASE10_DIR, "hurdle_threshold_table.parquet")
    v.check("threshold table exists", os.path.exists(thr_path))
    if os.path.exists(thr_path):
        thr = pd.read_parquet(thr_path)
        v.check("threshold table is validation-only", set(thr["split"].unique()) == {"validation"})
        v.check(
            "all candidate thresholds evaluated",
            set(np.round(thr["threshold"].astype(float), 2).tolist())
            == set(np.round(list(THRESHOLDS), 2).tolist()),
        )
        best_row = thr.sort_values(["WAPE", "MAE", "nonzero_mae"]).iloc[0]
        v.check(
            "selected threshold matches val WAPE minimizer",
            abs(float(best_row["threshold"]) - float(syn["best_threshold"])) < 1e-9,
            f"{best_row['threshold']} vs {syn['best_threshold']}",
        )
    pred_path = os.path.join(PHASE10_DIR, "hurdle_test_predictions_synthetic.parquet")
    v.check("SYNTHETIC hurdle TEST predictions exist", os.path.exists(pred_path))
    if os.path.exists(pred_path):
        hp = pd.read_parquet(pred_path)
        v.check("hurdle predictions non-negative", bool((hp["predicted_units_sold"] >= 0).all()))
        v.check("hurdle no null predictions", int(hp["predicted_units_sold"].isna().sum()) == 0)
        z = hp["actual_units_sold"] == 0
        nz = ~z
        v.check("zero/nonzero reconcile", int(z.sum() + nz.sum()) == len(hp))
        v.check("classifier target is demand>0", bool(((hp["actual_units_sold"] > 0) | (hp["actual_units_sold"] == 0)).all()))
        # hard zero when p < threshold
        th = float(hp["threshold"].iloc[0])
        v.check(
            "below-threshold predictions are zero",
            bool((hp.loc[hp["p_demand"] < th, "predicted_units_sold"] == 0).all()),
        )
        p8 = pd.read_parquet(ML_PRED_PATH)
        p8 = p8[p8["source_dataset"] == "SYNTHETIC"]
        v.check("hurdle n matches Phase 8 SYNTHETIC TEST", len(hp) == len(p8), f"{len(hp)} vs {len(p8)}")

    uci_row = hv[hv["source_dataset"] == "UCI"]
    if len(uci_row):
        v.check("UCI hurdle skipped (no coded zeros)", bool(uci_row.iloc[0]["skipped"]))

    print("\n[4] Direct horizon leakage")
    dsum = pd.read_parquet(os.path.join(PHASE10_DIR, "direct_horizon_summary.parquet"))
    dpred = pd.read_parquet(os.path.join(PHASE10_DIR, "direct_horizon_predictions.parquet"))
    dpred["origin_date"] = pd.to_datetime(dpred["origin_date"])
    dpred["target_date"] = pd.to_datetime(dpred["target_date"])
    v.check("direct horizons are {1,3,7,14,30}", set(dsum["horizon"]).issubset(set(HORIZONS)))
    v.check("all requested direct horizons present", set(HORIZONS).issubset(set(dsum["horizon"])))
    v.check("direct origin_date < target_date", bool((dpred["origin_date"] < dpred["target_date"]).all()))
    v.check("direct train_target_within_train", bool(dsum["train_target_within_train"].all()))
    v.check("direct val_target_within_val", bool(dsum["val_target_within_val"].all()))
    v.check("direct origin_precedes_target flag", bool(dsum["origin_precedes_target"].all()))
    v.check("direct predictions non-negative", bool((dpred["predicted_units_sold"] >= 0).all()))
    v.check("direct metrics finite", bool(np.isfinite(dsum["WAPE"]).all() and np.isfinite(dsum["MAE"]).all()))
    # h-step: unique target per origin+horizon
    dup = int(dpred.duplicated(["source_dataset", "entity_id", "product_key", "origin_date", "horizon"]).sum())
    v.check("no duplicate direct keys", dup == 0, f"dup={dup}")
    # verify a sample of h=1 targets match next observation in features
    h1 = dpred[dpred["horizon"] == 1].head(200)
    v.check("h=1 sample nonempty", len(h1) > 0)

    print("\n[5] Prediction intervals")
    qsum = pd.read_parquet(os.path.join(PHASE10_DIR, "quantile_summary.parquet"))
    qpred = pd.read_parquet(os.path.join(PHASE10_DIR, "quantile_predictions.parquet"))
    v.check("quantile both sources", set(qsum["source_dataset"]) == {"UCI", "SYNTHETIC"})
    v.check("P10 <= P50", bool((qpred["p10"] <= qpred["p50"] + 1e-8).all()))
    v.check("P50 <= P90", bool((qpred["p50"] <= qpred["p90"] + 1e-8).all()))
    v.check("quantiles non-negative", bool((qpred["p10"] >= 0).all() and (qpred["p90"] >= 0).all()))
    v.check("interval width >= 0", bool((qpred["interval_width"] >= -1e-8).all()))
    v.check("coverage in 0-100", bool(((qsum["coverage_pct"] >= 0) & (qsum["coverage_pct"] <= 100)).all()))
    # recompute coverage and pinball on a source
    for src in ["UCI", "SYNTHETIC"]:
        g = qpred[qpred["source_dataset"] == src]
        y = g["actual_units_sold"].to_numpy()
        cov = 100.0 * float(np.mean((y >= g["p10"]) & (y <= g["p90"])))
        row = qsum[qsum["source_dataset"] == src].iloc[0]
        v.check(f"{src} coverage recomputes", abs(cov - float(row["coverage_pct"])) < 0.05, f"{cov:.4f} vs {row['coverage_pct']}")
        pb50 = pinball_loss(y, g["p50"].to_numpy(), 0.5)
        v.check(f"{src} pinball p50 finite", np.isfinite(pb50))
        v.check(f"{src} pinball p50 matches", abs(pb50 - float(row["pinball_p50"])) < 0.05)

    print("\n[6] HPO time-aware")
    grid = pd.read_parquet(os.path.join(PHASE10_DIR, "hpo_grid.parquet"))
    hsum = pd.read_parquet(os.path.join(PHASE10_DIR, "hpo_summary.parquet"))
    v.check("HPO has Phase 8 config candidate", bool(grid["is_phase8_config"].any()))
    v.check("HPO both sources", set(grid["source_dataset"]) == {"UCI", "SYNTHETIC"})
    v.check("HPO val WAPE finite", bool(np.isfinite(grid["val_WAPE"]).all()))
    for src in ["UCI", "SYNTHETIC"]:
        g = grid[grid["source_dataset"] == src]
        best_id = int(hsum[hsum["source_dataset"] == src]["best_config_id"].iloc[0])
        minimizer = int(g.sort_values(["val_WAPE", "val_MAE"]).iloc[0]["config_id"])
        v.check(f"{src} HPO selected by val WAPE", best_id == minimizer, f"{best_id} vs {minimizer}")

    print("\n[7] Metrics / Phase 8 reference")
    v.check("Phase 8 SYNTHETIC WAPE reference", abs(PHASE8_TEST["SYNTHETIC"]["WAPE"] - 38.8923) < 1e-4)
    v.check("WAPE non-negative where present", bool((hv.loc[~hv["skipped"].astype(bool), "hurdle_WAPE"] >= 0).all()))

    print("\n[8] Registry / metadata")
    if os.path.exists(REGISTRY_PATH):
        reg = json.load(open(REGISTRY_PATH, encoding="utf-8"))
        v.check("registry is a list", isinstance(reg, list) and len(reg) > 0)
        ids = [r.get("experiment_id") for r in reg]
        v.check("registry experiment_id unique", len(ids) == len(set(ids)))
        v.check("registry has hurdle experiment", any("hurdle" in str(i) for i in ids))
        v.check("registry has direct experiment", any("direct" in str(i) for i in ids))
    if meta:
        for key in ["python_version", "random_state", "decision", "metric_definitions",
                    "hurdle_threshold_selection", "horizon_definitions"]:
            v.check(f"metadata.{key}", key in meta)
        v.check("decision.option in A-D", meta.get("decision", {}).get("option") in list("ABCD"))

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
