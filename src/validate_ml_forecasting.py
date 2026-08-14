"""
Phase 8 — ML Forecasting Validation
===================================
Run: python src/validate_ml_forecasting.py
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml_forecasting import (
    BASELINE_BENCHMARKS,
    FEATURES_PATH,
    METADATA_PATH,
    ML_DIR,
    MODELS_DIR,
    TARGET,
)

PRED_PATH = os.path.join(ML_DIR, "ml_predictions.parquet")
METRICS_PATH = os.path.join(ML_DIR, "ml_model_metrics.parquet")
IMP_PATH = os.path.join(ML_DIR, "feature_importance.parquet")


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


def run_validation() -> ValidationResult:
    v = ValidationResult()
    print("=" * 60)
    print("PHASE 8 ML FORECASTING VALIDATION")
    print("=" * 60)

    print("\n[1] Prerequisites & Artifacts")
    v.check("Phase 6 features exist", os.path.exists(FEATURES_PATH))
    v.check("Predictions exist", os.path.exists(PRED_PATH))
    v.check("Metrics exist", os.path.exists(METRICS_PATH))
    v.check("Feature importance exists", os.path.exists(IMP_PATH))
    v.check("Training metadata exists", os.path.exists(METADATA_PATH))

    for src in ["UCI", "SYNTHETIC"]:
        path = os.path.join(MODELS_DIR, f"{src.lower()}_best_model.joblib")
        v.check(f"{src} best model file exists", os.path.exists(path), path)

    print("\n[2] Predictions Integrity")
    preds = pd.read_parquet(PRED_PATH)
    preds["date"] = pd.to_datetime(preds["date"])
    required = [
        "date", "source_dataset", "entity_id", "product_key",
        "actual_units_sold", "predicted_units_sold", "model",
    ]
    for c in required:
        v.check(f"Prediction column '{c}'", c in preds.columns)

    dup = int(preds.duplicated(
        subset=["date", "source_dataset", "entity_id", "product_key"]
    ).sum())
    v.check("No duplicate prediction keys", dup == 0, f"duplicates={dup}")
    v.check("Predictions are numeric", pd.api.types.is_numeric_dtype(preds["predicted_units_sold"]))
    v.check("Actual target present & numeric", pd.api.types.is_numeric_dtype(preds["actual_units_sold"]))
    inf_p = int(np.isinf(preds["predicted_units_sold"]).sum())
    v.check("No infinite predictions", inf_p == 0, f"inf={inf_p}")
    v.check("No null predictions", int(preds["predicted_units_sold"].isna().sum()) == 0)
    v.check("No negative predictions", int((preds["predicted_units_sold"] < 0).sum()) == 0)

    print("\n[3] Source Separation")
    sources = set(preds["source_dataset"].unique())
    v.check("Both sources in predictions", sources == {"UCI", "SYNTHETIC"}, f"{sources}")
    uci = preds[preds["source_dataset"] == "UCI"]
    syn = preds[preds["source_dataset"] == "SYNTHETIC"]
    v.check(
        "No shared entity across sources",
        len(set(uci["entity_id"]) & set(syn["entity_id"])) == 0,
    )
    v.check(
        "No shared product_key across sources",
        len(set(uci["product_key"]) & set(syn["product_key"])) == 0,
    )

    print("\n[4] Chronology / No Test Leakage (metadata)")
    meta = json.load(open(METADATA_PATH, encoding="utf-8"))
    for src in ["UCI", "SYNTHETIC"]:
        s = meta["sources"][src]
        v.check(
            f"{src}: train_end < validation_start",
            s["train_end"] < s["validation_start"],
            f"{s['train_end']} < {s['validation_start']}",
        )
        v.check(
            f"{src}: validation_end < test_start",
            s["validation_end"] < s["test_start"],
            f"{s['validation_end']} < {s['test_start']}",
        )
        # Predictions should only cover test window
        src_pred = preds[preds["source_dataset"] == src]
        v.check(
            f"{src}: prediction dates within test window",
            str(src_pred["date"].min().date()) >= s["test_start"]
            and str(src_pred["date"].max().date()) <= s["test_end"],
            f"{src_pred['date'].min().date()}..{src_pred['date'].max().date()} vs "
            f"{s['test_start']}..{s['test_end']}",
        )
        v.check(f"{src}: feature list non-empty", len(s["feature_list"]) > 0)
        v.check(f"{src}: target not in feature list", TARGET not in s["feature_list"])
        v.check(f"{src}: revenue not in feature list", "revenue" not in s["feature_list"])

    print("\n[5] Metrics & Baseline Comparison")
    metrics = pd.read_parquet(METRICS_PATH)
    for col in ["MAE", "RMSE", "sMAPE", "WAPE", "baseline_WAPE", "wape_improvement_pct"]:
        v.check(f"Metric column {col}", col in metrics.columns)
        if col in metrics.columns:
            v.check(
                f"{col} finite on non-null",
                bool(np.isfinite(metrics[col].dropna()).all()),
            )

    for src in ["UCI", "SYNTHETIC"]:
        base = BASELINE_BENCHMARKS[src]["WAPE"]
        sub = metrics[(metrics["source_dataset"] == src) & (metrics["split"] == "test")]
        v.check(f"{src}: baseline_WAPE matches Phase 7", 
                bool(np.isclose(sub["baseline_WAPE"].iloc[0], base)),
                f"stored={sub['baseline_WAPE'].iloc[0]} expected={base}")
        selected = sub[sub.get("selected", False) == True] if "selected" in sub.columns else sub.nsmallest(1, "WAPE")
        # fall back: metadata best model
        best_name = meta["sources"][src]["best_model"]
        best_row = sub[sub["model"] == best_name].iloc[0]
        v.check(
            f"{src}: selected model present in test metrics",
            best_name in set(sub["model"]),
            best_name,
        )
        # Recalc improvement
        expected_imp = (base - float(best_row["WAPE"])) / base * 100.0
        v.check(
            f"{src}: wape_improvement_pct consistent",
            abs(float(best_row["wape_improvement_pct"]) - expected_imp) < 0.05,
            f"stored={best_row['wape_improvement_pct']} expected={expected_imp:.4f}",
        )

    print("\n[6] Model Artifacts Load")
    for src in ["UCI", "SYNTHETIC"]:
        path = os.path.join(MODELS_DIR, f"{src.lower()}_best_model.joblib")
        try:
            obj = joblib.load(path)
            ok = (
                isinstance(obj, dict)
                and "model" in obj
                and "preprocessor" in obj
                and "feature_names" in obj
            )
            v.check(f"{src} model artifact loads", ok, type(obj).__name__)
            # smoke predict
            pre = obj["preprocessor"]
            model = obj["model"]
            X_dummy = np.zeros((2, len(obj["feature_names"])), dtype=np.float32)
            # Use preprocessor transform on empty-like frame
            feat_df = pd.DataFrame(
                {c: [np.nan, np.nan] for c in obj["numeric_features"] + obj["categorical_features"]}
            )
            Xt = pre.transform(feat_df)
            pred = model.predict(Xt)
            v.check(
                f"{src} model smoke predict works",
                len(pred) == 2 and np.all(np.isfinite(pred)),
            )
        except Exception as e:
            v.check(f"{src} model artifact loads", False, str(e))

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
