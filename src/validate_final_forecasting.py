"""
Phase 11 — Final forecasting validation.

Run: python src/validate_final_forecasting.py
Target: 100% PASS
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

from src.final_forecasting import FinalForecastError, FinalForecaster  # noqa: E402
from src.ml_forecasting import load_feature_dataset, prepare_features  # noqa: E402
from src.phase10_common import (  # noqa: E402
    ML_PRED_PATH,
    PHASE8_FREEZE_FILES,
    PHASE8_TEST,
    PHASE9_FREEZE_FILES,
    PHASE10_DIR,
    hashes_unchanged,
    snapshot_hashes,
)
from src.phase10_hurdle_forecasting import _usable  # noqa: E402
from src.phase11_common import (  # noqa: E402
    CANDIDATE_PATH,
    FIGURES_FINAL_DIR,
    FINAL_PRED_PATH,
    FORECASTS_FINAL_DIR,
    LEAKAGE_FORBIDDEN,
    METADATA_PATH,
    MODELS_FINAL_DIR,
    MONITOR_PATH,
    OUTPUT_SCHEMA,
    REGISTRY_PATH,
    REPORT_PATH,
    file_sha256,
    verify_phase10_ready,
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


REQUIRED_FIGURES = [
    "final_model_comparison.png",
    "final_horizon_comparison.png",
    "final_store_stability.png",
    "final_residual_analysis.png",
    "final_zero_demand_comparison.png",
    "final_prediction_intervals.png",
]


def _load_registry() -> list[dict]:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_validation() -> ValidationResult:
    v = ValidationResult()
    print("=" * 60)
    print("PHASE 11 FINAL VALIDATION")
    print("=" * 60)

    print("\n[1] Phase 10 / freeze")
    ready = verify_phase10_ready()
    v.check("Phase 10 complete and frozen", ready["ready"], "; ".join(ready["issues"]))
    if os.path.exists(METADATA_PATH):
        meta = json.load(open(METADATA_PATH, encoding="utf-8"))
        cur8 = snapshot_hashes(PHASE8_FREEZE_FILES)
        cur9 = snapshot_hashes(PHASE9_FREEZE_FILES)
        ok8, ch8 = hashes_unchanged(meta.get("phase8_hashes") or {}, cur8)
        ok9, ch9 = hashes_unchanged(meta.get("phase9_hashes") or {}, cur9)
        v.check("Phase 8 hashes unchanged after Phase 11", ok8, str(ch8))
        v.check("Phase 9 hashes unchanged after Phase 11", ok9, str(ch9))
        v.check("metadata.phase8_unchanged", bool(meta.get("phase8_unchanged")))
        v.check("metadata.phase9_unchanged", bool(meta.get("phase9_unchanged")))
    else:
        meta = {}
        v.check("phase11_metadata.json exists", False)

    print("\n[2] Model integrity")
    v.check("registry exists", os.path.exists(REGISTRY_PATH))
    v.check("models/final exists", os.path.isdir(MODELS_FINAL_DIR))
    registry = _load_registry() if os.path.exists(REGISTRY_PATH) else []
    v.check("registry is a list", isinstance(registry, list) and len(registry) > 0)
    ids = [r.get("model_id") for r in registry]
    v.check("model_id unique", len(ids) == len(set(ids)))
    selected = [r for r in registry if r.get("status") == "selected"]
    interval = [r for r in registry if r.get("status") == "interval_companion"]
    v.check("has UCI h=1 Phase 8 model", any(r.get("model_id") == "uci_h1_phase8_lightgbm" for r in selected))
    v.check("has SYNTHETIC hurdle", any(r.get("model_id") == "synthetic_h1_hurdle_th050" for r in selected))
    v.check("has 8 direct models", sum(1 for r in selected if r.get("model_type") == "direct_lightgbm") == 8)
    v.check("has two interval companions", len(interval) == 2)
    # only selected + companions live in models/final
    joblibs = [f for f in os.listdir(MODELS_FINAL_DIR) if f.endswith(".joblib")] if os.path.isdir(MODELS_FINAL_DIR) else []
    v.check("final dir file count matches registry", len(joblibs) == len(registry), f"{len(joblibs)} vs {len(registry)}")

    for r in registry:
        mid = r["model_id"]
        path = r["model_file"]
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        v.check(f"{mid} file exists", os.path.exists(path))
        if not os.path.exists(path):
            continue
        got = file_sha256(path)
        v.check(f"{mid} hash matches registry", got == r.get("hash"), f"{got} vs {r.get('hash')}")
        try:
            payload = joblib.load(path)
            loaded = True
        except Exception as e:
            payload, loaded = {}, False
            v.check(f"{mid} loads", False, str(e))
            continue
        v.check(f"{mid} loads", loaded)
        v.check(f"{mid} has required features list", bool(payload.get("numeric_features")))
        leak = [c for c in (payload.get("numeric_features") or []) + (payload.get("categorical_features") or [])
                if c in LEAKAGE_FORBIDDEN]
        v.check(f"{mid} no leakage features", leak == [], str(leak))
        v.check(f"{mid} units_sold not a feature", "units_sold" not in (payload.get("numeric_features") or []))

    print("\n[3] Prediction integrity")
    v.check("final predictions exist", os.path.exists(FINAL_PRED_PATH))
    if os.path.exists(FINAL_PRED_PATH):
        pred = pd.read_parquet(FINAL_PRED_PATH)
        for c in OUTPUT_SCHEMA:
            v.check(f"schema column {c}", c in pred.columns)
        v.check("predictions finite", bool(np.isfinite(pred["prediction"].to_numpy(dtype=float)).all()))
        v.check("predictions non-negative", bool((pred["prediction"] >= -1e-12).all()))
        v.check("horizons in {1,3,7,14,30}", set(pred["horizon"].astype(int)) <= {1, 3, 7, 14, 30})
        dup = int(pred.duplicated(
            ["forecast_date", "source_dataset", "entity_id", "product_key", "horizon"]
        ).sum())
        v.check("no duplicate forecast keys", dup == 0, f"dup={dup}")
        v.check("dates valid", bool(pd.to_datetime(pred["forecast_date"], errors="coerce").notna().all()))
        # actuals only on TEST evaluation file — allowed here; must be finite or NaN not mixed junk
        n_null_pred = int(pred["prediction"].isna().sum())
        v.check("no null predictions", n_null_pred == 0)
        h1 = pred[pred["horizon"] == 1]
        v.check("h=1 has both datasets", set(h1["source_dataset"]) == {"UCI", "SYNTHETIC"})
        # lower <= upper when both present
        both = h1["lower_bound"].notna() & h1["upper_bound"].notna()
        if both.any():
            v.check(
                "interval lower <= upper",
                bool((h1.loc[both, "lower_bound"] <= h1.loc[both, "upper_bound"] + 1e-8).all()),
            )
            v.check("interval bounds non-negative", bool((h1.loc[both, "lower_bound"] >= -1e-12).all()))

        # UCI h=1 matches frozen Phase 8 predictions
        p8 = pd.read_parquet(ML_PRED_PATH)
        p8["date"] = pd.to_datetime(p8["date"])
        uci = h1[h1["source_dataset"] == "UCI"]
        merged = uci.merge(
            p8[p8["source_dataset"] == "UCI"],
            left_on=["forecast_date", "entity_id", "product_key"],
            right_on=["date", "entity_id", "product_key"],
            how="inner",
        )
        v.check("UCI h=1 matches Phase 8 row count", len(merged) == len(uci) == len(p8[p8["source_dataset"] == "UCI"]))
        if len(merged):
            dmax = float(np.max(np.abs(
                merged["prediction"].to_numpy(dtype=float)
                - merged["predicted_units_sold"].to_numpy(dtype=float)
            )))
            v.check("UCI h=1 matches Phase 8 predictions", dmax < 1e-6, f"max_abs_diff={dmax}")

        syn = h1[h1["source_dataset"] == "SYNTHETIC"]
        v.check("SYNTHETIC h=1 n=147000", len(syn) == 147000, f"n={len(syn)}")
        hp = pd.read_parquet(os.path.join(PHASE10_DIR, "hurdle_test_predictions_synthetic.parquet"))
        hp["date"] = pd.to_datetime(hp["date"])
        sm = syn.merge(
            hp,
            left_on=["forecast_date", "entity_id", "product_key"],
            right_on=["date", "entity_id", "product_key"],
            how="inner",
        )
        if len(sm):
            dmax = float(np.max(np.abs(
                sm["prediction"].to_numpy(dtype=float)
                - sm["predicted_units_sold"].to_numpy(dtype=float)
            )))
            v.check("SYNTHETIC hurdle close to Phase 10", dmax < 1e-4, f"max_abs_diff={dmax}")

    print("\n[4] Data integrity / docs")
    v.check("candidate matrix exists", os.path.exists(CANDIDATE_PATH))
    v.check("report exists", os.path.exists(REPORT_PATH) and os.path.getsize(REPORT_PATH) > 1000)
    v.check("monitoring plan exists", os.path.exists(MONITOR_PATH) and os.path.getsize(MONITOR_PATH) > 500)
    if os.path.exists(REPORT_PATH):
        txt = open(REPORT_PATH, encoding="utf-8").read()
        for heading in [
            "## 1. Executive Summary",
            "## 5. Final Model Selection",
            "## 7. Zero-Demand Results",
            "## 12. Production Readiness",
        ]:
            v.check(f"report has {heading}", heading in txt)
        v.check("readiness is READY WITH MONITORING", "READY WITH MONITORING" in txt)
        v.check("report does not claim READY without monitoring", "**READY**" not in txt.split("## 12")[-1][:200] or "READY WITH MONITORING" in txt)
    for fn in REQUIRED_FIGURES:
        p = os.path.join(FIGURES_FINAL_DIR, fn)
        v.check(f"figure {fn}", os.path.exists(p) and os.path.getsize(p) > 500)

    print("\n[5] Leakage")
    v.check("forbidden columns documented", "units_sold" in LEAKAGE_FORBIDDEN)
    # inference path cannot require future actual: units_sold is optional actual only
    if registry:
        hurdle = next(r for r in registry if r["model_id"] == "synthetic_h1_hurdle_th050")
        ff = FinalForecaster.from_registry(hurdle["model_id"])
        v.check("hurdle features exclude units_sold", "units_sold" not in ff.feature_cols)
        direct = next(r for r in registry if r["model_type"] == "direct_lightgbm")
        dff = FinalForecaster.from_registry(direct["model_id"])
        v.check("direct features exclude units_sold", "units_sold" not in dff.feature_cols)
        v.check("direct uses hcal calendar", any(str(c).startswith("hcal_") for c in dff.feature_cols))

    print("\n[6] Input rejection")
    if os.path.exists(FINAL_PRED_PATH) and registry:
        uci_ff = FinalForecaster.from_registry("uci_h1_phase8_lightgbm")
        feat = load_feature_dataset()
        sub, _, _ = prepare_features(feat, "UCI")
        test = _usable(sub)
        test = test[test["split"] == "test"].head(50).copy()
        try:
            uci_ff.predict(test)
            v.check("valid UCI sample predicts", True)
        except Exception as e:
            v.check("valid UCI sample predicts", False, str(e))
        bad = test.drop(columns=["units_sold_lag_1"])
        try:
            uci_ff.predict(bad)
            v.check("rejects missing lag", False)
        except FinalForecastError:
            v.check("rejects missing lag", True)
        dup = pd.concat([test, test.iloc[[0]]], ignore_index=True)
        try:
            uci_ff.predict(dup)
            v.check("rejects duplicate keys", False)
        except FinalForecastError:
            v.check("rejects duplicate keys", True)
        neg = test.copy()
        neg["units_sold_lag_1"] = -1.0
        try:
            uci_ff.predict(neg)
            v.check("rejects negative lag", False)
        except FinalForecastError:
            v.check("rejects negative lag", True)
        miss = test.copy()
        miss.loc[miss.index[0], "units_sold_lag_1"] = np.nan
        try:
            uci_ff.predict(miss)
            v.check("rejects missing lag_1 values", False)
        except FinalForecastError:
            v.check("rejects missing lag_1 values", True)

    print("\n[7] Reproducibility")
    if os.path.exists(FINAL_PRED_PATH) and registry:
        uci_ff = FinalForecaster.from_registry("uci_h1_phase8_lightgbm")
        feat = load_feature_dataset()
        sub, _, _ = prepare_features(feat, "UCI")
        test = _usable(sub)
        test = test[test["split"] == "test"].head(200).copy()
        a = uci_ff.predict(test, include_actual=True)
        b = uci_ff.predict(test, include_actual=True)
        dmax = float(np.max(np.abs(a["prediction"].to_numpy() - b["prediction"].to_numpy())))
        v.check("identical predictions on repeat", dmax < 1e-10, f"max_abs_diff={dmax}")
        syn_ff = FinalForecaster.from_registry("synthetic_h1_hurdle_th050")
        ssub, _, _ = prepare_features(feat, "SYNTHETIC")
        stest = _usable(ssub)
        stest = stest[stest["split"] == "test"].head(200).copy()
        a = syn_ff.predict(stest)
        b = syn_ff.predict(stest)
        dmax = float(np.max(np.abs(a["prediction"].to_numpy() - b["prediction"].to_numpy())))
        v.check("hurdle identical on repeat", dmax < 1e-10, f"max_abs_diff={dmax}")

    print("\n[8] Metrics vs Phase 8 reference")
    v.check("Phase 8 UCI WAPE reference", abs(PHASE8_TEST["UCI"]["WAPE"] - 79.4710) < 1e-4)
    v.check("Phase 8 SYNTHETIC WAPE reference", abs(PHASE8_TEST["SYNTHETIC"]["WAPE"] - 38.8923) < 1e-4)
    if os.path.exists(FINAL_PRED_PATH):
        pred = pd.read_parquet(FINAL_PRED_PATH)
        syn = pred[(pred["source_dataset"] == "SYNTHETIC") & (pred["horizon"] == 1)]
        den = float(np.sum(np.abs(syn["actual"])))
        wape = float(np.sum(np.abs(syn["actual"] - syn["prediction"])) / den * 100.0) if den else np.nan
        v.check("SYNTHETIC final WAPE near hurdle 26.25", abs(wape - 26.2505) < 0.15, f"WAPE={wape:.4f}")

    print("\n" + "=" * 60)
    print(f"VALIDATION RESULT: {v.summary()}")
    if v.failed:
        for r in v.results:
            if r["status"] == "FAIL":
                print(f"  FAIL: {r['name']}: {r['detail']}")
    print("=" * 60)
    if os.path.exists(METADATA_PATH):
        meta = json.load(open(METADATA_PATH, encoding="utf-8"))
        meta["validation"] = {
            "passed": v.passed,
            "total": v.total,
            "summary": v.summary(),
        }
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
    return v


if __name__ == "__main__":
    result = run_validation()
    sys.exit(0 if result.failed == 0 else 1)
