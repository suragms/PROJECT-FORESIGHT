"""
Phase 10 orchestrator — experimental forecasting improvements.
Never writes to Phase 8/9 artifact paths.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml_forecasting import load_feature_dataset
from src.phase10_common import (
    FEATURES_PATH,
    HORIZONS,
    METADATA_PATH,
    PHASE8_FREEZE_FILES,
    PHASE8_TEST,
    PHASE9_FREEZE_FILES,
    PHASE10_DIR,
    RANDOM_STATE,
    REGISTRY_PATH,
    REPORT_PATH,
    THRESHOLDS,
    ExperimentRegistry,
    ensure_dirs,
    hashes_unchanged,
    snapshot_hashes,
)
from src.phase10_direct_horizon import run_direct_horizon
from src.phase10_hurdle_forecasting import run_hurdle_forecasting
from src.phase10_hyperparameter_tuning import run_hyperparameter_tuning
from src.phase10_intermittent_baselines import run_intermittent_baselines
from src.phase10_prediction_intervals import run_prediction_intervals


def _pkg(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "not-installed"


def _decision(bundle: dict) -> dict:
    cmp_h = bundle["hurdle"]["comparisons"].get("SYNTHETIC", {})
    hurdle_imp = cmp_h.get("wape_improvement_pct") if not cmp_h.get("skipped") else None
    zero_p8 = cmp_h.get("zero_pos_phase8")
    zero_h = cmp_h.get("zero_pos_hurdle")
    hz = bundle["direct"]["comparison"]
    direct_help = False
    if hz is not None and not hz.empty:
        long_h = hz[hz["horizon"] >= 7]
        if len(long_h):
            direct_help = bool((long_h["wape_improvement_pct"] > 5).any())
    hpo = bundle["hpo"]["summary"]
    hpo_beats = False
    if hpo is not None and not hpo.empty:
        for _, r in hpo.iterrows():
            src = r["source_dataset"]
            if r["WAPE"] + 0.05 < PHASE8_TEST[src]["WAPE"] and not r["best_is_phase8_config"]:
                hpo_beats = True
    structural = False
    if hurdle_imp is not None and hurdle_imp > 5:
        structural = True
    if zero_p8 and zero_h is not None and zero_h < zero_p8 - 10:
        structural = True
    if direct_help:
        structural = True

    if structural and (hurdle_imp or 0) > 15 and direct_help:
        option, label = "A", "Major improvement"
        text = "Structural models substantially outperform frozen LightGBM."
    elif structural or hpo_beats:
        option, label = "B", "Targeted improvement"
        text = (
            "Some structural improvements clearly help, but the original LightGBM remains useful."
        )
    elif hurdle_imp is not None and hurdle_imp < -5 and not direct_help:
        option, label = "D", "Regression"
        text = "New approaches perform worse and should not replace the frozen model."
    else:
        option, label = "C", "No meaningful improvement"
        text = "The frozen LightGBM remains competitive."
    return {
        "option": option,
        "label": label,
        "statement": text,
        "hurdle_wape_improvement_pct": hurdle_imp,
        "direct_helps_long_horizon": direct_help,
        "hpo_beats_phase8_on_test": hpo_beats,
    }


def write_report(bundle: dict) -> str:
    hurdle = bundle["hurdle"]
    inter = bundle["intermittent"]
    direct = bundle["direct"]
    pi = bundle["intervals"]
    hpo = bundle["hpo"]
    dec = bundle["decision"]

    def tbl(df):
        if df is None or (hasattr(df, "empty") and df.empty):
            return "_n/a_"
        return "```\n" + df.to_string(index=False) + "\n```"

    syn_h = hurdle["results"].get("SYNTHETIC", {})
    syn_c = hurdle["comparisons"].get("SYNTHETIC", {})
    lines = []
    lines.append("# Phase 10 — Forecasting Strategy Improvement\n")
    lines.append(f"**Status:** COMPLETE  \n**Validation:** {bundle.get('validation_summary', 'pending')}\n")
    lines.append("## 1. Executive Summary\n")
    lines.append(f"**Decision (Option {dec['option']} — {dec['label']}):** {dec['statement']}\n")
    if syn_c and not syn_c.get("skipped"):
        lines.append(
            f"SYNTHETIC hurdle vs Phase 8 TEST: WAPE {syn_c['phase8']['WAPE']} -> "
            f"{syn_c['hurdle']['WAPE']} ({syn_c['wape_improvement_pct']:+.2f}%). "
            f"Zero-day positive prediction rate {syn_c['zero_pos_phase8']} -> "
            f"{syn_c['zero_pos_hurdle']}.\n"
        )
    lines.append(
        "Phase 8 LightGBM and Phase 9 stability artifacts were not modified.\n"
    )

    lines.append("## 2. Phase 8 Baseline\n")
    lines.append(
        f"- UCI LightGBM TEST: WAPE={PHASE8_TEST['UCI']['WAPE']}, "
        f"MAE={PHASE8_TEST['UCI']['MAE']}, sMAPE={PHASE8_TEST['UCI']['sMAPE']}\n"
        f"- SYNTHETIC LightGBM TEST: WAPE={PHASE8_TEST['SYNTHETIC']['WAPE']}, "
        f"MAE={PHASE8_TEST['SYNTHETIC']['MAE']}, sMAPE={PHASE8_TEST['SYNTHETIC']['sMAPE']}\n"
        "- Selection remains frozen; Phase 10 models are experimental comparators.\n"
    )

    lines.append("## 3. Zero-Demand Problem\n")
    lines.append(
        "SYNTHETIC TEST is ~61.27% zero-demand. Phase 8 LightGBM predicted positive "
        "demand on 82.64% of those zeros (zero-day MAE 1.44). UCI TEST has no coded "
        "zero-demand rows (invoice-day grain), so a hurdle model is not identified there.\n"
    )

    lines.append("## 4. Hurdle Model\n")
    lines.append(
        "Stage 1: LightGBM classifier P(demand>0), `is_unbalance=True`. "
        "Stage 2: LightGBM regressor trained only on actual demand>0. "
        f"Thresholds tried on validation only: {list(THRESHOLDS)}. "
        "Objective: min validation WAPE, then MAE, then nonzero MAE.\n"
    )
    for src, r in hurdle["results"].items():
        if r.get("skipped"):
            lines.append(f"### {src}\nSkipped: {r.get('skip_reason')}\n")
            continue
        lines.append(f"### {src}\n")
        lines.append(f"- Best threshold (validation): **{r['best_threshold']:.2f}**\n")
        lines.append(f"- Train zeros: {r['train_zero_share_pct']}%\n")
        vc = r.get("val_classifier_selected", {})
        lines.append(
            f"- Val classifier at selected threshold: ROC-AUC={vc.get('roc_auc')}, "
            f"PR-AUC={vc.get('pr_auc')}, F1={vc.get('f1')}, precision={vc.get('precision')}, "
            f"recall={vc.get('recall')}, Brier={vc.get('brier')}\n"
        )
        tm = r["test_metrics"]
        lines.append(
            f"- TEST hurdle: WAPE={tm['WAPE']} MAE={tm['MAE']} sMAPE={tm['sMAPE']} "
            f"bias={tm['bias']} zero_MAE={tm['zero_mae']} nonzero_MAE={tm['nonzero_mae']} "
            f"zero_pos_rate={tm['zero_positive_prediction_rate']}\n"
        )
        lines.append("Validation threshold table:\n")
        lines.append(tbl(r["threshold_table"]))
        lines.append("\n")
    if syn_c and not syn_c.get("skipped"):
        lines.append("Matched TEST comparison vs Phase 8:\n")
        lines.append(
            f"- Phase 8 WAPE={syn_c['phase8']['WAPE']} vs hurdle {syn_c['hurdle']['WAPE']}\n"
            f"- Phase 8 zero-pos-rate={syn_c['zero_pos_phase8']} vs hurdle {syn_c['zero_pos_hurdle']}\n"
            f"- Phase 8 nonzero MAE={syn_c['phase8']['nonzero_mae']} vs hurdle {syn_c['hurdle']['nonzero_mae']}\n"
        )

    lines.append("## 5. Intermittent Baselines\n")
    lines.append(
        "Croston / SBA / TSB / Naive implemented as rolling 1-step per series "
        "(alpha=0.1, TSB beta=0.1). Forecast uses only history before the origin, "
        "then updates with the realized actual. UCI skipped: zeros are not coded.\n"
    )
    lines.append(tbl(inter.get("summary")))
    for src, note in inter.get("skip_notes", {}).items():
        if src == "UCI" or (inter.get("summary") is None or (hasattr(inter.get("summary"), "empty") and inter["summary"].empty)):
            lines.append(f"- **{src}:** {note}\n")

    lines.append("## 6. Direct Multi-Horizon\n")
    lines.append(
        "Each horizon has its own LightGBM predicting `units_sold` at t+h from origin-t "
        "features. Target calendar (known in advance) is attached; demand lags stay at t. "
        "Train rows require target_date within the train window (no test labels in training).\n"
    )
    lines.append(tbl(direct.get("summary")))
    lines.append("\nRecursive (Phase 9) vs direct:\n")
    lines.append(tbl(direct.get("comparison")))
    lines.append(
        "\nPopulations are not identical: Phase 9 used strided origins and a 400-series "
        "UCI cap. Directional comparison is still informative.\n"
    )

    lines.append("## 7. Prediction Intervals\n")
    lines.append(
        "Quantile LightGBM (objective=quantile) at 0.10/0.50/0.90. "
        "Raw quantiles may be negative; they are clipped at 0 because demand cannot be "
        "negative, then sorted to restore P10<=P50<=P90. Crossing counts are reported "
        "before reorder.\n"
    )
    lines.append(tbl(pi.get("summary")))

    lines.append("## 8. Hyperparameter Optimization\n")
    lines.append(
        "Eight LightGBM configs, selected by chronological validation WAPE then MAE. "
        "TEST evaluated once after selection. Config 0 is the Phase 8 frozen hyperparameter set.\n"
    )
    lines.append(tbl(hpo.get("grid")))
    lines.append("\nSelected TEST scores:\n")
    lines.append(tbl(hpo.get("summary")))

    lines.append("## 9. Final Model Comparison\n")
    rows = []
    for src in ["UCI", "SYNTHETIC"]:
        p8 = PHASE8_TEST[src]
        rows.append({
            "Dataset": src, "Model": "Phase8_LightGBM", "Horizon": 1,
            "MAE": p8["MAE"], "RMSE": p8["RMSE"], "sMAPE": p8["sMAPE"], "WAPE": p8["WAPE"],
            "Bias": "see Phase 9", "Zero-demand": "see Phase 9", "Training time": "frozen",
        })
        r = hurdle["results"].get(src, {})
        if not r.get("skipped") and r.get("test_metrics"):
            tm = r["test_metrics"]
            rows.append({
                "Dataset": src, "Model": f"Hurdle_th{r['best_threshold']:.2f}", "Horizon": 1,
                "MAE": tm["MAE"], "RMSE": tm["RMSE"], "sMAPE": tm["sMAPE"], "WAPE": tm["WAPE"],
                "Bias": tm["bias"], "Zero-demand": tm.get("zero_positive_prediction_rate"),
                "Training time": r.get("training_time_sec"),
            })
        if hpo.get("summary") is not None:
            hs = hpo["summary"]
            sub = hs[hs["source_dataset"] == src]
            if len(sub):
                rr = sub.iloc[0]
                rows.append({
                    "Dataset": src, "Model": f"HPO_cfg{int(rr['best_config_id'])}", "Horizon": 1,
                    "MAE": rr["MAE"], "RMSE": rr["RMSE"], "sMAPE": rr["sMAPE"], "WAPE": rr["WAPE"],
                    "Bias": rr["bias"], "Zero-demand": "n/a", "Training time": "see grid",
                })
    lines.append(tbl(pd.DataFrame(rows)))

    lines.append("\n## 10. Recommended Model\n")
    rec_model = "Keep Phase 8 LightGBM as the production 1-step point forecast."
    if syn_c and not syn_c.get("skipped"):
        z_better = (syn_c["zero_pos_hurdle"] is not None and syn_c["zero_pos_phase8"] is not None
                    and syn_c["zero_pos_hurdle"] < syn_c["zero_pos_phase8"] - 5)
        wape_ok = syn_c["hurdle"]["WAPE"] <= syn_c["phase8"]["WAPE"] * 1.05
        nz_ok = syn_c["hurdle"]["nonzero_mae"] <= syn_c["phase8"]["nonzero_mae"] * 1.10
        if z_better and wape_ok and nz_ok:
            rec_model = (
                f"SYNTHETIC: hurdle (threshold={syn_h.get('best_threshold')}) is preferred "
                "when zero-day false positives matter and overall WAPE/nonzero MAE stay close. "
                "UCI: keep Phase 8 LightGBM (hurdle not applicable)."
            )
        elif syn_c["hurdle"]["WAPE"] < syn_c["phase8"]["WAPE"]:
            rec_model = (
                "SYNTHETIC hurdle improves TEST WAPE; still review nonzero-demand MAE before replacement. "
                "UCI: keep Phase 8 LightGBM."
            )
    lines.append(rec_model + "\n")
    lines.append(
        "Do not auto-replace Phase 8 artifacts. Direct models may be used for longer-horizon "
        "planning if they improve WAPE at h>=7. Quantile P10/P90 are diagnostic intervals, "
        "not a replacement point forecast unless P50 beats LightGBM on WAPE.\n"
    )

    lines.append("## 11. Remaining Limitations\n")
    lines.append(
        "- Hurdle threshold is a hard zero; it does not produce calibrated expected demand "
        "E[y]=P(y>0)*E[y|y>0] unless used as a mixture (not selected here).\n"
        "- Direct vs recursive comparison uses different origin samples.\n"
        "- Quantile models are independent and can cross before reorder.\n"
        "- HPO search is small; not a full AutoML sweep.\n"
        "- UCI intermittency is not identified in the current grain.\n"
        "- No prediction intervals around the hurdle mixture.\n"
    )

    lines.append("## 12. Phase 11 Recommendation\n")
    lines.append(
        "Do **not** implement Phase 11 here. Evidence-based next step: inventory risk "
        "scoring / replenishment using the frozen Phase 8 1-step LightGBM as the demand "
        "engine, with optional SYNTHETIC hurdle overlay for zero-demand SKUs, and P10/P90 "
        "bands for safety-stock diagnostics. Productionize only after stakeholder review "
        "of zero-demand false positives vs missed demand.\n"
    )
    lines.append(f"\n## Overall decision\n\n**Option {dec['option']} — {dec['label']}**\n\n{dec['statement']}\n")
    lines.append("## Charts\n")
    for p in bundle.get("charts", []):
        rel = os.path.relpath(p, BASE_DIR) if os.path.isabs(str(p)) else p
        lines.append(f"- `{rel}`\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return REPORT_PATH


def write_metadata(bundle: dict, hashes8_before, hashes8_after, hashes9_before, hashes9_after) -> str:
    import lightgbm
    import numpy as np
    import sklearn

    meta = {
        "phase": 10,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "random_state": RANDOM_STATE,
        "features_path": FEATURES_PATH,
        "phase8_test_benchmarks": PHASE8_TEST,
        "hurdle_thresholds_considered": list(THRESHOLDS),
        "hurdle_threshold_selection": "min validation WAPE, then MAE, then nonzero MAE",
        "hurdle_best_threshold": {
            src: bundle["hurdle"]["results"][src].get("best_threshold")
            for src in bundle["hurdle"]["results"]
        },
        "horizon_definitions": {
            "horizons": list(HORIZONS),
            "method": "direct_lightgbm_origin_features_plus_known_target_calendar",
        },
        "quantile_definitions": {
            "taus": [0.1, 0.5, 0.9],
            "nonneg_clip": True,
            "reorder_if_crossing": True,
        },
        "metric_definitions": {
            "WAPE": "sum(|y-yhat|)/sum(|y|)*100; 0 if sum(|y|)=0",
            "sMAPE": "mean(2|y-yhat|/(|y|+|yhat|))*100; (0,0)->0",
            "bias": "mean(prediction - actual)",
            "residual": "actual - prediction",
        },
        "decision": bundle["decision"],
        "phase8_hashes_before": hashes8_before,
        "phase8_hashes_after": hashes8_after,
        "phase8_unchanged": hashes_unchanged(hashes8_before, hashes8_after)[0],
        "phase9_hashes_before": hashes9_before,
        "phase9_hashes_after": hashes9_after,
        "phase9_unchanged": hashes_unchanged(hashes9_before, hashes9_after)[0],
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return METADATA_PATH


def write_notebook() -> str:
    from src.generate_phase10_notebook import write_notebook as _wn
    return _wn()


def run_phase10() -> dict:
    ensure_dirs()
    h8b = snapshot_hashes(PHASE8_FREEZE_FILES)
    h9b = snapshot_hashes(PHASE9_FREEZE_FILES)
    print("[Phase 10] Phase 8/9 freeze snapshot taken.")
    df = load_feature_dataset()
    registry = ExperimentRegistry()

    hurdle = run_hurdle_forecasting(df, save=True)
    for src, r in hurdle["results"].items():
        registry.add(
            experiment_id=f"hurdle_{src.lower()}",
            dataset=src,
            model_type="hurdle_lightgbm",
            model_configuration={"classifier": "LGBMClassifier", "regressor": "LGBMRegressor",
                                 "params": "Phase8-like", "is_unbalance": True},
            features="Phase 8 contract",
            horizon=1,
            validation_method="chronological split column",
            threshold=r.get("best_threshold"),
            metrics=r.get("test_metrics"),
            training_time=r.get("training_time_sec"),
            status="skipped" if r.get("skipped") else "completed",
        )

    hw = None
    syn_c = hurdle["comparisons"].get("SYNTHETIC", {})
    if syn_c and not syn_c.get("skipped"):
        hw = syn_c["hurdle"]["WAPE"]
    inter = run_intermittent_baselines(df, save=True, hurdle_wape=hw)
    if inter.get("summary") is not None and not inter["summary"].empty:
        for _, row in inter["summary"].iterrows():
            registry.add(
                experiment_id=f"intermittent_{row['model']}",
                dataset=row["source_dataset"],
                model_type=row["model"],
                model_configuration={"alpha": 0.1, "beta": 0.1},
                features="univariate series history",
                horizon=1,
                validation_method="rolling 1-step on TEST with pre-origin history",
                threshold=None,
                metrics={"WAPE": row["WAPE"], "MAE": row["MAE"]},
                training_time=0,
            )

    direct = run_direct_horizon(df, save=True)
    for src, by_h in direct["by_source"].items():
        for h, rec in by_h.items():
            registry.add(
                experiment_id=f"direct_{src.lower()}_h{h}",
                dataset=src,
                model_type="lightgbm_direct",
                model_configuration={"params": "Phase8-like", "target_calendar": True},
                features="origin operational + known target calendar",
                horizon=h,
                validation_method="origin in split; target_date constrained to split window for train/val",
                threshold=None,
                metrics=rec.get("test_metrics"),
                training_time=rec.get("training_time_sec"),
                status="skipped" if rec.get("skipped") else "completed",
            )

    intervals = run_prediction_intervals(df, save=True)
    for src, r in intervals["results"].items():
        registry.add(
            experiment_id=f"quantile_{src.lower()}",
            dataset=src,
            model_type="lightgbm_quantile",
            model_configuration={"taus": [0.1, 0.5, 0.9], "nonneg_clip": True},
            features="Phase 8 contract",
            horizon=1,
            validation_method="chronological split",
            threshold=None,
            metrics={"coverage_pct": r["coverage_pct"], "mean_width": r["mean_width"],
                     "pinball": r["pinball"]},
            training_time=r.get("training_time_sec"),
        )

    hpo = run_hyperparameter_tuning(df, save=True)
    for src, r in hpo["results"].items():
        registry.add(
            experiment_id=f"hpo_{src.lower()}",
            dataset=src,
            model_type="lightgbm_hpo",
            model_configuration={"search": "8 configs", "best_config_id": r["best_config_id"]},
            features="Phase 8 contract",
            horizon=1,
            validation_method="chronological validation WAPE",
            threshold=None,
            metrics=r.get("test_metrics"),
            training_time=None,
        )

    charts = (
        hurdle.get("charts", []) + inter.get("charts", []) + direct.get("charts", [])
        + intervals.get("charts", []) + hpo.get("charts", [])
    )
    bundle = {
        "hurdle": hurdle,
        "intermittent": inter,
        "direct": direct,
        "intervals": intervals,
        "hpo": hpo,
        "charts": charts,
        "validation_summary": "pending",
    }
    bundle["decision"] = _decision(bundle)
    write_report(bundle)
    registry.save(REGISTRY_PATH)
    h8a = snapshot_hashes(PHASE8_FREEZE_FILES)
    h9a = snapshot_hashes(PHASE9_FREEZE_FILES)
    ok8, c8 = hashes_unchanged(h8b, h8a)
    ok9, c9 = hashes_unchanged(h9b, h9a)
    if not ok8:
        raise RuntimeError(f"Phase 8 artifacts changed: {c8}")
    if not ok9:
        raise RuntimeError(f"Phase 9 artifacts changed: {c9}")
    write_metadata(bundle, h8b, h8a, h9b, h9a)
    try:
        write_notebook()
    except Exception as e:
        print("[Phase 10] Notebook write skipped/failed:", e)
    print(f"[Phase 10] Decision Option {bundle['decision']['option']} — {bundle['decision']['label']}")
    print("[Phase 10] Phase 8 and Phase 9 artifacts unchanged.")
    return bundle


if __name__ == "__main__":
    run_phase10()
