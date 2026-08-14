"""
Phase 9 orchestrator — run analyses, write report/metadata, leave Phase 8 frozen.
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
from src.phase9_common import (
    FEATURES_PATH,
    FIGURES_DIR,
    HORIZONS,
    METADATA_PATH,
    PHASE8_TEST,
    PHASE9_DIR,
    RANDOM_STATE,
    REPORT_PATH,
    STABILITY_RULES,
    ensure_dirs,
    snapshot_phase8_hashes,
    hashes_unchanged,
)
from src.phase9_horizon_analysis import run_horizon_analysis
from src.phase9_residual_analysis import run_residual_analysis
from src.phase9_walk_forward import load_walk_forward, run_walk_forward


def _pkg(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "not-installed"


def _overall_conclusion(wf_summaries: dict, horizon: pd.DataFrame, zero: pd.DataFrame) -> dict:
    import numpy as np

    labels = [wf_summaries[s]["stability"]["label"] for s in ["UCI", "SYNTHETIC"]]
    degr = {}
    for src in ["UCI", "SYNTHETIC"]:
        sub = horizon[horizon["source_dataset"] == src].set_index("horizon")["WAPE"]
        if 1 in sub.index and 30 in sub.index and sub.loc[1] > 0:
            degr[src] = float(sub.loc[30] / sub.loc[1])
        else:
            degr[src] = np.nan
    syn_zero = zero[zero["source_dataset"] == "SYNTHETIC"]
    zero_fp = float(syn_zero["zero_actual_positive_prediction_rate"].iloc[0]) if len(syn_zero) else np.nan

    # Decision rule applied to evidence:
    # Unstable if any source Unstable OR horizon 30/1 WAPE ratio >= 3
    # Moderately Stable if any Moderately Stable OR horizon ratio >= 1.75 OR SYN zero FP >= 80%
    # else Stable
    if "Unstable" in labels or any(np.isfinite(v) and v >= 3.0 for v in degr.values()):
        option = "C"
        text = (
            "Phase 8 aggregate test performance is insufficient evidence of robustness; "
            "additional modeling work is required."
        )
        label = "Unstable"
    elif (
        "Moderately Stable" in labels
        or any(np.isfinite(v) and v >= 1.75 for v in degr.values())
        or (np.isfinite(zero_fp) and zero_fp >= 80)
    ):
        option = "B"
        text = (
            "LightGBM is promising but requires targeted improvements before productionization."
        )
        label = "Moderately Stable"
    else:
        option = "A"
        text = (
            "Phase 8 LightGBM is sufficiently stable for progression toward "
            "production-oriented forecasting."
        )
        label = "Stable"
    return {
        "option": option,
        "label": label,
        "statement": text,
        "horizon_wape_ratio_30_over_1": degr,
        "synthetic_zero_positive_pred_rate": zero_fp,
        "walk_forward_labels": {s: wf_summaries[s]["stability"]["label"] for s in ["UCI", "SYNTHETIC"]},
    }


def write_report(bundle: dict) -> str:
    wf = bundle["walk_forward"]
    res = bundle["residual"]
    hz = bundle["horizon"]
    conc = bundle["conclusion"]
    overall = res["overall"]
    zero = res["zero_demand"]
    entities = res["entities"]
    regimes = res["regimes"]
    diagnostics = res.get("diagnostics")
    folds = wf["folds"]
    hsum = hz["summary"]

    def _tbl(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "_n/a_"
        return "```\n" + df.to_string(index=False) + "\n```"

    syn_ent = entities[entities["source_dataset"] == "SYNTHETIC"].sort_values("WAPE")
    best_store = syn_ent.iloc[0] if len(syn_ent) else None
    worst_store = syn_ent.iloc[-1] if len(syn_ent) else None

    lines = []
    lines.append("# Phase 9 — Advanced Forecast Validation, Stability, Residual & Horizon Analysis\n")
    lines.append(f"**Status:** COMPLETE  \n**Validation:** {bundle.get('validation_summary', 'pending')}\n")
    lines.append("## 1. Executive Summary\n")
    lines.append(
        f"**Conclusion (Option {conc['option']} — {conc['label']}):** {conc['statement']}\n"
    )
    lines.append("Walk-forward stability labels:\n")
    for src, lab in conc["walk_forward_labels"].items():
        s = wf["summaries"][src]["stability"]
        lines.append(
            f"- **{src}:** {lab} (mean WAPE={s.get('mean_wape')}, CV={s.get('cv_wape')}, "
            f"max/min={s.get('range_ratio')})\n"
        )
    lines.append(
        "Phase 8 TEST LightGBM remains the frozen production candidate. Phase 9 did not "
        "retrain or replace those artifacts; walk-forward retrains *copies* of the same "
        "configuration on earlier windows only.\n"
    )

    lines.append("## 2. Methodology\n")
    lines.append("### Walk-forward\n")
    lines.append(
        "Expanding-window validation: each fold trains LightGBM (Phase 8 hyperparameters, "
        f"`random_state={RANDOM_STATE}`) on all observations with `date <= train_end`, "
        "evaluates on `val_start..val_end`. Training always precedes validation; sources "
        "are never mixed. Grain: `date + source_dataset + entity_id + product_key`.\n"
    )
    lines.append("### Residual analysis\n")
    lines.append(
        "`residual = actual - prediction` on frozen Phase 8 TEST predictions. "
        "`bias = mean(prediction - actual)` matches Phase 8.\n"
    )
    lines.append("### Horizon analysis\n")
    lines.append(
        "Phase 8 is a **1-observation-ahead** model. Horizons "
        f"{list(HORIZONS)} are evaluated with **iterated recursive** forecasts from the "
        "frozen LightGBM (same approach as `generate_multi_step_forecast` in the legacy "
        "engine). Lag/rolling/trend features are updated from the prediction buffer. "
        "Price/promo/inventory are held at origin values (no future leakage). "
        "Recursive h=1 is therefore **not identical** to Phase 8 TEST: Phase 8 uses "
        "target-row operational fields (especially SYNTHETIC inventory). UCI has no "
        "inventory features, so recursive h=1 WAPE should closely match Phase 8 TEST. "
        "Horizon unit = observation step (calendar day for SYNTHETIC; next observed "
        "date for gappy UCI). UCI recursive evaluation uses the 400 longest eligible "
        "test series (deterministic compute cap).\n"
    )
    lines.append("### Zero-demand\n")
    lines.append(
        "Split TEST rows into `actual == 0` vs `actual != 0`. WAPE is undefined/zero-safe "
        "when `sum(|actual|)=0`. sMAPE treats (0,0) as 0 and remains large when the model "
        "predicts positive demand against true zeros.\n"
    )
    lines.append("### Store/entity\n")
    lines.append("Per-entity MAE/RMSE/WAPE/bias recomputed from Phase 8 TEST predictions.\n")
    lines.append("### Stability rule\n")
    lines.append(f"{STABILITY_RULES}\n")

    lines.append("## 3. Walk-Forward Results\n")
    lines.append(_tbl(folds[[
        "source_dataset", "fold", "train_end", "val_start", "val_end",
        "train_rows", "val_rows", "MAE", "RMSE", "sMAPE", "WAPE", "bias",
        "overprediction_pct", "underprediction_pct",
    ]]))
    lines.append("\n")
    for src, s in wf["summaries"].items():
        lines.append(f"### {src} fold summary\n")
        lines.append(f"- Stability: **{s['stability']['label']}** — {s['stability']['reason']}\n")
        for metric in ["WAPE", "MAE", "RMSE", "sMAPE", "bias"]:
            st = s[metric]
            lines.append(
                f"- {metric}: mean={st['mean']}, median={st['median']}, std={st['std']}, "
                f"min={st['min']}, max={st['max']}\n"
            )

    lines.append("## 4. Residual Analysis\n")
    lines.append(_tbl(overall))
    lines.append("\nDemand regimes (zero = actual 0; low/medium/high = tertiles of positive demand):\n")
    lines.append(_tbl(regimes))
    if diagnostics is not None and len(diagnostics) > 0:
        lines.append("\nResidual vs actual / prediction diagnostics (heteroscedasticity):\n")
        lines.append(_tbl(diagnostics))
    lines.append("\n")
    diag_sources = set()
    if diagnostics is not None and len(diagnostics) > 0:
        diag_sources = set(diagnostics["source_dataset"].astype(str))
    for _, r in overall.iterrows():
        src = str(r["source_dataset"])
        lines.append(f"### {src}\n")
        lines.append(
            f"- Mean residual (actual-pred)={r['mean_residual']}, bias (pred-actual)={r['bias']}\n"
        )
        lines.append(
            f"- Over-prediction {r['overprediction_pct']}% / under-prediction {r['underprediction_pct']}%\n"
        )
        if src in diag_sources:
            d = diagnostics[diagnostics["source_dataset"].astype(str) == src].iloc[0]
            lines.append(
                f"- |error| vs actual correlation={d['corr_abs_error_actual']}; "
                f"|error| vs prediction correlation={d['corr_abs_error_prediction']}. "
                f"Low-actual MAE={d['low_actual_mae']}, high-actual MAE={d['high_actual_mae']}.\n"
            )

    lines.append("## 5. Forecast Horizon Results\n")
    lines.append(_tbl(hsum))
    lines.append("\n")
    for src in ["UCI", "SYNTHETIC"]:
        sub = hsum[hsum["source_dataset"] == src].sort_values("horizon")
        if sub.empty:
            continue
        w = sub.set_index("horizon")["WAPE"]
        increasing = bool((w.diff().dropna() >= -1e-6).all())
        lines.append(
            f"- **{src}:** WAPE from h=1 ({w.iloc[0]:.4f}) to h={int(w.index[-1])} "
            f"({w.iloc[-1]:.4f}); monotone non-decreasing={increasing}; "
            f"ratio h30/h1={conc['horizon_wape_ratio_30_over_1'].get(src)}\n"
        )

    lines.append("## 6. Zero-Demand Results\n")
    lines.append(_tbl(zero))
    lines.append("\n")
    lines.append(
        "sMAPE is unstable when actuals are zero because the denominator is "
        "`|y|+|ŷ|`. If ŷ>0 and y=0, each term equals 2, so sMAPE can approach 200% "
        "even when absolute errors are small. WAPE for a pure-zero subset has "
        "denominator 0 and is reported as 0.0 by the Phase 7/8 zero-safe rule "
        "(not a claim of perfect accuracy). Prefer MAE on the zero-demand slice.\n"
    )

    lines.append("## 7. Store/Entity Stability\n")
    lines.append(_tbl(syn_ent[["entity_id", "n", "MAE", "RMSE", "WAPE", "bias"]] if len(syn_ent) else pd.DataFrame()))
    if best_store is not None:
        lines.append(
            f"\nBest: **{best_store['entity_id']}** WAPE={best_store['WAPE']:.4f}, "
            f"MAE={best_store['MAE']:.4f}. Worst: **{worst_store['entity_id']}** "
            f"WAPE={worst_store['WAPE']:.4f}, MAE={worst_store['MAE']:.4f}.\n"
        )
        lines.append(
            "Phase 8 reported STORE_005 WAPE≈37.82 and STORE_003 WAPE≈40.23. "
            "Phase 9 recomputes from the same TEST predictions; small differences can "
            "arise from rounding. Values above are the recalculated figures.\n"
        )

    lines.append("## 8. Metric Interpretation\n")
    lines.append(
        "- **MAE:** scale of typical absolute error in units; comparable within a source, "
        "not across UCI vs SYNTHETIC volume scales.\n"
        "- **RMSE:** penalizes large misses (UCI bulk orders inflate RMSE).\n"
        "- **WAPE:** volume-weighted; robust to zeros in the mixed set because the "
        "denominator is total actual volume. Uninformative on an all-zero slice.\n"
        "- **sMAPE:** bounded but inflates on sparse/zero demand when predictions are "
        "positive. High SYNTHETIC sMAPE does not by itself mean the model is unusable.\n"
    )

    lines.append("## 9. Phase 9 Findings\n")
    lines.append("### Confirmed finding\n")
    lines.append(
        f"- Frozen Phase 8 TEST WAPE remains UCI {PHASE8_TEST['UCI']['WAPE']}, "
        f"SYNTHETIC {PHASE8_TEST['SYNTHETIC']['WAPE']}.\n"
    )
    lines.append(
        f"- Walk-forward labels: UCI={conc['walk_forward_labels']['UCI']}, "
        f"SYNTHETIC={conc['walk_forward_labels']['SYNTHETIC']}.\n"
    )
    uci_f2 = folds[(folds["source_dataset"] == "UCI") & (folds["fold"] == 2)]
    if len(uci_f2):
        lines.append(
            f"- UCI fold 2 (2011-01-01..2011-04-30) WAPE={float(uci_f2.iloc[0]['WAPE']):.4f} "
            "is the worst UCI window (post-holiday wholesale gap).\n"
        )
    syn_z = zero[zero["source_dataset"] == "SYNTHETIC"]
    if len(syn_z):
        lines.append(
            f"- SYNTHETIC TEST zero-demand share={float(syn_z.iloc[0]['zero_share_pct']):.2f}%; "
            f"P(pred>0|actual=0)={float(syn_z.iloc[0]['zero_actual_positive_prediction_rate']):.2f}%.\n"
        )
    if best_store is not None:
        lines.append(
            f"- Recalculated SYNTHETIC store range: {best_store['entity_id']} "
            f"WAPE={best_store['WAPE']:.4f} through {worst_store['entity_id']} "
            f"WAPE={worst_store['WAPE']:.4f}.\n"
        )
    lines.append("### Possible explanation\n")
    lines.append(
        "- SYNTHETIC intermittency (~60% zero-demand TEST days) drives high sMAPE and "
        "positive predictions on zero days.\n"
        "- Recursive error accumulation can increase WAPE with horizon even if 1-step WAPE is acceptable.\n"
        "- UCI low-demand regime over-prediction (mean actual << mean prediction) is consistent "
        "with a model pulled toward the mean of a heavy-tailed wholesale series.\n"
        "- UCI fold-2 WAPE spike is consistent with a thinner post-holiday invoice calendar, "
        "not necessarily a broken model class.\n"
        "- SYNTHETIC recursive h=1 WAPE is worse than Phase 8 TEST because origin-held "
        "inventory/promo features remove same-day operational information the 1-step model uses.\n"
    )
    lines.append("### Limitation\n")
    lines.append(
        "- Recursive multi-step is not a direct h-step model; it is the only leakage-safe "
        "multi-horizon method supported by the Phase 8 1-step architecture.\n"
        "- UCI horizon analysis uses the 400 longest series (documented cap).\n"
        "- SYNTHETIC recursive WAPE is not monotone in horizon (origins near the TEST end drop out).\n"
        "- No prediction intervals in this phase.\n"
        "- Walk-forward retrains LightGBM on each fold (same config) but does not tune hyperparameters.\n"
    )

    lines.append("## 10. Phase 10 Recommendations\n")
    lines.append("Do **not** implement these here. Evidence-based next steps:\n")
    lines.append("1. Intermittent-demand / hurdle / Croston-style methods for SYNTHETIC zeros.\n")
    lines.append("2. Quantile regression or LightGBM quantile loss for prediction intervals.\n")
    lines.append("3. Direct multi-horizon models (separate heads for h=7/14/30) vs recursive.\n")
    lines.append("4. Hyperparameter optimization only after residual/zero-demand fixes.\n")
    lines.append("5. Hierarchical store-SKU reconciliation for SYNTHETIC.\n")

    lines.append(f"\n## Overall decision\n\n**Option {conc['option']} — {conc['label']}**\n\n{conc['statement']}\n")
    lines.append("## Charts\n")
    for p in bundle.get("charts", []):
        rel = os.path.relpath(p, BASE_DIR) if os.path.isabs(str(p)) else p
        lines.append(f"- `{rel}`\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return REPORT_PATH


def write_metadata(bundle: dict, hashes_before: dict, hashes_after: dict) -> str:
    import lightgbm
    import numpy as np
    import pandas as pd
    import sklearn

    meta = {
        "phase": 9,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "xgboost": _pkg("xgboost"),
            "matplotlib": _pkg("matplotlib"),
            "joblib": _pkg("joblib"),
        },
        "random_state": RANDOM_STATE,
        "features_path": FEATURES_PATH,
        "features_rows": int(len(bundle["features"])),
        "phase8_frozen_models": {
            "UCI": "lightgbm",
            "SYNTHETIC": "lightgbm",
            "paths": {
                "UCI": os.path.join(BASE_DIR, "models", "uci_best_model.joblib"),
                "SYNTHETIC": os.path.join(BASE_DIR, "models", "synthetic_best_model.joblib"),
            },
            "test_benchmarks": PHASE8_TEST,
        },
        "lightgbm_walk_forward_params": {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
        },
        "fold_definitions": {
            "UCI": [dict(x) for x in __import__("src.phase9_common", fromlist=["FOLDS"]).FOLDS["UCI"]],
            "SYNTHETIC": [dict(x) for x in __import__("src.phase9_common", fromlist=["FOLDS"]).FOLDS["SYNTHETIC"]],
        },
        "horizon_definitions": {
            "horizons": list(HORIZONS),
            "unit": "observation_step_at_forecast_grain",
            "method": "iterated_recursive_frozen_lightgbm",
            "uci_series_cap": 400,
            "origin_stride": {"UCI": 5, "SYNTHETIC": 7},
        },
        "metric_definitions": {
            "MAE": "mean(|y-yhat|)",
            "RMSE": "sqrt(mean((y-yhat)^2))",
            "WAPE": "sum(|y-yhat|)/sum(|y|)*100; 0 if sum(|y|)=0",
            "sMAPE": "mean(2|y-yhat|/(|y|+|yhat|))*100; (0,0)->0",
            "bias": "mean(prediction - actual)  # Phase 8 convention",
            "residual": "actual - prediction",
        },
        "stability_rules": STABILITY_RULES,
        "walk_forward_summaries": {
            src: {
                "label": s["stability"]["label"],
                "cv_wape": s["stability"]["cv_wape"],
                "range_ratio": s["stability"]["range_ratio"],
                "WAPE": s["WAPE"],
            }
            for src, s in bundle["walk_forward"]["summaries"].items()
        },
        "conclusion": bundle["conclusion"],
        "phase8_hashes_before": hashes_before,
        "phase8_hashes_after": hashes_after,
        "phase8_unchanged": hashes_unchanged(hashes_before, hashes_after)[0],
        "evaluation_windows": {
            "UCI_test": "2011-09-26 to 2011-12-09",
            "SYNTHETIC_test": "2025-08-07 to 2025-12-31",
        },
        "walk_forward_reused": bool(bundle.get("walk_forward_reused")),
        "demand_regimes": {
            "zero": "actual == 0",
            "low_medium_high": "tertiles of strictly positive actual demand, computed per source",
        },
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return METADATA_PATH


def run_phase9() -> dict:
    ensure_dirs()
    hashes_before = snapshot_phase8_hashes()
    print("[Phase 9] Phase 8 freeze snapshot taken.")
    df = load_feature_dataset()
    existing_wf = os.path.join(PHASE9_DIR, "walk_forward_folds.parquet")
    if os.path.exists(existing_wf):
        print("[Phase 9.1] Reusing saved walk-forward folds (LightGBM config already evaluated).")
        wf = load_walk_forward(refresh_charts=True)
        wf_reused = True
    else:
        wf = run_walk_forward(df, save=True)
        wf_reused = False
    res = run_residual_analysis(save=True)
    hz = run_horizon_analysis(df, save=True)
    conclusion = _overall_conclusion(wf["summaries"], hz["summary"], res["zero_demand"])
    charts = wf["charts"] + res["charts"] + hz["charts"]
    bundle = {
        "features": df,
        "walk_forward": wf,
        "residual": res,
        "horizon": hz,
        "conclusion": conclusion,
        "charts": charts,
        "validation_summary": "pending",
        "walk_forward_reused": wf_reused,
    }
    write_report(bundle)
    from src.generate_phase9_notebook import write_notebook
    write_notebook()
    hashes_after = snapshot_phase8_hashes()
    ok, changed = hashes_unchanged(hashes_before, hashes_after)
    if not ok:
        raise RuntimeError(f"Phase 8 artifacts changed during Phase 9: {changed}")
    write_metadata(bundle, hashes_before, hashes_after)
    print(f"[Phase 9] Conclusion Option {conclusion['option']} — {conclusion['label']}")
    print("[Phase 9] Phase 8 artifacts unchanged.")
    return bundle


def rebuild_report_from_disk(validation_summary: str = "pending") -> str:
    """Rewrite the markdown report from saved Phase 9 artifacts (no retraining)."""
    wf = load_walk_forward(refresh_charts=False)
    res = run_residual_analysis(save=False)
    hsum = pd.read_parquet(os.path.join(PHASE9_DIR, "horizon_summary.parquet"))
    hz = {"summary": hsum, "detail": pd.DataFrame(), "charts": [
        os.path.join(FIGURES_DIR, "horizon_performance_uci.png"),
        os.path.join(FIGURES_DIR, "horizon_performance_synthetic.png"),
    ]}
    conclusion = _overall_conclusion(wf["summaries"], hsum, res["zero_demand"])
    charts = []
    if os.path.isdir(FIGURES_DIR):
        charts = [os.path.join(FIGURES_DIR, f) for f in sorted(os.listdir(FIGURES_DIR)) if f.endswith(".png")]
    bundle = {
        "walk_forward": wf,
        "residual": res,
        "horizon": hz,
        "conclusion": conclusion,
        "charts": charts,
        "validation_summary": validation_summary,
    }
    path = write_report(bundle)
    print(f"[Phase 9] Report rebuilt: {path}")
    return path


if __name__ == "__main__":
    run_phase9()
