"""
Phase 10.1 — Two-stage hurdle (zero-demand) forecasting.

Stage 1: P(demand > 0) LightGBM classifier, train-fitted preprocessor.
Stage 2: LightGBM regressor trained only on actual demand > 0.
Threshold selected on VALIDATION WAPE (not TEST).

Does not overwrite Phase 8/9 artifacts.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml_forecasting import (
    FeaturePreprocessor,
    _predict_nonneg,
    feature_lists_for_source,
    load_feature_dataset,
    prepare_features,
)
from src.phase10_common import (
    FIGURES_DIR,
    GRAIN,
    LGB_POINT_PARAMS,
    ML_PRED_PATH,
    PHASE10_DIR,
    RANDOM_STATE,
    TARGET,
    THRESHOLDS,
    apply_mpl_style,
    ensure_dirs,
    forecast_metrics,
)

MIN_TRAIN_ZERO_SHARE = 0.05


def _usable(df_src: pd.DataFrame) -> pd.DataFrame:
    out = df_src.copy()
    if "units_sold_lag_1" in out.columns:
        out = out[out["units_sold_lag_1"].notna()].copy()
    return out.sort_values(GRAIN).reset_index(drop=True)


def _split_frames(df_src: pd.DataFrame, numeric: list[str], cats: list[str]) -> dict[str, pd.DataFrame]:
    feat = numeric + cats
    parts = {}
    for sp in ["train", "validation", "test"]:
        part = df_src[df_src["split"] == sp].copy()
        parts[sp] = part
        parts[f"{sp}_X"] = part[feat]
        parts[f"{sp}_y"] = part[TARGET].astype(float).to_numpy()
        parts[f"{sp}_ybin"] = (part[TARGET].astype(float) > 0).astype(int).to_numpy()
    parts["feat_cols"] = feat
    return parts


def _clf_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    out = {
        "threshold": threshold,
        "n": int(len(y_true)),
        "actual_positive_rate": round(100.0 * float(y_true.mean()), 2) if len(y_true) else np.nan,
        "predicted_positive_rate": round(100.0 * float(pred.mean()), 2) if len(y_true) else np.nan,
        "accuracy": round(float(accuracy_score(y_true, pred)), 4) if len(y_true) else np.nan,
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, proba)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, proba)), 4)
        out["brier"] = round(float(brier_score_loss(y_true, proba)), 4)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
        out["brier"] = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n_zero = tn + fp
    n_pos = tp + fn
    out.update({
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_demand_rate": round(100.0 * fp / n_zero, 2) if n_zero else np.nan,
        "false_negative_demand_rate": round(100.0 * fn / n_pos, 2) if n_pos else np.nan,
    })
    return out


def _combine(proba: np.ndarray, qty: np.ndarray, threshold: float) -> np.ndarray:
    qty = np.maximum(0.0, np.asarray(qty, dtype=float))
    return np.where(proba < threshold, 0.0, qty)


def run_hurdle_source(df: pd.DataFrame, source: str) -> dict[str, Any]:
    print(f"[Phase 10.1] Hurdle {source}...")
    df_src, numeric, cats = prepare_features(df, source)
    df_src = _usable(df_src)
    parts = _split_frames(df_src, numeric, cats)
    y_train = parts["train_y"]
    zero_share = float(np.mean(y_train == 0)) if len(y_train) else 0.0
    skipped = zero_share < MIN_TRAIN_ZERO_SHARE
    result: dict[str, Any] = {
        "source_dataset": source,
        "skipped": skipped,
        "skip_reason": None if not skipped else (
            f"Train zero-demand share={100*zero_share:.2f}% < {100*MIN_TRAIN_ZERO_SHARE:.0f}%. "
            "Hurdle is for intermittent demand; UCI rows are predominantly positive-demand "
            "invoice days (missing days are absent, not coded as zero)."
        ),
        "train_zero_share_pct": round(100.0 * zero_share, 2),
        "train_n": int(len(y_train)),
        "val_n": int(len(parts["validation_y"])),
        "test_n": int(len(parts["test_y"])),
    }
    if skipped:
        print(f"  SKIP {source}: {result['skip_reason']}")
        return result

    pre = FeaturePreprocessor(numeric, cats, impute=False)
    t0 = time.perf_counter()
    X_train = pre.fit_transform(parts["train_X"])
    ybin = parts["train_ybin"]
    clf = lgb.LGBMClassifier(
        **LGB_POINT_PARAMS,
        is_unbalance=True,
    )
    clf.fit(X_train, ybin, feature_name=pre.feature_names_)
    pos_idx = np.flatnonzero(parts["train_y"] > 0)
    if len(pos_idx) < 100:
        result["skipped"] = True
        result["skip_reason"] = "Too few positive-demand train rows for stage-2 regressor."
        return result
    pre_reg = FeaturePreprocessor(numeric, cats, impute=False)
    y_pos = parts["train_y"][pos_idx]
    X_pos = pre_reg.fit_transform(parts["train_X"].iloc[pos_idx])
    n_pos = int(len(pos_idx))
    n_neg = int(len(y_train) - n_pos)
    reg = lgb.LGBMRegressor(**LGB_POINT_PARAMS)
    reg.fit(X_pos, y_pos, feature_name=pre_reg.feature_names_)
    train_s = time.perf_counter() - t0

    X_val = pre.transform(parts["validation_X"])
    X_val_reg = pre_reg.transform(parts["validation_X"])
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_qty = _predict_nonneg(reg, X_val_reg)
    y_val = parts["validation_y"]

    rows = []
    for th in THRESHOLDS:
        pred = _combine(val_proba, val_qty, th)
        m = forecast_metrics(y_val, pred, f"hurdle_{th:.2f}")
        clf_m = _clf_metrics(parts["validation_ybin"], val_proba, th)
        rows.append({
            "source_dataset": source,
            "split": "validation",
            "threshold": th,
            **{k: m[k] for k in ["MAE", "RMSE", "sMAPE", "WAPE", "bias", "n",
                                 "zero_mae", "nonzero_mae", "zero_positive_prediction_rate"]},
            "precision": clf_m["precision"],
            "recall": clf_m["recall"],
            "f1": clf_m["f1"],
            "false_positive_demand_rate": clf_m["false_positive_demand_rate"],
            "false_negative_demand_rate": clf_m["false_negative_demand_rate"],
        })
    thr_df = pd.DataFrame(rows)
    # Validation objective: WAPE, then MAE, then nonzero_mae (retain positive-demand skill)
    thr_df = thr_df.sort_values(["WAPE", "MAE", "nonzero_mae"]).reset_index(drop=True)
    best_th = float(thr_df.iloc[0]["threshold"])
    print(f"  selected threshold={best_th:.2f} val WAPE={thr_df.iloc[0]['WAPE']:.4f}")

    X_test = pre.transform(parts["test_X"])
    X_test_reg = pre_reg.transform(parts["test_X"])
    test_proba = clf.predict_proba(X_test)[:, 1]
    test_qty = _predict_nonneg(reg, X_test_reg)
    test_pred = _combine(test_proba, test_qty, best_th)
    y_test = parts["test_y"]
    test_m = forecast_metrics(y_test, test_pred, "hurdle")
    test_clf = _clf_metrics(parts["test_ybin"], test_proba, best_th)
    val_clf_best = _clf_metrics(parts["validation_ybin"], val_proba, best_th)
    val_clf_05 = _clf_metrics(parts["validation_ybin"], val_proba, 0.5)

    pred_df = parts["test"][GRAIN].copy()
    pred_df["actual_units_sold"] = y_test
    pred_df["p_demand"] = test_proba
    pred_df["qty_if_positive"] = test_qty
    pred_df["predicted_units_sold"] = test_pred
    pred_df["model"] = "hurdle_lightgbm"
    pred_df["threshold"] = best_th

    cm = confusion_matrix(parts["test_ybin"], (test_proba >= best_th).astype(int), labels=[0, 1])

    result.update({
        "best_threshold": best_th,
        "training_time_sec": round(train_s, 3),
        "n_pos_train": n_pos,
        "n_neg_train": n_neg,
        "threshold_table": thr_df,
        "val_classifier_selected": val_clf_best,
        "val_classifier_0.5": val_clf_05,
        "test_metrics": test_m,
        "test_classifier": test_clf,
        "test_predictions": pred_df,
        "confusion_matrix_test": cm,
        "preprocessor_clf_features": pre.feature_names_,
        "preprocessor_reg_features": pre_reg.feature_names_,
    })
    print(
        f"  TEST WAPE={test_m['WAPE']:.4f} MAE={test_m['MAE']:.4f} "
        f"zero_pos_rate={test_m['zero_positive_prediction_rate']}"
    )
    return result


def load_phase8_test(source: str) -> pd.DataFrame:
    if not os.path.exists(ML_PRED_PATH):
        raise FileNotFoundError(ML_PRED_PATH)
    p = pd.read_parquet(ML_PRED_PATH)
    p["date"] = pd.to_datetime(p["date"])
    return p[p["source_dataset"] == source].copy()


def compare_with_phase8(hurdle: dict) -> dict:
    src = hurdle["source_dataset"]
    if hurdle.get("skipped"):
        return {"source_dataset": src, "skipped": True}
    p8 = load_phase8_test(src)
    h = hurdle["test_predictions"]
    merged = p8.merge(
        h[GRAIN + ["predicted_units_sold", "p_demand"]],
        on=GRAIN, how="inner", suffixes=("_p8", "_hurdle"),
    )
    if len(merged) != len(p8) or len(merged) != len(h):
        print(f"  WARN {src}: Phase 8/hurdle merge n p8={len(p8)} hurdle={len(h)} merged={len(merged)}")
    a = merged["actual_units_sold"].to_numpy()
    m_a = forecast_metrics(a, merged["predicted_units_sold_p8"].to_numpy(), "phase8_lightgbm")
    m_b = forecast_metrics(a, merged["predicted_units_sold_hurdle"].to_numpy(), "hurdle_lightgbm")
    return {
        "source_dataset": src,
        "skipped": False,
        "n_matched": int(len(merged)),
        "phase8": m_a,
        "hurdle": m_b,
        "wape_improvement_pct": round(
            (m_a["WAPE"] - m_b["WAPE"]) / m_a["WAPE"] * 100.0, 4
        ) if m_a["WAPE"] else np.nan,
        "mae_improvement_pct": round(
            (m_a["MAE"] - m_b["MAE"]) / m_a["MAE"] * 100.0, 4
        ) if m_a["MAE"] else np.nan,
        "zero_pos_phase8": m_a["zero_positive_prediction_rate"],
        "zero_pos_hurdle": m_b["zero_positive_prediction_rate"],
    }


def create_hurdle_charts(results: dict[str, dict], comparisons: dict[str, dict]) -> list[str]:
    import matplotlib.pyplot as plt

    apply_mpl_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = []
    syn = results.get("SYNTHETIC")
    if not syn or syn.get("skipped"):
        return paths

    thr = syn["threshold_table"]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(thr["threshold"], thr["WAPE"], marker="o", color="#1d4ed8", lw=2, label="WAPE")
    ax1.plot(thr["threshold"], thr["MAE"], marker="s", color="#d97706", lw=2, label="MAE")
    ax2.plot(
        thr["threshold"], thr["zero_positive_prediction_rate"],
        marker="^", color="#059669", lw=2, label="Zero-day P(pred>0) %",
    )
    ax1.axvline(syn["best_threshold"], color="#111827", ls="--", lw=1.2, label="selected")
    ax1.set_xlabel("Classifier threshold P(demand>0)")
    ax1.set_ylabel("Validation error")
    ax2.set_ylabel("Zero-day positive prediction rate (%)")
    ax2.spines["right"].set_visible(True)
    ax1.set_title("SYNTHETIC hurdle threshold analysis (validation only)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    p = os.path.join(FIGURES_DIR, "zero_demand_threshold_analysis.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    cm = syn["confusion_matrix_test"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred zero", "Pred demand"])
    ax.set_yticks([0, 1], ["Actual zero", "Actual demand"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black")
    ax.set_title(f"SYNTHETIC hurdle confusion (TEST, th={syn['best_threshold']:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    p = os.path.join(FIGURES_DIR, "classifier_confusion_matrix.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    cmp_ = comparisons.get("SYNTHETIC")
    if cmp_ and not cmp_.get("skipped"):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8))
        labels = ["Phase 8 LGBM", "Hurdle"]
        for ax, key, title, ylab in [
            (axes[0], "WAPE", "TEST WAPE", "WAPE (%)"),
            (axes[1], "MAE", "TEST MAE", "MAE (units)"),
            (axes[2], "zero_positive_prediction_rate", "Zero-day P(pred>0)", "%"),
        ]:
            vals = [cmp_["phase8"][key], cmp_["hurdle"][key]]
            ax.bar(labels, vals, color=["#94a3b8", "#1d4ed8"])
            ax.set_title(title)
            ax.set_ylabel(ylab)
        fig.suptitle("SYNTHETIC hurdle vs frozen Phase 8 LightGBM (matched TEST)")
        fig.tight_layout()
        p = os.path.join(FIGURES_DIR, "hurdle_vs_lightgbm.png")
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def run_hurdle_forecasting(df: pd.DataFrame | None = None, save: bool = True) -> dict[str, Any]:
    ensure_dirs()
    if df is None:
        df = load_feature_dataset()
    results, comparisons = {}, {}
    for src in ["UCI", "SYNTHETIC"]:
        results[src] = run_hurdle_source(df, src)
        comparisons[src] = compare_with_phase8(results[src])
    charts = create_hurdle_charts(results, comparisons) if save else []
    if save:
        frames = []
        for src, r in results.items():
            if r.get("skipped") or "threshold_table" not in r:
                continue
            r["threshold_table"].to_parquet(
                os.path.join(PHASE10_DIR, f"hurdle_threshold_{src.lower()}.parquet"), index=False
            )
            r["test_predictions"].to_parquet(
                os.path.join(PHASE10_DIR, f"hurdle_test_predictions_{src.lower()}.parquet"),
                index=False,
            )
            frames.append(r["threshold_table"])
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(
                os.path.join(PHASE10_DIR, "hurdle_threshold_table.parquet"), index=False
            )
        rows = []
        for src, c in comparisons.items():
            if c.get("skipped"):
                rows.append({"source_dataset": src, "skipped": True, "reason": results[src].get("skip_reason")})
                continue
            rows.append({
                "source_dataset": src,
                "skipped": False,
                "best_threshold": results[src]["best_threshold"],
                "n_matched": c["n_matched"],
                "phase8_WAPE": c["phase8"]["WAPE"],
                "hurdle_WAPE": c["hurdle"]["WAPE"],
                "phase8_MAE": c["phase8"]["MAE"],
                "hurdle_MAE": c["hurdle"]["MAE"],
                "phase8_sMAPE": c["phase8"]["sMAPE"],
                "hurdle_sMAPE": c["hurdle"]["sMAPE"],
                "phase8_bias": c["phase8"]["bias"],
                "hurdle_bias": c["hurdle"]["bias"],
                "phase8_zero_mae": c["phase8"]["zero_mae"],
                "hurdle_zero_mae": c["hurdle"]["zero_mae"],
                "phase8_nonzero_mae": c["phase8"]["nonzero_mae"],
                "hurdle_nonzero_mae": c["hurdle"]["nonzero_mae"],
                "phase8_zero_pos_rate": c["zero_pos_phase8"],
                "hurdle_zero_pos_rate": c["zero_pos_hurdle"],
                "wape_improvement_pct": c["wape_improvement_pct"],
            })
        pd.DataFrame(rows).to_parquet(
            os.path.join(PHASE10_DIR, "hurdle_vs_phase8.parquet"), index=False
        )
    return {"results": results, "comparisons": comparisons, "charts": charts}


if __name__ == "__main__":
    run_hurdle_forecasting(save=True)
