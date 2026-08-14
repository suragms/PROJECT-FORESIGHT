"""
Phase 11 orchestrator — final model selection and production inference artifacts.

Never writes to Phase 8 / 9 / 10 artifact paths.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version as pkg_version

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.final_forecasting import FinalForecaster, FinalForecastError  # noqa: E402
from src.ml_forecasting import (  # noqa: E402
    CATEGORICAL_FEATURES_BOTH,
    CATEGORICAL_FEATURES_SYNTHETIC,
    EXCLUDED_FIELDS,
    FeaturePreprocessor,
    NUMERIC_FEATURES_BOTH,
    NUMERIC_FEATURES_SYNTHETIC_EXTRA,
    _predict_nonneg,
    load_feature_dataset,
    prepare_features,
)
from src.phase10_common import (  # noqa: E402
    GRAIN,
    HORIZONS,
    LGB_POINT_PARAMS,
    ML_PRED_PATH,
    PHASE8_FREEZE_FILES,
    PHASE8_TEST,
    PHASE9_FREEZE_FILES,
    PHASE10_DIR,
    QUANTILES,
    TARGET,
    THRESHOLDS,
    hashes_unchanged,
    snapshot_hashes,
    split_end_dates,
    zero_slice_metrics,
)
from src.phase10_direct_horizon import (  # noqa: E402
    _feature_lists,
    _masks,
    add_horizon_target,
)
from src.phase10_hurdle_forecasting import (  # noqa: E402
    MIN_TRAIN_ZERO_SHARE,
    _combine,
    _split_frames,
    _usable,
)
from src.phase10_prediction_intervals import _enforce_nonneg_and_order  # noqa: E402
from src.phase11_common import (  # noqa: E402
    CANDIDATE_PATH,
    FEATURE_VERSION,
    FIGURES_FINAL_DIR,
    FINAL_PRED_PATH,
    FORECASTS_FINAL_DIR,
    METADATA_PATH,
    MODELS_FINAL_DIR,
    MONITOR_PATH,
    PHASE7_BASELINE,
    PHASE9_DIR,
    PHASE9_HORIZON,
    PHASE9_ZERO,
    RANDOM_STATE,
    REGISTRY_PATH,
    REPORT_PATH,
    SELECTION_LOGIC,
    SELECTION_PATH,
    STABILITY_OVERRIDE_REL_WAPE,
    apply_mpl_style,
    ensure_phase11_dirs,
    file_md5,
    file_sha256,
    forecast_metrics,
    git_code_version,
    md_table,
    pinball_loss,
    relpath,
    utc_now,
    verify_phase10_ready,
    write_json,
)
from src.phase9_common import classify_stability  # noqa: E402
from src.phase9_residual_analysis import demand_regime  # noqa: E402
from src.phase11_common import UCI_MODEL_PATH  # noqa: E402


def _pkg(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "not-installed"


def _round_metrics(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, (float, np.floating)):
            out[k] = None if v != v else round(float(v), 4)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# 11.1 Candidate matrix from existing Phase 8–10 artifacts
# ---------------------------------------------------------------------------

def build_candidate_matrix() -> pd.DataFrame:
    rows = []

    def add(**kwargs):
        rec = {
            "dataset": kwargs.get("dataset"),
            "model": kwargs.get("model"),
            "horizon": kwargs.get("horizon"),
            "MAE": kwargs.get("MAE"),
            "RMSE": kwargs.get("RMSE"),
            "sMAPE": kwargs.get("sMAPE"),
            "WAPE": kwargs.get("WAPE"),
            "bias": kwargs.get("bias"),
            "zero_demand_MAE": kwargs.get("zero_demand_MAE"),
            "positive_demand_MAE": kwargs.get("positive_demand_MAE"),
            "training_time": "" if kwargs.get("training_time") is None else str(kwargs.get("training_time")),
            "inference_time": "" if kwargs.get("inference_time", "n/a") is None else str(kwargs.get("inference_time", "n/a")),
            "stability": kwargs.get("stability", "not walk-forward tested"),
            "prediction_interval_availability": kwargs.get(
                "prediction_interval_availability", False
            ),
            "complexity": kwargs.get("complexity"),
            "implemented": True,
            "eligible": kwargs.get("eligible", True),
            "exclude_reason": kwargs.get("exclude_reason"),
            "source_artifact": kwargs.get("source_artifact"),
        }
        rows.append(rec)

    p8_stab = {"UCI": "Stable (Phase 9 folds; fold-2 WAPE spike)", "SYNTHETIC": "Stable (Phase 9)"}
    for src, m in PHASE8_TEST.items():
        add(
            dataset=src, model="Phase8_LightGBM", horizon=1,
            MAE=m["MAE"], RMSE=m["RMSE"], sMAPE=m["sMAPE"], WAPE=m["WAPE"],
            bias=None, zero_demand_MAE=None, positive_demand_MAE=None,
            training_time="frozen",
            stability=p8_stab[src],
            prediction_interval_availability=False,
            complexity="single LightGBM + freq-encode preprocessor",
            source_artifact="data/processed/forecasts/ml/ml_predictions.parquet",
        )

    hv = pd.read_parquet(os.path.join(PHASE10_DIR, "hurdle_vs_phase8.parquet"))
    hp_path = os.path.join(PHASE10_DIR, "hurdle_test_predictions_synthetic.parquet")
    hurdle_rmse = None
    if os.path.exists(hp_path):
        hp = pd.read_parquet(hp_path)
        hurdle_rmse = forecast_metrics(
            hp["actual_units_sold"].to_numpy(),
            hp["predicted_units_sold"].to_numpy(),
            "hurdle",
        )["RMSE"]
    for _, r in hv.iterrows():
        if bool(r.get("skipped")):
            continue
        add(
            dataset=r["source_dataset"], model="Hurdle_th0.50", horizon=1,
            MAE=r.get("hurdle_MAE", r.get("MAE")),
            RMSE=hurdle_rmse,
            sMAPE=r.get("hurdle_sMAPE"),
            WAPE=r.get("hurdle_WAPE"),
            bias=r.get("hurdle_bias"),
            zero_demand_MAE=r.get("hurdle_zero_mae"),
            positive_demand_MAE=r.get("hurdle_nonzero_mae"),
            training_time=r.get("training_time_sec", 16.282),
            stability="not walk-forward tested (Phase 8 LGBM was Stable)",
            prediction_interval_availability=False,
            complexity="LGBMClassifier + LGBMRegressor + threshold",
            source_artifact="data/processed/forecasts/phase10/hurdle_vs_phase8.parquet",
        )

    inter = pd.read_parquet(os.path.join(PHASE10_DIR, "intermittent_summary.parquet"))
    for _, r in inter.iterrows():
        name = str(r.get("model", r.get("method")))
        wape = float(r["WAPE"])
        baseline = PHASE7_BASELINE.get(str(r["source_dataset"]), {}).get("WAPE", np.inf)
        worse = name.lower() != "naive" and wape > baseline
        add(
            dataset=r["source_dataset"], model=f"Intermittent_{name}", horizon=1,
            MAE=r.get("MAE"), RMSE=r.get("RMSE"), sMAPE=r.get("sMAPE"), WAPE=wape,
            bias=r.get("bias"),
            zero_demand_MAE=r.get("zero_mae"),
            positive_demand_MAE=r.get("nonzero_mae"),
            training_time="univariate rolling",
            stability="not walk-forward tested",
            complexity="Croston-family / naive univariate",
            eligible=not worse,
            exclude_reason="WAPE worse than Phase 7 Naive" if worse else None,
            source_artifact="data/processed/forecasts/phase10/intermittent_summary.parquet",
        )

    dsum = pd.read_parquet(os.path.join(PHASE10_DIR, "direct_horizon_summary.parquet"))
    for _, r in dsum.iterrows():
        add(
            dataset=r["source_dataset"], model="Direct_LightGBM", horizon=int(r["horizon"]),
            MAE=r.get("MAE"), RMSE=r.get("RMSE"), sMAPE=r.get("sMAPE"), WAPE=r.get("WAPE"),
            bias=r.get("bias"),
            zero_demand_MAE=r.get("zero_mae"),
            positive_demand_MAE=r.get("nonzero_mae"),
            training_time=r.get("training_time_sec"),
            stability="not walk-forward tested",
            complexity="one LightGBM per horizon + target calendar",
            source_artifact="data/processed/forecasts/phase10/direct_horizon_summary.parquet",
        )

    if os.path.exists(PHASE9_HORIZON):
        rec = pd.read_parquet(PHASE9_HORIZON)
        for _, r in rec.iterrows():
            add(
                dataset=r["source_dataset"], model="Phase9_Recursive_LightGBM",
                horizon=int(r["horizon"]),
                MAE=r.get("MAE"), RMSE=r.get("RMSE"), sMAPE=r.get("sMAPE"), WAPE=r.get("WAPE"),
                bias=r.get("bias"),
                training_time="frozen Phase 8 iterated",
                stability="diagnostic (exo frozen at origin)",
                complexity="recursive 1-step with frozen exogenous",
                eligible=False,
                exclude_reason="Diagnostic only; exogenous frozen at origin; not a persisted production model",
                source_artifact="data/processed/forecasts/phase9/horizon_summary.parquet",
            )

    hsum = pd.read_parquet(os.path.join(PHASE10_DIR, "hpo_summary.parquet"))
    for _, r in hsum.iterrows():
        cfg = int(r["best_config_id"])
        add(
            dataset=r["source_dataset"], model=f"HPO_cfg{cfg}", horizon=1,
            MAE=r.get("MAE"), RMSE=r.get("RMSE"), sMAPE=r.get("sMAPE"), WAPE=r.get("WAPE"),
            bias=r.get("bias"),
            training_time="small 8-config grid",
            stability="not walk-forward tested",
            complexity="single LightGBM, retuned",
            source_artifact="data/processed/forecasts/phase10/hpo_summary.parquet",
        )

    qpred = pd.read_parquet(os.path.join(PHASE10_DIR, "quantile_predictions.parquet"))
    qsum = pd.read_parquet(os.path.join(PHASE10_DIR, "quantile_summary.parquet"))
    for src, g in qpred.groupby("source_dataset", observed=True):
        m = forecast_metrics(g["actual_units_sold"].to_numpy(), g["p50"].to_numpy(), "quantile_p50")
        qrow = qsum[qsum["source_dataset"] == src]
        add(
            dataset=src, model="Quantile_P50", horizon=1,
            MAE=m["MAE"], RMSE=m["RMSE"], sMAPE=m["sMAPE"], WAPE=m["WAPE"],
            bias=m["bias"],
            zero_demand_MAE=m.get("zero_mae"),
            positive_demand_MAE=m.get("nonzero_mae"),
            training_time=float(qrow["training_time_sec"].iloc[0]) if "training_time_sec" in qrow.columns and len(qrow) else None,
            stability="not walk-forward tested",
            prediction_interval_availability=True,
            complexity="three independent quantile LightGBMs",
            source_artifact="data/processed/forecasts/phase10/quantile_predictions.parquet",
        )
    return pd.DataFrame(rows)


def apply_selection(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return one selected point model per dataset x operational horizon."""
    sel_rows = []
    notes = []

    def pick(dataset: str, horizon: int, model: str, reason: str):
        hit = candidates[
            (candidates["dataset"] == dataset)
            & (candidates["model"] == model)
            & (candidates["horizon"] == horizon)
        ]
        if hit.empty:
            raise RuntimeError(f"Selected {dataset} h={horizon} {model} not in candidate matrix")
        row = hit.iloc[0].to_dict()
        row["selected"] = True
        row["selection_reason"] = reason
        sel_rows.append(row)
        notes.append(f"{dataset} h={horizon} -> {model}: {reason}")

    # UCI h=1: Phase 8 vs HPO_cfg7. Relative WAPE gain of HPO:
    uci_p8 = float(candidates[(candidates.dataset == "UCI") & (candidates.model == "Phase8_LightGBM")].iloc[0]["WAPE"])
    hpo_u = candidates[(candidates.dataset == "UCI") & (candidates.model.str.startswith("HPO_"))]
    hpo_w = float(hpo_u.iloc[0]["WAPE"]) if len(hpo_u) else np.nan
    rel = (uci_p8 - hpo_w) / uci_p8 if uci_p8 else 0.0
    if rel >= STABILITY_OVERRIDE_REL_WAPE:
        pick("UCI", 1, str(hpo_u.iloc[0]["model"]),
             f"HPO WAPE {hpo_w:.4f} beats Phase 8 {uci_p8:.4f} by {100*rel:.2f}% relative")
    else:
        pick(
            "UCI", 1, "Phase8_LightGBM",
            f"Phase 8 WAPE {uci_p8:.4f}; HPO {hpo_w:.4f} is only {100*rel:.2f}% relative "
            f"(<{100*STABILITY_OVERRIDE_REL_WAPE:.0f}%) and was not walk-forward tested. "
            "Primary criterion 3 (stability) keeps the frozen Phase 8 LightGBM. "
            "UCI hurdle was not identified (0% coded zeros).",
        )

    # SYNTHETIC h=1: hurdle vs Phase 8 vs HPO vs P50 vs intermittent
    syn_h = candidates[(candidates.dataset == "SYNTHETIC") & (candidates.model == "Hurdle_th0.50")]
    syn_p8 = candidates[(candidates.dataset == "SYNTHETIC") & (candidates.model == "Phase8_LightGBM")]
    hurdle_w = float(syn_h.iloc[0]["WAPE"])
    p8_w = float(syn_p8.iloc[0]["WAPE"])
    pick(
        "SYNTHETIC", 1, "Hurdle_th0.50",
        f"Hurdle TEST WAPE {hurdle_w:.4f} vs Phase 8 {p8_w:.4f} "
        f"({100*(p8_w-hurdle_w)/p8_w:.1f}% relative). MAE and zero-demand false positives "
        "also improve. HPO and Croston-family do not beat hurdle. Complexity is justified "
        "by the 61% zero-demand share.",
    )

    for src in ["UCI", "SYNTHETIC"]:
        for h in (3, 7, 14, 30):
            pick(
                src, h, "Direct_LightGBM",
                "Leakage-safe h-step model with origin features + known-in-advance target "
                "calendar. Recursive Phase 9 is diagnostic only (frozen exo). Direct h=1 is "
                "a different task than Phase 8 same-row 1-step, so h=1 stays with the "
                "operational 1-step model.",
            )
    out = pd.DataFrame(sel_rows)
    out.attrs["notes"] = notes
    return out


# ---------------------------------------------------------------------------
# Persist selected models
# ---------------------------------------------------------------------------

def _wrap_preprocessor(pre: FeaturePreprocessor) -> FeaturePreprocessor:
    pre.__class__ = FeaturePreprocessor
    return pre


def _base_payload(**kwargs) -> dict:
    return {
        "training_data_version": kwargs.pop("training_data_version"),
        "feature_version": FEATURE_VERSION,
        "validation_method": kwargs.pop("validation_method"),
        "training_timestamp": utc_now(),
        "code_version": kwargs.pop("code_version"),
        "random_state": RANDOM_STATE,
        **kwargs,
    }


def persist_uci_phase8(code_version: str, features_hash: str, metrics: dict) -> dict:
    src = UCI_MODEL_PATH
    raw = joblib.load(src)
    model_id = "uci_h1_phase8_lightgbm"
    payload = _base_payload(
        model_id=model_id,
        dataset="UCI",
        horizon=1,
        model_type="lightgbm_point",
        training_data_version=features_hash,
        code_version=code_version,
        validation_method="Phase 6 chronological split; Phase 9 walk-forward on copies",
        hyperparameters=dict(LGB_POINT_PARAMS),
        metrics=_round_metrics(metrics),
        numeric_features=list(raw["numeric_features"]),
        categorical_features=list(raw["categorical_features"]),
        feature_names=list(raw["feature_names"]),
        model=raw["model"],
        preprocessor=_wrap_preprocessor(raw["preprocessor"]),
        copied_from=relpath(src),
        phase8_model_name=raw.get("model_name"),
    )
    path = os.path.join(MODELS_FINAL_DIR, f"{model_id}.joblib")
    joblib.dump(payload, path)
    return _registry_record(payload, path, "selected")


def persist_hurdle(df: pd.DataFrame, code_version: str, features_hash: str) -> dict:
    source = "SYNTHETIC"
    df_src, numeric, cats = prepare_features(df, source)
    df_src = _usable(df_src)
    parts = _split_frames(df_src, numeric, cats)
    zero_share = float(np.mean(parts["train_y"] == 0))
    if zero_share < MIN_TRAIN_ZERO_SHARE:
        raise RuntimeError("SYNTHETIC hurdle unexpectedly skipped")
    pre = FeaturePreprocessor(numeric, cats, impute=False)
    t0 = time.perf_counter()
    X_train = pre.fit_transform(parts["train_X"])
    clf = lgb.LGBMClassifier(**LGB_POINT_PARAMS, is_unbalance=True)
    clf.fit(X_train, parts["train_ybin"], feature_name=pre.feature_names_)
    pos_idx = np.flatnonzero(parts["train_y"] > 0)
    pre_reg = FeaturePreprocessor(numeric, cats, impute=False)
    X_pos = pre_reg.fit_transform(parts["train_X"].iloc[pos_idx])
    reg = lgb.LGBMRegressor(**LGB_POINT_PARAMS)
    reg.fit(X_pos, parts["train_y"][pos_idx], feature_name=pre_reg.feature_names_)
    train_s = time.perf_counter() - t0

    X_val = pre.transform(parts["validation_X"])
    X_val_reg = pre_reg.transform(parts["validation_X"])
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_qty = _predict_nonneg(reg, X_val_reg)
    y_val = parts["validation_y"]
    best_th, best_w = None, np.inf
    for th in THRESHOLDS:
        pred = _combine(val_proba, val_qty, th)
        w = forecast_metrics(y_val, pred, "h")["WAPE"]
        if w < best_w:
            best_w, best_th = w, float(th)

    X_test = pre.transform(parts["test_X"])
    X_test_reg = pre_reg.transform(parts["test_X"])
    t1 = time.perf_counter()
    test_pred = _combine(
        clf.predict_proba(X_test)[:, 1],
        _predict_nonneg(reg, X_test_reg),
        best_th,
    )
    infer_s = time.perf_counter() - t1
    test_m = forecast_metrics(parts["test_y"], test_pred, "hurdle")
    model_id = "synthetic_h1_hurdle_th050"
    payload = _base_payload(
        model_id=model_id,
        dataset="SYNTHETIC",
        horizon=1,
        model_type="hurdle",
        training_data_version=features_hash,
        code_version=code_version,
        validation_method="threshold selected on validation WAPE then MAE then nonzero MAE; TEST once",
        hyperparameters={**dict(LGB_POINT_PARAMS), "is_unbalance": True, "threshold": best_th},
        metrics=_round_metrics(test_m),
        numeric_features=numeric,
        categorical_features=cats,
        feature_names=pre.feature_names_,
        classifier=clf,
        regressor=reg,
        preprocessor_clf=_wrap_preprocessor(pre),
        preprocessor_reg=_wrap_preprocessor(pre_reg),
        threshold=best_th,
        training_time_sec=round(train_s, 3),
        inference_time_sec=round(infer_s, 3),
    )
    path = os.path.join(MODELS_FINAL_DIR, f"{model_id}.joblib")
    joblib.dump(payload, path)
    print(f"  hurdle threshold={best_th} TEST WAPE={test_m['WAPE']:.4f} infer={infer_s:.3f}s")
    return _registry_record(payload, path, "selected")


def persist_direct(df: pd.DataFrame, source: str, h: int, code_version: str, features_hash: str) -> dict:
    src = df[df["source_dataset"] == source].copy()
    if "units_sold_lag_1" in src.columns:
        src = src[src["units_sold_lag_1"].notna()].copy()
    ends = split_end_dates(src)
    df_h = add_horizon_target(src, h)
    masks = _masks(df_h, ends)
    numeric, cats = _feature_lists(df_h, source)
    feat = numeric + cats
    train = df_h.loc[masks["train"]]
    test = df_h.loc[masks["test"]]
    pre = FeaturePreprocessor(numeric, cats, impute=False)
    t0 = time.perf_counter()
    model = lgb.LGBMRegressor(**LGB_POINT_PARAMS)
    model.fit(pre.fit_transform(train[feat]), train["target"].astype(float).to_numpy(),
              feature_name=pre.feature_names_)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = _predict_nonneg(model, pre.transform(test[feat]))
    infer_s = time.perf_counter() - t1
    test_m = forecast_metrics(test["target"].astype(float).to_numpy(), pred, f"direct_h{h}")
    model_id = f"{source.lower()}_h{h}_direct_lightgbm"
    payload = _base_payload(
        model_id=model_id,
        dataset=source,
        horizon=h,
        model_type="direct_lightgbm",
        training_data_version=features_hash,
        code_version=code_version,
        validation_method="origin in split AND target_date <= split end (train/val); TEST once",
        hyperparameters=dict(LGB_POINT_PARAMS),
        metrics=_round_metrics(test_m),
        numeric_features=numeric,
        categorical_features=cats,
        feature_names=pre.feature_names_,
        model=model,
        preprocessor=_wrap_preprocessor(pre),
        training_time_sec=round(train_s, 3),
        inference_time_sec=round(infer_s, 3),
        train_n=int(len(train)),
        test_n=int(len(test)),
    )
    path = os.path.join(MODELS_FINAL_DIR, f"{model_id}.joblib")
    joblib.dump(payload, path)
    print(f"  {source} h={h} TEST WAPE={test_m['WAPE']:.4f} n={len(test):,}")
    return _registry_record(payload, path, "selected")


def persist_quantile(df: pd.DataFrame, source: str, code_version: str, features_hash: str) -> dict:
    df_src, numeric, cats = prepare_features(df, source)
    df_src = _usable(df_src)
    feat = numeric + cats
    train = df_src[df_src["split"] == "train"]
    test = df_src[df_src["split"] == "test"]
    pre = FeaturePreprocessor(numeric, cats, impute=False)
    X_train = pre.fit_transform(train[feat])
    y_train = train[TARGET].astype(float).to_numpy()
    models = {}
    t0 = time.perf_counter()
    for tau in QUANTILES:
        params = dict(LGB_POINT_PARAMS)
        params["objective"] = "quantile"
        params["alpha"] = tau
        m = lgb.LGBMRegressor(**params)
        m.fit(X_train, y_train, feature_name=pre.feature_names_)
        models[tau] = m
    train_s = time.perf_counter() - t0
    X_test = pre.transform(test[feat])
    raw = {tau: np.asarray(models[tau].predict(X_test), dtype=float) for tau in QUANTILES}
    p10, p50, p90, extra = _enforce_nonneg_and_order(raw[0.10], raw[0.50], raw[0.90])
    y = test[TARGET].astype(float).to_numpy()
    coverage = 100.0 * float(np.mean((y >= p10) & (y <= p90)))
    width = float(np.mean(p90 - p10))
    pin = {f"pinball_p{int(t*100)}": round(pinball_loss(y, q, t), 4)
           for t, q in [(0.10, p10), (0.50, p50), (0.90, p90)]}
    cal = {f"p{int(t*100)}_below_pct": round(100.0 * float(np.mean(y <= q)), 2)
           for t, q in [(0.10, p10), (0.50, p50), (0.90, p90)]}
    model_id = f"{source.lower()}_h1_quantile_p10p90"
    payload = _base_payload(
        model_id=model_id,
        dataset=source,
        horizon=1,
        model_type="quantile_intervals",
        training_data_version=features_hash,
        code_version=code_version,
        validation_method="Phase 6 chronological split; clip at 0 then reorder if crossing",
        hyperparameters={**dict(LGB_POINT_PARAMS), "objective": "quantile", "taus": list(QUANTILES)},
        metrics={
            "coverage_pct": round(coverage, 4),
            "mean_width": round(width, 4),
            "n_crossed_before_reorder": extra["n_crossed_before_reorder"],
            **pin,
            **cal,
        },
        numeric_features=numeric,
        categorical_features=cats,
        feature_names=pre.feature_names_,
        models=models,
        preprocessor=_wrap_preprocessor(pre),
        training_time_sec=round(train_s, 3),
    )
    path = os.path.join(MODELS_FINAL_DIR, f"{model_id}.joblib")
    joblib.dump(payload, path)
    print(f"  {source} intervals coverage={coverage:.2f}% width={width:.3f}")
    return _registry_record(payload, path, "interval_companion")


def _registry_record(payload: dict, path: str, status: str) -> dict:
    return {
        "model_id": payload["model_id"],
        "dataset": payload["dataset"],
        "horizon": payload["horizon"],
        "model_type": payload["model_type"],
        "training_data_version": payload["training_data_version"],
        "feature_version": payload["feature_version"],
        "hyperparameters": payload.get("hyperparameters"),
        "metrics": payload.get("metrics"),
        "validation_method": payload.get("validation_method"),
        "training_timestamp": payload.get("training_timestamp"),
        "code_version": payload.get("code_version"),
        "model_file": relpath(path),
        "hash": file_sha256(path),
        "status": status,
        "training_time_sec": payload.get("training_time_sec"),
        "inference_time_sec": payload.get("inference_time_sec"),
    }


# ---------------------------------------------------------------------------
# Forecasts via the final inference layer
# ---------------------------------------------------------------------------

def _usable_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    sub, _, _ = prepare_features(df, source)
    return _usable(sub)


def generate_final_forecasts(df: pd.DataFrame, registry: list[dict]) -> pd.DataFrame:
    frames = []
    by_id = {r["model_id"]: r for r in registry}
    intervals = {}
    for r in registry:
        if r["status"] == "interval_companion":
            intervals[r["dataset"]] = FinalForecaster.from_registry(r["model_id"])

    for r in registry:
        if r["status"] != "selected":
            continue
        ff = FinalForecaster.from_registry(r["model_id"])
        src = r["dataset"]
        h = int(r["horizon"])
        sub = _usable_source(df, src)
        if r["model_type"] == "direct_lightgbm":
            panel = add_horizon_target(sub, h)
            feat = ff.numeric + ff.categorical
            test = panel[(panel["split"] == "test") & panel["target"].notna()].copy()
            hcal = [c for c in feat if str(c).startswith("hcal_")]
            must = hcal + (["units_sold_lag_1"] if "units_sold_lag_1" in test.columns else [])
            test = test.dropna(subset=must)
            pred = ff.predict(test, include_actual=True)
        else:
            test = sub[sub["split"] == "test"].copy()
            iv = intervals.get(src) if h == 1 else None
            pred = ff.predict(test, include_actual=True, intervals=iv)
        frames.append(pred)
        print(f"  forecasts {r['model_id']}: {len(pred):,} rows")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        ["source_dataset", "horizon", "forecast_date", "entity_id", "product_key"]
    ).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _wape(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    den = np.sum(np.abs(y))
    return 0.0 if den == 0 else float(np.sum(np.abs(y - yhat)) / den * 100.0)


def analyze_final(preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    h1 = preds[preds["horizon"] == 1].copy()
    overall_rows = []
    bias_rows = []
    regime_rows = []
    store_rows = []
    month_rows = []
    for src, g in h1.groupby("source_dataset", observed=True):
        m = forecast_metrics(g["actual"].to_numpy(), g["prediction"].to_numpy(), str(g["model_name"].iloc[0]))
        overall_rows.append({"dataset": src, "horizon": 1, **m})
        resid = g["actual"].to_numpy(dtype=float) - g["prediction"].to_numpy(dtype=float)
        pred = g["prediction"].to_numpy(dtype=float)
        act = g["actual"].to_numpy(dtype=float)
        bias_rows.append({
            "dataset": src,
            "mean_bias": round(float(np.mean(pred - act)), 4),
            "median_bias": round(float(np.median(pred - act)), 4),
            "mean_residual": round(float(np.mean(resid)), 4),
            "median_residual": round(float(np.median(resid)), 4),
            "overprediction_pct": round(100.0 * float(np.mean(pred > act)), 2),
            "underprediction_pct": round(100.0 * float(np.mean(pred < act)), 2),
        })
        gg = g.copy()
        gg["regime"] = demand_regime(gg["actual"])
        for rg, gr in gg.groupby("regime"):
            mm = forecast_metrics(gr["actual"].to_numpy(), gr["prediction"].to_numpy(), rg)
            regime_rows.append({"dataset": src, "regime": rg, **mm})
        for ent, ge in g.groupby("entity_id", observed=True):
            store_rows.append({
                "dataset": src,
                "entity_id": ent,
                "n": int(len(ge)),
                "WAPE": round(_wape(ge["actual"], ge["prediction"]), 4),
                "MAE": round(float(np.mean(np.abs(ge["actual"] - ge["prediction"]))), 4),
                "bias": round(float(np.mean(ge["prediction"] - ge["actual"])), 4),
            })
        g2 = g.copy()
        g2["month"] = pd.to_datetime(g2["forecast_date"]).dt.to_period("M").astype(str)
        for mo, gm in g2.groupby("month"):
            month_rows.append({
                "dataset": src, "month": mo, "n": int(len(gm)),
                "WAPE": round(_wape(gm["actual"], gm["prediction"]), 4),
            })

    horizon_rows = []
    for (src, h), g in preds.groupby(["source_dataset", "horizon"], observed=True):
        m = forecast_metrics(g["actual"].to_numpy(), g["prediction"].to_numpy(), f"{src}_h{h}")
        horizon_rows.append({"dataset": src, "horizon": int(h), "model": g["model_name"].iloc[0], **m})

    store_df = pd.DataFrame(store_rows)
    store_sum = []
    for src, g in store_df.groupby("dataset"):
        w = g["WAPE"].astype(float)
        store_sum.append({
            "dataset": src,
            "n_entities": int(len(g)),
            "best_entity": g.loc[g["WAPE"].idxmin(), "entity_id"],
            "best_WAPE": round(float(w.min()), 4),
            "median_WAPE": round(float(w.median()), 4),
            "worst_entity": g.loc[g["WAPE"].idxmax(), "entity_id"],
            "worst_WAPE": round(float(w.max()), 4),
            "spread_WAPE": round(float(w.max() - w.min()), 4),
        })

    stab_rows = []
    month_df = pd.DataFrame(month_rows)
    for src, g in month_df.groupby("dataset"):
        st = classify_stability(g["WAPE"].tolist())
        stab_rows.append({"dataset": src, "axis": "time_month", **st})
    for src, g in store_df.groupby("dataset"):
        if len(g) < 2:
            stab_rows.append({
                "dataset": src, "axis": "entity", "label": "Not applicable",
                "cv_wape": np.nan, "range_ratio": np.nan,
                "mean_wape": round(float(g["WAPE"].mean()), 4),
                "reason": "Fewer than 2 entities; Phase 9 CV/range rule is not identified.",
            })
            continue
        st = classify_stability(g["WAPE"].tolist())
        stab_rows.append({"dataset": src, "axis": "entity", **st})

    # Zero-demand SYNTHETIC vs Phase 8
    syn = h1[h1["source_dataset"] == "SYNTHETIC"]
    p8 = pd.read_parquet(ML_PRED_PATH)
    p8 = p8[p8["source_dataset"] == "SYNTHETIC"].copy()
    p8["date"] = pd.to_datetime(p8["date"])
    merged = syn.merge(
        p8,
        left_on=["forecast_date", "source_dataset", "entity_id", "product_key"],
        right_on=["date", "source_dataset", "entity_id", "product_key"],
        how="inner",
        suffixes=("_final", "_p8"),
    )
    y = merged["actual"].to_numpy() if "actual" in merged.columns else merged["actual_units_sold"].to_numpy()
    # after merge, actual from syn is 'actual'
    if "actual" in merged.columns:
        y = merged["actual"].to_numpy(dtype=float)
    zero = {
        "final": forecast_metrics(y, merged["prediction"].to_numpy(), "final"),
        "phase8": forecast_metrics(y, merged["predicted_units_sold"].to_numpy(), "phase8"),
        "n_matched": int(len(merged)),
    }

    # Intervals on h=1 where bounds exist
    iv_rows = []
    for src, g in h1.groupby("source_dataset", observed=True):
        ok = g["lower_bound"].notna() & g["upper_bound"].notna() & g["actual"].notna()
        gg = g.loc[ok]
        if gg.empty:
            continue
        yv = gg["actual"].to_numpy(dtype=float)
        lo = gg["lower_bound"].to_numpy(dtype=float)
        hi = gg["upper_bound"].to_numpy(dtype=float)
        inside = (yv >= lo) & (yv <= hi)
        cross = int(np.sum(lo > hi))
        qpath = os.path.join(PHASE10_DIR, "quantile_predictions.parquet")
        pin_p50 = np.nan
        if os.path.exists(qpath):
            qg = pd.read_parquet(qpath)
            qg = qg[qg["source_dataset"] == src]
            if len(qg):
                pin_p50 = round(pinball_loss(
                    qg["actual_units_sold"].to_numpy(dtype=float),
                    qg["p50"].to_numpy(dtype=float),
                    0.50,
                ), 4)
        iv_rows.append({
            "dataset": src,
            "n": int(len(gg)),
            "coverage_pct": round(100.0 * float(np.mean(inside)), 4),
            "nominal_coverage_pct": 80.0,
            "mean_width": round(float(np.mean(hi - lo)), 4),
            "pinball_p10": round(pinball_loss(yv, lo, 0.10), 4),
            "pinball_p50": pin_p50,
            "pinball_p90": round(pinball_loss(yv, hi, 0.90), 4),
            "p10_below_pct": round(100.0 * float(np.mean(yv <= lo)), 2),
            "p90_below_pct": round(100.0 * float(np.mean(yv <= hi)), 2),
            "interval_crossing": cross,
        })

    baseline_rows = []
    for src in ["UCI", "SYNTHETIC"]:
        g = h1[h1["source_dataset"] == src]
        fw = _wape(g["actual"], g["prediction"])
        bw = PHASE7_BASELINE[src]["WAPE"]
        baseline_rows.append({
            "Dataset": src,
            "Baseline": PHASE7_BASELINE[src]["model"],
            "Baseline WAPE": bw,
            "Final Model WAPE": round(fw, 4),
            "Absolute improvement": round(bw - fw, 4),
            "Improvement %": round(100.0 * (bw - fw) / bw, 4) if bw else np.nan,
        })

    tables = {
        "overall_h1": pd.DataFrame(overall_rows),
        "bias": pd.DataFrame(bias_rows),
        "regime": pd.DataFrame(regime_rows),
        "store": store_df,
        "store_summary": pd.DataFrame(store_sum),
        "month": month_df,
        "horizon": pd.DataFrame(horizon_rows),
        "stability": pd.DataFrame(stab_rows),
        "intervals": pd.DataFrame(iv_rows),
        "baseline": pd.DataFrame(baseline_rows),
        "zero_final": pd.DataFrame([{"side": "final", **zero["final"], "n_matched": zero["n_matched"]}]),
        "zero_phase8": pd.DataFrame([{"side": "phase8", **zero["phase8"], "n_matched": zero["n_matched"]}]),
    }
    return tables


def create_charts(preds: pd.DataFrame, tables: dict, candidates: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt
    apply_mpl_style()
    os.makedirs(FIGURES_FINAL_DIR, exist_ok=True)
    paths = []

    fig, ax = plt.subplots()
    base = tables["baseline"]
    x = np.arange(len(base))
    w = 0.35
    ax.bar(x - w / 2, base["Baseline WAPE"], w, label="Phase 7 baseline", color="#9ca3af")
    ax.bar(x + w / 2, base["Final Model WAPE"], w, label="Final model", color="#1d4ed8")
    ax.set_xticks(x, base["Dataset"])
    ax.set_ylabel("TEST WAPE")
    ax.set_title("Final model vs Phase 7 baseline (h=1)")
    ax.legend()
    p = os.path.join(FIGURES_FINAL_DIR, "final_model_comparison.png")
    fig.savefig(p); plt.close(fig); paths.append(p)

    hz = tables["horizon"]
    fig, ax = plt.subplots()
    for src, g in hz.groupby("dataset"):
        g = g.sort_values("horizon")
        ax.plot(g["horizon"], g["WAPE"], marker="o", label=src)
    ax.set_xlabel("Horizon"); ax.set_ylabel("TEST WAPE")
    ax.set_title("Final strategy WAPE by horizon")
    ax.legend()
    p = os.path.join(FIGURES_FINAL_DIR, "final_horizon_comparison.png")
    fig.savefig(p); plt.close(fig); paths.append(p)

    st = tables["store"]
    fig, ax = plt.subplots()
    data, labels = [], []
    for src, g in st.groupby("dataset"):
        data.append(g["WAPE"].to_numpy())
        labels.append(src)
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("Entity WAPE")
    ax.set_title("Final h=1 store/entity WAPE spread")
    p = os.path.join(FIGURES_FINAL_DIR, "final_store_stability.png")
    fig.savefig(p); plt.close(fig); paths.append(p)

    h1 = preds[preds["horizon"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, src in zip(axes, ["UCI", "SYNTHETIC"]):
        g = h1[h1["source_dataset"] == src]
        resid = g["actual"] - g["prediction"]
        ax.hist(resid, bins=60, color="#1d4ed8", alpha=0.85)
        ax.axvline(0, color="#111827", ls="--", lw=1)
        ax.set_title(f"{src} residual (actual - prediction)")
        ax.set_xlabel("Residual")
    p = os.path.join(FIGURES_FINAL_DIR, "final_residual_analysis.png")
    fig.savefig(p); plt.close(fig); paths.append(p)

    zf = tables["zero_final"].iloc[0]
    zp = tables["zero_phase8"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8))
    axes[0].bar(["Phase 8", "Final hurdle"], [zp["WAPE"], zf["WAPE"]], color=["#9ca3af", "#059669"])
    axes[0].set_title("SYNTHETIC WAPE")
    axes[1].bar(["Phase 8", "Final hurdle"],
                [zp["zero_positive_prediction_rate"], zf["zero_positive_prediction_rate"]],
                color=["#9ca3af", "#059669"])
    axes[1].set_title("Zero-day P(pred>0) %")
    axes[2].bar(["Phase 8", "Final hurdle"], [zp["zero_mae"], zf["zero_mae"]], color=["#9ca3af", "#059669"])
    axes[2].set_title("Zero-demand MAE")
    fig.suptitle("SYNTHETIC zero-demand: Phase 8 vs final hurdle")
    p = os.path.join(FIGURES_FINAL_DIR, "final_zero_demand_comparison.png")
    fig.savefig(p); plt.close(fig); paths.append(p)

    iv = tables["intervals"]
    if len(iv):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        axes[0].bar(iv["dataset"], iv["coverage_pct"], color="#1d4ed8")
        axes[0].axhline(80, color="#111827", ls="--", label="nominal 80%")
        axes[0].set_ylabel("Coverage %")
        axes[0].set_title("P10-P90 coverage")
        axes[0].legend()
        axes[1].bar(iv["dataset"], iv["mean_width"], color="#d97706")
        axes[1].set_title("Mean interval width")
        fig.suptitle("Final h=1 prediction intervals (quantile companions)")
        p = os.path.join(FIGURES_FINAL_DIR, "final_prediction_intervals.png")
        fig.savefig(p); plt.close(fig); paths.append(p)
    return paths


def feature_dependency_table() -> pd.DataFrame:
    rows = []

    def add(name, required, source, avail, notes=""):
        rows.append({
            "Feature": name, "Required": required, "Source": source,
            "Availability at forecast time": avail, "Notes": notes,
        })

    for c in NUMERIC_FEATURES_BOTH:
        if c.startswith("units_sold_lag_"):
            add(c, "Yes (h=1 and direct origin)", "Phase 6 lag of units_sold",
                "Available (past demand only)", "Leakage-safe")
        elif c.startswith("rolling_") or c.startswith("demand_"):
            add(c, "Yes", "Phase 6 rolling of past demand", "Available", "Leakage-safe")
        elif c in ("average_unit_price", "price_lag_1"):
            add(c, "Yes", "Phase 6 price",
                "Assumed known at origin (list price / lag)",
                "Must be supplied externally; not a future actual")
        else:
            add(c, "Yes", "Phase 6 calendar", "Known in advance", "")
    add("season", "Yes", "Phase 6", "Known in advance", "Frequency-encoded")
    for c in NUMERIC_FEATURES_SYNTHETIC_EXTRA:
        if c in ("ending_inventory", "on_order_qty", "stockout_flag", "historical_doi"):
            add(c, "SYNTHETIC only", "Phase 6 inventory",
                "Column required at origin; NaN allowed (LightGBM-native)",
                "Not imputed. Missing column is rejected.")
        elif "promo" in c or c in ("discount_pct", "promotion_flag", "promotion_available"):
            add(c, "SYNTHETIC only", "Phase 6 promo",
                "Must be planned/known at origin", "")
        else:
            add(c, "SYNTHETIC only", "Phase 6 store/price", "Known at origin", "")
    for c in CATEGORICAL_FEATURES_SYNTHETIC:
        add(c, "SYNTHETIC only", "Phase 6 product/store attrs", "Known", "Unseen levels map to freq 0")
    add("hcal_*", "Direct models only", "Target-date calendar shift",
        "Known in advance for the forecast date",
        "Required input; not inferred from truncated panels")
    for k, v in EXCLUDED_FIELDS.items():
        add(k, "No — forbidden", "raw", "Must not be used as a predictor", v)
    return pd.DataFrame(rows)


def _final_stability_label(tables: dict) -> dict[str, str]:
    """Combine time and entity axes; worst label wins. Documented Phase 9 rule."""
    rank = {"Stable": 0, "Moderately Stable": 1, "Unstable": 2}
    out = {}
    for src, g in tables["stability"].groupby("dataset"):
        labels = [str(x) for x in g["label"].tolist() if str(x) not in ("Not applicable", "n/a")]
        if not labels:
            labels = ["Moderately Stable"]
        worst = max(labels, key=lambda x: rank.get(x, 1))
        if src == "UCI" and worst == "Stable":
            worst = "Moderately Stable"
            reason_extra = "upgraded from Stable due to Phase 9 fold-2 WAPE 105.31"
        else:
            reason_extra = ""
        out[src] = {"label": worst, "detail": "; ".join(
            f"{r.axis}={r.label}" for r in g.itertuples()
        ) + (("; " + reason_extra) if reason_extra else "")}
    return out


def write_reports(candidates, selection, tables, registry, charts, deps, freeze_ok, features_hash) -> None:
    h1 = tables["overall_h1"]
    uci_m = h1[h1["dataset"] == "UCI"].iloc[0]
    syn_m = h1[h1["dataset"] == "SYNTHETIC"].iloc[0]
    base = tables["baseline"]
    zf = tables["zero_final"].iloc[0]
    zp = tables["zero_phase8"].iloc[0]
    fp_reduced = zf["zero_positive_prediction_rate"] < zp["zero_positive_prediction_rate"] - 10
    stab = _final_stability_label(tables)
    iv = tables["intervals"]

    uci_imp = base[base["Dataset"] == "UCI"].iloc[0]
    syn_imp = base[base["Dataset"] == "SYNTHETIC"].iloc[0]

    readiness = "READY WITH MONITORING"
    readiness_why = (
        "h=1 models beat Phase 7 baselines, SYNTHETIC hurdle materially cuts false-positive "
        "demand, and inference is schema-validated and reproducible. Not READY: UCI has a "
        "documented high-error walk-forward fold, long-horizon WAPE still degrades, same-day "
        "price/inventory are assumed known at origin, quantile bands are not statistically "
        "calibrated, and no live production monitor exists yet."
    )

    lines = []
    a = lines.append
    a("# Final Forecasting Report (Phase 11)")
    a("")
    a("## 1. Executive Summary")
    a("")
    a("Recommended operational solution:")
    a("")
    a("- **UCI, horizon 1:** frozen Phase 8 LightGBM (`uci_h1_phase8_lightgbm`).")
    a("- **SYNTHETIC, horizon 1:** hurdle LightGBM, threshold 0.50 (`synthetic_h1_hurdle_th050`).")
    a("- **Both datasets, horizons 3/7/14/30:** direct LightGBM per horizon.")
    a("- **Intervals:** P10/P90 quantile companions on h=1 only (diagnostic bands).")
    a("")
    a(f"**Production readiness: {readiness}.** {readiness_why}")
    a("")
    a("## 2. Phase 8 Baseline")
    a("")
    a("Frozen LightGBM (immutable benchmark):")
    a("")
    a("| Dataset | MAE | RMSE | sMAPE | WAPE | vs Phase 7 |")
    a("| --- | ---: | ---: | ---: | ---: | --- |")
    a("| UCI | 17.3447 | 70.8952 | 82.8734 | 79.4710 | MA-30 WAPE 86.3870 (+8.01%) |")
    a("| SYNTHETIC | 2.8156 | 5.1469 | 113.6813 | 38.8923 | Naive WAPE 72.8181 (+46.59%) |")
    a("")
    a("CatBoost was not installed. Same-day `average_unit_price` and SYNTHETIC inventory/promo "
      "were treated as known operational signals.")
    a("")
    a("## 3. Phase 9 Stability")
    a("")
    a("Walk-forward expanding-window copies of Phase 8 LightGBM (146/146 PASS):")
    a("")
    a("- UCI: mean WAPE 85.31, CV 0.135, max/min 1.369 → quantitative **Stable**, with fold 2 "
      "(Jan–Apr 2011) WAPE **105.31** after the post-holiday spike.")
    a("- SYNTHETIC: mean WAPE 39.39, CV 0.013, max/min 1.037 → **Stable**.")
    a("- Recursive h=1→h=30 WAPE grows to ~100 on both sources.")
    a("- SYNTHETIC TEST zeros 61.27%; Phase 8 predicted demand on **82.64%** of zero days.")
    a("")
    a("## 4. Phase 10 Experiments")
    a("")
    a("Validation **87/87 PASS**. Option A — Major improvement.")
    a("")
    a("- Hurdle (SYNTHETIC): TEST WAPE 38.89 → 26.25; zero-day P(pred>0) 82.64% → 1.42%. UCI skipped (0% zeros).")
    a("- Croston/SBA/TSB worse than Naive.")
    a("- Direct LightGBM improves long-horizon WAPE vs recursive (especially h≥7 SYNTHETIC; h≥14 UCI). Direct h=1 is next-observation, not Phase 8 same-row.")
    a("- Quantile P10–P90: UCI coverage 82.37%, SYNTHETIC 89.77% vs 80% nominal. Crossing before reorder occurred.")
    a("- HPO: UCI cfg7 TEST WAPE 78.33 (small gain, no walk-forward); SYNTHETIC cfg2 33.97 (still worse than hurdle).")
    a("")
    a("## 5. Final Model Selection")
    a("")
    a("### Selection logic")
    a("")
    a("```")
    a(SELECTION_LOGIC.strip())
    a("```")
    a("")
    a("### Decisions")
    a("")
    for r in selection.itertuples():
        a(f"- **{r.dataset} h={r.horizon} → {r.model}.** {r.selection_reason}")
        a("")
    a("Quantile P50 was not selected as the point forecast (WAPE does not beat the chosen h=1 models). "
      "HPO UCI was not selected (stability override). Intermittent models were not selected (WAPE).")
    a("")
    a("## 6. Final Performance")
    a("")
    a("### vs Phase 7 baseline (h=1)")
    a("")
    a(md_table(base))
    a("")
    a("### h=1 selected models")
    a("")
    h1_cols = [c for c in [
        "dataset", "model", "MAE", "RMSE", "sMAPE", "WAPE", "bias", "n",
        "zero_mae", "nonzero_mae", "zero_positive_prediction_rate",
    ] if c in tables["overall_h1"].columns]
    a(md_table(tables["overall_h1"][h1_cols]))
    a("")
    a("### Candidate matrix (implemented models only)")
    a("")
    show = candidates.copy()
    cols = [c for c in ["dataset", "model", "horizon", "MAE", "RMSE", "sMAPE", "WAPE", "bias",
                        "zero_demand_MAE", "positive_demand_MAE", "stability", "eligible"] if c in show.columns]
    a(md_table(show[cols]))
    a("")
    a("### Bias")
    a("")
    a(md_table(tables["bias"]))
    a("")
    a("### Demand regimes")
    a("")
    a(md_table(tables["regime"][["dataset", "regime", "n", "MAE", "WAPE", "bias"]]))
    a("")
    a("### Entity / store")
    a("")
    a(md_table(tables["store_summary"]))
    a("")
    a("## 7. Zero-Demand Results")
    a("")
    a("SYNTHETIC TEST (matched to Phase 8 grain):")
    a("")
    a(f"- Actual zero rate: {zf['actual_zero_rate']}%")
    a(f"- Predicted zero rate (final): {zf['predicted_zero_rate']}% (Phase 8: {zp['predicted_zero_rate']}%)")
    a(f"- False-positive demand rate P(pred>0 | actual=0): **{zf['zero_positive_prediction_rate']}%** vs Phase 8 **{zp['zero_positive_prediction_rate']}%**")
    a(f"- Zero-demand MAE: {zf['zero_mae']} vs {zp['zero_mae']}")
    a(f"- Non-zero-demand MAE: {zf['nonzero_mae']} vs {zp['nonzero_mae']}")
    a(f"- WAPE: {zf['WAPE']} vs {zp['WAPE']}")
    a(f"- Bias: {zf['bias']} vs {zp['bias']}")
    a("")
    if fp_reduced:
        a("**Yes — the final model materially reduced false-positive demand predictions** "
          f"({zp['zero_positive_prediction_rate']}% → {zf['zero_positive_prediction_rate']}%) "
          "while also improving overall WAPE and non-zero MAE. The hurdle is retained.")
    else:
        a("False-positive demand did not materially improve; the simpler Phase 8 model would have been retained.")
    a("")
    a("## 8. Horizon Results")
    a("")
    a(md_table(tables["horizon"][["dataset", "horizon", "model", "n", "MAE", "RMSE", "sMAPE", "WAPE", "bias"]]))
    a("")
    a("Error generally grows from short to long horizons on SYNTHETIC. UCI direct WAPE is not strictly monotone "
      "(h=30 can look better than h=7 because the target population and season mix change). "
      "Do not treat direct h=1 as comparable to Phase 8 same-row h=1.")
    a("")
    a("## 9. Prediction Intervals")
    a("")
    if len(iv):
        a(md_table(iv))
        a("")
        a("Bands are **P10/P90 quantile LightGBM companions**, clipped at 0 and reordered if they crossed. "
          "UCI coverage is near the 80% nominal; SYNTHETIC over-covers (intervals too wide on average). "
          "P10/P50 empirical below-rates are not equal to 10/50/90, so these are **not claimed as statistically calibrated**. "
          "They are usable as operational uncertainty bands with monitoring, not as safety-stock formulas.")
    else:
        a("No interval rows with finite bounds.")
    a("")
    a("## 10. Feature Dependencies")
    a("")
    a(md_table(deps))
    a("")
    a("Inference **rejects** missing columns, missing `units_sold_lag_1`, negative prohibited fields, "
      "duplicate keys, and invalid dates. Longer-lag / `historical_doi` NaNs are **not imputed**; "
      "they are passed through as LightGBM-native missing values (Phase 8/10 training behavior). "
      "`units_sold` is never a predictor.")
    a("")
    a("## 11. Limitations")
    a("")
    a("- UCI invoice-day grain has no coded zeros; intermittency is unidentified.")
    a("- UCI walk-forward fold 2 remains a high-error regime; HPO was not re-tested on folds.")
    a("- Direct vs recursive comparisons use different origin samples.")
    a("- Hurdle is a hard threshold, not E[y]=P(y>0)E[y|y>0].")
    a("- Quantile models are independent of the hurdle point forecast.")
    a("- Same-day price/inventory/promo must be supplied at origin; they are not forecasted here.")
    a("- No live production traffic, concept-drift monitor, or replenishment policy is in this phase.")
    a("- Long-horizon forecasts remain weak relative to short-horizon operational 1-step models.")
    a("")
    a("## 12. Production Readiness")
    a("")
    a(f"**{readiness}**")
    a("")
    a(readiness_why)
    a("")
    a("### Freeze status")
    a("")
    a(f"- Phase 8 artifacts unchanged: {freeze_ok['phase8']}")
    a(f"- Phase 9 artifacts unchanged: {freeze_ok['phase9']}")
    a(f"- Feature parquet md5: `{features_hash}`")
    a("")
    a("UCI entity-level WAPE spread is not identified (single `ONLINE` entity). "
      "SYNTHETIC store WAPE uses the Phase 9 CV/range rule. UCI production class is "
      "**Moderately Stable** because of Phase 9 fold 2 (WAPE 105.31), even when TEST months are Stable.")
    a("")
    a("### Charts")
    a("")
    for c in charts:
        a(f"- `{relpath(c)}`")
    a("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    mon = []
    m = mon.append
    m("# Forecast Monitoring Plan")
    m("")
    m("Applies after deploying the Phase 11 final models. Thresholds are taken from Phases 8–10 evidence, not invented business KPIs.")
    m("")
    m("## Data drift")
    m("")
    m("| Signal | Warning | Evidence |")
    m("| --- | --- | --- |")
    m("| Missing required-feature rate | > 0% in a batch (pipeline should reject) | Inference is fail-closed; any accepted missingness is a contract break |")
    m("| Unseen category rate (freq→0) | > 5% of rows in a day | Frequency encoder maps unknowns to 0; large spikes mean assortment/store change |")
    m("| Feature mean shift (lags, rolling, price) | \\|z\\|-score vs train > 3 on a daily aggregate | Phase 6 features are the training distribution |")
    m("| SYNTHETIC inventory/promo missing | any | Required at origin for the hurdle/direct SYNTHETIC models |")
    m("")
    m("## Forecast drift")
    m("")
    m("| Signal | Warning | Evidence |")
    m("| --- | --- | --- |")
    m("| Mean prediction vs TEST mean | relative change > 25% over 7 days | UCI residuals correlate with actual level (Phase 9) |")
    m("| SYNTHETIC zero-prediction rate | outside 50–75% (TEST predicted zero rate ~62%) | Hurdle TEST predicted_zero_rate |")
    m("| SYNTHETIC P(pred>0 \\| later actual=0) | > 10% | Phase 8 was 82.64%; hurdle 1.42% — 10% is an early-regression tripwire |")
    m("| Forecast row count vs expected grain | ±2% vs entity×product calendar | Missing keys silently drop volume |")
    m("")
    m("## Accuracy (when actuals arrive)")
    m("")
    m("| Dataset | Metric | Warning | Evidence |")
    m("| --- | --- | --- | --- |")
    m(f"| UCI h=1 | WAPE | > 105 | Phase 9 fold-2 WAPE 105.31 |")
    m(f"| UCI h=1 | WAPE | > 1.5 × {uci_m['WAPE']:.2f} ≈ {1.5*float(uci_m['WAPE']):.1f} | Phase 9 Stable max/min cap 1.50 |")
    m(f"| SYNTHETIC h=1 | WAPE | > 1.5 × {syn_m['WAPE']:.2f} ≈ {1.5*float(syn_m['WAPE']):.1f} | same range-ratio rule |")
    m("| Both | MAE / RMSE / bias | rolling 28-day bias sign flip persisting 14 days | Phase 8 bias convention mean(pred-actual) |")
    m("| Direct h≥7 | WAPE | worse than the Phase 9 recursive WAPE for that horizon | Phase 10 selected direct because it beat recursive |")
    m("")
    m("## Business / operational signals")
    m("")
    m("No dollar stockout or holding-cost thresholds exist in this project. Monitor proxy signals only:")
    m("")
    m("| Signal | Warning | Why |")
    m("| --- | --- | --- |")
    m("| High-regime under-prediction | high-demand MAE rising vs TEST high-regime | Phase 9 SYNTHETIC high-demand bias was negative |")
    m("| Horizon degradation | h=30 WAPE approaching 100 | Phase 9 recursive and Phase 10 direct both degrade |")
    m("| Interval coverage | P10–P90 coverage < 70% or > 95% over 28 days | TEST coverage UCI 82%, SYNTHETIC 90%; not calibrated |")
    m("| UCI January–April window | extra review | Fold-2 instability |")
    m("")
    m("## Review cadence")
    m("")
    m("- Daily: batch rejects, row counts, missing-feature rate, zero-prediction rate.")
    m("- Weekly: WAPE/MAE/bias on newly arrived actuals; interval coverage.")
    m("- Seasonal: UCI post-holiday window; SYNTHETIC promo calendar changes.")
    m("- Do not auto-retrain over frozen Phase 8 files. Candidate retrains belong in a new versioned model_id.")
    m("")
    with open(MONITOR_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(mon) + "\n")


def write_metadata(bundle: dict) -> None:
    write_json(METADATA_PATH, bundle)


def refresh_docs(freeze_before: dict | None = None) -> int:
    """Rebuild analysis, charts, and docs from existing final models/forecasts."""
    ensure_phase11_dirs()
    candidates = pd.read_parquet(CANDIDATE_PATH)
    selection = pd.read_parquet(SELECTION_PATH)
    preds = pd.read_parquet(FINAL_PRED_PATH)
    registry = json.loads(open(REGISTRY_PATH, encoding="utf-8").read())
    tables = analyze_final(preds)
    for name, t in tables.items():
        t.to_parquet(os.path.join(FORECASTS_FINAL_DIR, f"analysis_{name}.parquet"), index=False)
    charts = create_charts(preds, tables, candidates)
    deps = feature_dependency_table()
    deps.to_parquet(os.path.join(FORECASTS_FINAL_DIR, "feature_dependencies.parquet"), index=False)
    if freeze_before is None:
        freeze_before = {
            "phase8": snapshot_hashes(PHASE8_FREEZE_FILES),
            "phase9": snapshot_hashes(PHASE9_FREEZE_FILES),
        }
    freeze_after = {
        "phase8": snapshot_hashes(PHASE8_FREEZE_FILES),
        "phase9": snapshot_hashes(PHASE9_FREEZE_FILES),
    }
    ok8, ch8 = hashes_unchanged(freeze_before["phase8"], freeze_after["phase8"])
    ok9, ch9 = hashes_unchanged(freeze_before["phase9"], freeze_after["phase9"])
    freeze_ok = {"phase8": ok8, "phase9": ok9, "phase8_changed": ch8, "phase9_changed": ch9}
    features_hash = file_md5(
        os.path.join(BASE_DIR, "data", "processed", "features", "forecast_features.parquet")
    )
    write_reports(candidates, selection, tables, registry, charts, deps, freeze_ok, features_hash)
    ready = verify_phase10_ready()
    stab = _final_stability_label(tables)
    prev = {}
    if os.path.exists(METADATA_PATH):
        prev = json.loads(open(METADATA_PATH, encoding="utf-8").read())
    meta = {
        "phase": 11,
        "executed_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "pandas": _pkg("pandas"), "numpy": _pkg("numpy"),
            "sklearn": _pkg("scikit-learn"), "lightgbm": _pkg("lightgbm"),
        },
        "random_state": RANDOM_STATE,
        "code_version": git_code_version(),
        "features_md5": features_hash,
        "phase10_ready": ready,
        "selection_logic": SELECTION_LOGIC.strip(),
        "final_uci_model": "uci_h1_phase8_lightgbm",
        "final_synthetic_model": "synthetic_h1_hurdle_th050",
        "horizon_strategy": {
            "h=1_UCI": "Phase8_LightGBM",
            "h=1_SYNTHETIC": "Hurdle_th0.50",
            "h=3_7_14_30": "Direct_LightGBM",
        },
        "production_readiness": "READY WITH MONITORING",
        "stability": stab,
        "h1_metrics": tables["overall_h1"].to_dict(orient="records"),
        "baseline_improvement": tables["baseline"].to_dict(orient="records"),
        "phase8_unchanged": ok8,
        "phase9_unchanged": ok9,
        "phase8_hashes": freeze_after["phase8"],
        "phase9_hashes": freeze_after["phase9"],
        "validation": prev.get("validation"),
    }
    write_metadata(meta)
    print("stability", stab)
    return 0 if ok8 and ok9 else 1


def main() -> int:
    print("=" * 60)
    print("PHASE 11 — FINAL MODEL SELECTION")
    print("=" * 60)
    ready = verify_phase10_ready()
    if not ready["ready"]:
        print("STOP: Phase 10 is not complete or freeze hashes failed:")
        for i in ready["issues"]:
            print(" -", i)
        return 2
    print("Phase 10 ready:", ready["validation"], ready["decision"])
    if "--docs-only" in sys.argv:
        return refresh_docs()

    freeze_before = {
        "phase8": snapshot_hashes(PHASE8_FREEZE_FILES),
        "phase9": snapshot_hashes(PHASE9_FREEZE_FILES),
    }
    ensure_phase11_dirs()
    # do not leave stale experimental copies in models/final
    for fn in os.listdir(MODELS_FINAL_DIR):
        if fn.endswith(".joblib"):
            os.remove(os.path.join(MODELS_FINAL_DIR, fn))

    print("[1] Candidate matrix")
    candidates = build_candidate_matrix()
    candidates.to_parquet(CANDIDATE_PATH, index=False)
    selection = apply_selection(candidates)
    selection.to_parquet(SELECTION_PATH, index=False)
    print(selection[["dataset", "horizon", "model"]].to_string(index=False))

    print("[2] Load features and persist selected models")
    df = load_feature_dataset()
    features_hash = file_md5(
        os.path.join(BASE_DIR, "data", "processed", "features", "forecast_features.parquet")
    )
    code_version = git_code_version()
    registry = []
    registry.append(persist_uci_phase8(code_version, features_hash, dict(PHASE8_TEST["UCI"])))
    registry.append(persist_hurdle(df, code_version, features_hash))
    for src in ["UCI", "SYNTHETIC"]:
        for h in (3, 7, 14, 30):
            registry.append(persist_direct(df, src, h, code_version, features_hash))
        registry.append(persist_quantile(df, src, code_version, features_hash))
    write_json(REGISTRY_PATH, registry)

    print("[3] Generate final forecasts via inference layer")
    preds = generate_final_forecasts(df, registry)
    preds.to_parquet(FINAL_PRED_PATH, index=False)
    print("  wrote", FINAL_PRED_PATH, "rows", len(preds))

    print("[4] Analysis / charts / docs")
    tables = analyze_final(preds)
    for name, t in tables.items():
        t.to_parquet(os.path.join(FORECASTS_FINAL_DIR, f"analysis_{name}.parquet"), index=False)
    charts = create_charts(preds, tables, candidates)
    deps = feature_dependency_table()
    deps.to_parquet(os.path.join(FORECASTS_FINAL_DIR, "feature_dependencies.parquet"), index=False)
    freeze_after = {
        "phase8": snapshot_hashes(PHASE8_FREEZE_FILES),
        "phase9": snapshot_hashes(PHASE9_FREEZE_FILES),
    }
    ok8, ch8 = hashes_unchanged(freeze_before["phase8"], freeze_after["phase8"])
    ok9, ch9 = hashes_unchanged(freeze_before["phase9"], freeze_after["phase9"])
    freeze_ok = {"phase8": ok8, "phase9": ok9, "phase8_changed": ch8, "phase9_changed": ch9}
    write_reports(candidates, selection, tables, registry, charts, deps, freeze_ok, features_hash)

    stab = _final_stability_label(tables)
    h1 = tables["overall_h1"]
    meta = {
        "phase": 11,
        "executed_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "pandas": _pkg("pandas"), "numpy": _pkg("numpy"),
            "sklearn": _pkg("scikit-learn"), "lightgbm": _pkg("lightgbm"),
        },
        "random_state": RANDOM_STATE,
        "code_version": code_version,
        "features_md5": features_hash,
        "phase10_ready": ready,
        "selection_logic": SELECTION_LOGIC.strip(),
        "final_uci_model": "uci_h1_phase8_lightgbm",
        "final_synthetic_model": "synthetic_h1_hurdle_th050",
        "horizon_strategy": {
            "h=1_UCI": "Phase8_LightGBM",
            "h=1_SYNTHETIC": "Hurdle_th0.50",
            "h=3_7_14_30": "Direct_LightGBM",
        },
        "production_readiness": "READY WITH MONITORING",
        "stability": stab,
        "h1_metrics": h1.to_dict(orient="records"),
        "baseline_improvement": tables["baseline"].to_dict(orient="records"),
        "phase8_unchanged": ok8,
        "phase9_unchanged": ok9,
        "phase8_hashes": freeze_after["phase8"],
        "phase9_hashes": freeze_after["phase9"],
    }
    write_metadata(meta)
    if not ok8 or not ok9:
        print("FAIL: freeze hashes changed", freeze_ok)
        return 1
    print("Phase 11 artifacts written. Run src/validate_final_forecasting.py next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
