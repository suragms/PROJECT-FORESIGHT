"""
Phase 8 — Machine Learning Demand Forecasting
==============================================
Project FORESIGHT: Demand & Inventory Intelligence

Trains source-specific tabular ML models on Phase 6
``forecast_features.parquet`` and compares against Phase 7 baselines.

Models (installed): RandomForest, HistGradientBoosting, LightGBM, XGBoost.
CatBoost is not installed — skipped intentionally.

Selection: validation WAPE → MAE → RMSE → sMAPE
Final reporting: untouched TEST only after selection.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.baseline_forecasting import calculate_metrics  # Phase 7 metric parity

FEATURES_PATH = os.path.join(
    BASE_DIR, "data", "processed", "features", "forecast_features.parquet"
)
ML_DIR = os.path.join(BASE_DIR, "data", "processed", "forecasts", "ml")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures", "forecasting", "ml")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
METADATA_PATH = os.path.join(DOCS_DIR, "model_training_metadata.json")

TARGET = "units_sold"
GRAIN = ["date", "source_dataset", "entity_id", "product_key"]
RANDOM_STATE = 42

# Official Phase 7 TEST benchmarks
BASELINE_BENCHMARKS = {
    "SYNTHETIC": {
        "model": "naive",
        "WAPE": 72.8181,
        "MAE": 5.2717,
        "RMSE": 10.4688,
        "sMAPE": 45.1750,
    },
    "UCI": {
        "model": "moving_average_30",
        "WAPE": 86.3870,
        "MAE": 18.8542,
        "RMSE": 72.0799,
        "sMAPE": 84.3796,
    },
}

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

NUMERIC_FEATURES_BOTH = [
    # calendar / cyclical
    "year", "month", "quarter", "week_of_year", "day_of_week",
    "day_of_month", "day_of_year", "is_weekend",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_holiday",
    # lags / rolling / trend (Phase 6 leakage-safe)
    "units_sold_lag_1", "units_sold_lag_2", "units_sold_lag_3",
    "units_sold_lag_7", "units_sold_lag_14", "units_sold_lag_21",
    "units_sold_lag_28", "units_sold_lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_std_7", "rolling_std_14", "rolling_std_30",
    "demand_change_1", "demand_change_7", "demand_growth_7", "demand_growth_30",
    # price (lag-safe preferred; same-day avg price treated as known list price signal)
    "average_unit_price", "price_lag_1",
]

NUMERIC_FEATURES_SYNTHETIC_EXTRA = [
    "base_price", "discount_pct", "price_change",
    "promotion_flag", "promotion_available", "promo_rolling_7",
    "store_size_sqft",
    "ending_inventory", "on_order_qty", "stockout_flag", "historical_doi",
]

CATEGORICAL_FEATURES_BOTH = ["season"]
CATEGORICAL_FEATURES_SYNTHETIC = [
    "category", "sub_category", "brand", "region", "store_type",
]

EXCLUDED_FIELDS = {
    "units_sold": "TARGET — never used as a predictor",
    "revenue": "Same-day revenue leaks target (≈ price × units)",
    "transaction_count": "Same-day order activity contemporaneous with demand",
    "unique_customers": "Same-day customer activity contemporaneous with demand",
    "date": "Temporal key — encoded via calendar features",
    "source_dataset": "Partition key — models trained per source",
    "entity_id": "Raw ID — use store attributes instead",
    "product_key": "Raw ID — use product attributes instead",
    "sku_id": "Raw ID duplicate of product identity",
    "entity_type": "Near-constant within source; not used as numeric code",
    "split": "Metadata for chronological partitioning only",
    "insufficient_history": "Metadata flag — rows dropped via lag availability",
}


def load_feature_dataset(path: str = FEATURES_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Phase 6 features missing: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    required = GRAIN + [TARGET, "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.sort_values(GRAIN).reset_index(drop=True)


def feature_lists_for_source(source: str) -> dict[str, list[str]]:
    numeric = list(NUMERIC_FEATURES_BOTH)
    cats = list(CATEGORICAL_FEATURES_BOTH)
    if source == "SYNTHETIC":
        numeric = numeric + [c for c in NUMERIC_FEATURES_SYNTHETIC_EXTRA]
        cats = cats + [c for c in CATEGORICAL_FEATURES_SYNTHETIC]
    return {"numeric": numeric, "categorical": cats}


def prepare_features(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Filter to one source and return frame + feature lists present in data."""
    sub = df[df["source_dataset"] == source].copy()
    lists = feature_lists_for_source(source)
    numeric = [c for c in lists["numeric"] if c in sub.columns]
    cats = [c for c in lists["categorical"] if c in sub.columns]
    return sub, numeric, cats


def prepare_train_validation_test(
    df_src: pd.DataFrame,
    numeric: list[str],
    cats: list[str],
) -> dict[str, Any]:
    """
    Chronological splits from Phase 6 `split` column.
    Drop rows lacking lag_1 (insufficient history) from train/val/test alike.
    """
    feat_cols = numeric + cats
    usable = df_src.copy()
    if "units_sold_lag_1" in usable.columns:
        usable = usable[usable["units_sold_lag_1"].notna()].copy()

    splits = {}
    for sp in ["train", "validation", "test"]:
        part = usable[usable["split"] == sp].copy()
        splits[sp] = {
            "X": part[feat_cols].copy(),
            "y": part[TARGET].astype(float).to_numpy(),
            "meta": part[GRAIN + ["revenue", "sku_id"]].copy()
            if "revenue" in part.columns
            else part[GRAIN].copy(),
            "n": len(part),
            "start": part["date"].min() if len(part) else None,
            "end": part["date"].max() if len(part) else None,
        }
    splits["feature_cols"] = feat_cols
    splits["numeric"] = numeric
    splits["categorical"] = cats
    return splits


# ---------------------------------------------------------------------------
# Preprocessing — frequency encode categoricals; median impute for RF only
# ---------------------------------------------------------------------------

class FeaturePreprocessor:
    """Train-fitted frequency encoding + optional median imputation."""

    def __init__(self, numeric: list[str], categorical: list[str], impute: bool):
        self.numeric = numeric
        self.categorical = categorical
        self.impute = impute
        self.freq_maps: dict[str, dict] = {}
        self.medians: dict[str, float] = {}
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        self.freq_maps = {}
        for c in self.categorical:
            vc = X[c].astype("string").value_counts(dropna=True)
            total = float(len(X))
            self.freq_maps[c] = {str(k): float(v) / total for k, v in vc.items()}
        if self.impute:
            for c in self.numeric:
                self.medians[c] = float(X[c].median()) if X[c].notna().any() else 0.0
        self.feature_names_ = list(self.numeric) + [f"{c}__freq" for c in self.categorical]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out = pd.DataFrame(index=X.index)
        for c in self.numeric:
            s = pd.to_numeric(X[c], errors="coerce")
            if self.impute:
                s = s.fillna(self.medians.get(c, 0.0))
            out[c] = s.astype(np.float32)
        for c in self.categorical:
            mapped = X[c].astype("string").astype(object).map(
                lambda v, m=self.freq_maps[c]: m.get(str(v), 0.0) if pd.notna(v) else 0.0
            )
            out[f"{c}__freq"] = mapped.astype(np.float32)
        return out.to_numpy(dtype=np.float32)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def build_preprocessor(numeric: list[str], categorical: list[str], impute: bool) -> FeaturePreprocessor:
    return FeaturePreprocessor(numeric, categorical, impute=impute)


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """Controlled RF — subsample rows for computational efficiency on large data."""
    model = RandomForestRegressor(
        n_estimators=40,
        max_depth=12,
        min_samples_leaf=8,
        max_features="sqrt",
        max_samples=min(250_000, len(y)) / len(y) if len(y) > 250_000 else None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def train_hist_gradient_boosting(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.06,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def train_lightgbm(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y, feature_name=feature_names)
    return model


def train_xgboost(X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X, y)
    return model


def _predict_nonneg(model, X: np.ndarray) -> np.ndarray:
    pred = np.asarray(model.predict(X), dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(0.0, pred)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    return calculate_metrics(y_true, y_pred, model_name=model_name)


# ---------------------------------------------------------------------------
# Train / select / test per source
# ---------------------------------------------------------------------------

def train_and_select_for_source(df: pd.DataFrame, source: str) -> dict[str, Any]:
    print(f"\n===== {source} =====")
    df_src, numeric, cats = prepare_features(df, source)
    splits = prepare_train_validation_test(df_src, numeric, cats)
    print(
        f"  rows train/val/test: "
        f"{splits['train']['n']:,} / {splits['validation']['n']:,} / {splits['test']['n']:,}"
    )
    print(f"  features: {len(splits['feature_cols'])} "
          f"({len(numeric)} numeric + {len(cats)} categorical->freq)")

    # Specs: (name, needs_impute, trainer)
    model_specs = [
        ("random_forest", True, lambda X, y, names: train_random_forest(X, y)),
        ("hist_gradient_boosting", False, lambda X, y, names: train_hist_gradient_boosting(X, y)),
        ("lightgbm", False, lambda X, y, names: train_lightgbm(X, y, names)),
        ("xgboost", False, lambda X, y, names: train_xgboost(X, y)),
    ]

    results = []
    artifacts = {}

    for name, needs_impute, trainer in model_specs:
        print(f"  Training {name}...")
        pre = build_preprocessor(numeric, cats, impute=needs_impute)
        t0 = time.perf_counter()
        X_train = pre.fit_transform(splits["train"]["X"])
        model = trainer(X_train, splits["train"]["y"], pre.feature_names_)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        X_val = pre.transform(splits["validation"]["X"])
        pred_val = _predict_nonneg(model, X_val)
        pred_time_val = time.perf_counter() - t1
        val_metrics = evaluate_model(splits["validation"]["y"], pred_val, name)

        results.append({
            "source_dataset": source,
            "model": name,
            "split": "validation",
            **{k: val_metrics[k] for k in ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "n"]},
            "training_time": round(train_time, 3),
            "prediction_time": round(pred_time_val, 3),
            "baseline_WAPE": BASELINE_BENCHMARKS[source]["WAPE"],
            "wape_improvement_pct": round(
                (BASELINE_BENCHMARKS[source]["WAPE"] - val_metrics["WAPE"])
                / BASELINE_BENCHMARKS[source]["WAPE"] * 100.0,
                4,
            ),
        })
        artifacts[name] = {
            "model": model,
            "preprocessor": pre,
            "train_time": train_time,
            "val_metrics": val_metrics,
            "feature_names": list(pre.feature_names_),
        }
        print(
            f"    val WAPE={val_metrics['WAPE']:.4f} MAE={val_metrics['MAE']:.4f} "
            f"time={train_time:.1f}s"
        )

    # Selection on VALIDATION only
    val_df = pd.DataFrame([r for r in results if r["split"] == "validation"])
    val_df = val_df.sort_values(["WAPE", "MAE", "RMSE", "sMAPE"]).reset_index(drop=True)
    best_name = val_df.iloc[0]["model"]
    print(f"  SELECTED (validation): {best_name}")

    # Final TEST evaluation for ALL models (reporting) but selection already locked
    test_rows = []
    test_preds_best = None
    for name, art in artifacts.items():
        pre = art["preprocessor"]
        model = art["model"]
        t1 = time.perf_counter()
        X_test = pre.transform(splits["test"]["X"])
        pred_test = _predict_nonneg(model, X_test)
        pred_time = time.perf_counter() - t1
        tm = evaluate_model(splits["test"]["y"], pred_test, name)
        test_rows.append({
            "source_dataset": source,
            "model": name,
            "split": "test",
            **{k: tm[k] for k in ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "n"]},
            "training_time": round(art["train_time"], 3),
            "prediction_time": round(pred_time, 3),
            "baseline_WAPE": BASELINE_BENCHMARKS[source]["WAPE"],
            "wape_improvement_pct": round(
                (BASELINE_BENCHMARKS[source]["WAPE"] - tm["WAPE"])
                / BASELINE_BENCHMARKS[source]["WAPE"] * 100.0,
                4,
            ),
            "selected": name == best_name,
        })
        if name == best_name:
            test_preds_best = pred_test
            print(
                f"  TEST best {name}: WAPE={tm['WAPE']:.4f} "
                f"(baseline {BASELINE_BENCHMARKS[source]['WAPE']:.4f}, "
                f"improv_pct={test_rows[-1]['wape_improvement_pct']:.2f})"
            )

    return {
        "source": source,
        "splits": splits,
        "artifacts": artifacts,
        "best_name": best_name,
        "val_metrics_table": val_df,
        "test_metrics_table": pd.DataFrame(test_rows),
        "all_metrics": pd.DataFrame(results + test_rows),
        "test_predictions": test_preds_best,
        "numeric": numeric,
        "categorical": cats,
        "excluded": EXCLUDED_FIELDS,
    }


def calculate_feature_importance(
    source_result: dict,
    max_perm_samples: int = 8000,
) -> pd.DataFrame:
    """Tree importance + permutation importance on VALIDATION (exploratory)."""
    source = source_result["source"]
    best = source_result["best_name"]
    art = source_result["artifacts"][best]
    model = art["model"]
    pre = art["preprocessor"]
    names = art["feature_names"]
    splits = source_result["splits"]

    rows = []
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(-imp)
        for rank, idx in enumerate(order, start=1):
            rows.append({
                "source_dataset": source,
                "model": best,
                "feature": names[int(idx)],
                "importance": float(imp[idx]),
                "importance_type": "native",
                "rank": int(rank),
            })

    X_val = pre.transform(splits["validation"]["X"])
    y_val = splits["validation"]["y"]
    rng = np.random.default_rng(RANDOM_STATE)
    if len(y_val) > max_perm_samples:
        idx = rng.choice(len(y_val), size=max_perm_samples, replace=False)
        X_p, y_p = X_val[idx], y_val[idx]
    else:
        X_p, y_p = X_val, y_val

    try:
        perm = permutation_importance(
            model, X_p, y_p, n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1,
            scoring="neg_mean_absolute_error",
        )
        order = np.argsort(-perm.importances_mean)
        for rank, i in enumerate(order, start=1):
            rows.append({
                "source_dataset": source,
                "model": best,
                "feature": names[int(i)],
                "importance": float(perm.importances_mean[i]),
                "importance_type": "permutation_mae",
                "rank": int(rank),
            })
    except Exception as e:
        print(f"  permutation importance skipped: {e}")

    return pd.DataFrame(rows)


def product_store_analysis(source_result: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Best-model TEST metrics by product and (Synthetic) entity + Pareto split."""
    source = source_result["source"]
    best = source_result["best_name"]
    meta = source_result["splits"]["test"]["meta"].reset_index(drop=True)
    y = source_result["splits"]["test"]["y"]
    pred = source_result["test_predictions"]
    frame = meta.copy()
    frame["actual"] = y
    frame["predicted"] = pred

    prod_rows = []
    for (ent, pk), g in frame.groupby(["entity_id", "product_key"], observed=True):
        m = evaluate_model(g["actual"].to_numpy(), g["predicted"].to_numpy(), best)
        prod_rows.append({
            "source_dataset": source,
            "model": best,
            "entity_id": ent,
            "product_key": pk,
            **{k: m[k] for k in ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "n"]},
            "units_total": float(g["actual"].sum()),
            "revenue_total": float(g["revenue"].sum()) if "revenue" in g.columns else np.nan,
        })
    by_product = pd.DataFrame(prod_rows)

    ent_rows = []
    for ent, g in frame.groupby("entity_id", observed=True):
        m = evaluate_model(g["actual"].to_numpy(), g["predicted"].to_numpy(), best)
        ent_rows.append({
            "source_dataset": source,
            "model": best,
            "entity_id": ent,
            **{k: m[k] for k in ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "n"]},
            "units_total": float(g["actual"].sum()),
            "revenue_total": float(g["revenue"].sum()) if "revenue" in g.columns else np.nan,
        })
    by_entity = pd.DataFrame(ent_rows)

    # Pareto high/low revenue from TEST revenue of this source's products
    sku_rev = frame.groupby("product_key", observed=True)["revenue"].sum().sort_values(ascending=False)
    total = float(sku_rev.sum()) if len(sku_rev) else 0.0
    hv_rows = []
    if total > 0:
        cum = sku_rev.cumsum() / total
        high = set(sku_rev.index[cum.shift(fill_value=0) < 0.80])
        low = set(sku_rev.index) - high
        agg = by_product.groupby("product_key", observed=True).agg(
            WAPE=("WAPE", "mean"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean"),
            revenue_total=("revenue_total", "sum"),
        )
        for label, skus in [("high_revenue", high), ("lower_revenue", low)]:
            part = agg.loc[agg.index.isin(skus)]
            if part.empty:
                continue
            hv_rows.append({
                "source_dataset": source,
                "segment": label,
                "n_skus": int(len(skus)),
                "revenue_share_pct": round(100.0 * float(sku_rev[sku_rev.index.isin(skus)].sum()) / total, 2),
                "model": best,
                "MAE": round(float(part["MAE"].mean()), 4),
                "RMSE": round(float(part["RMSE"].mean()), 4),
                "WAPE": round(float(part["WAPE"].mean()), 4),
            })
    high_value = pd.DataFrame(hv_rows)
    return by_product, by_entity, high_value


def error_analysis(source_result: dict) -> dict:
    y = source_result["splits"]["test"]["y"]
    pred = source_result["test_predictions"]
    err = pred - y
    abs_err = np.abs(err)
    return {
        "source_dataset": source_result["source"],
        "model": source_result["best_name"],
        "bias_mean_pred_minus_actual": round(float(np.mean(err)), 4),
        "median_error": round(float(np.median(err)), 4),
        "mean_abs_error": round(float(np.mean(abs_err)), 4),
        "p95_abs_error": round(float(np.quantile(abs_err, 0.95)), 4),
        "zero_demand_share_pct": round(100.0 * float(np.mean(y == 0)), 2),
        "zero_demand_mae": round(float(np.mean(abs_err[y == 0])), 4) if np.any(y == 0) else np.nan,
        "high_demand_mae": round(
            float(np.mean(abs_err[y >= np.quantile(y, 0.90)])), 4
        ) if len(y) else np.nan,
        "pct_overpredict": round(100.0 * float(np.mean(err > 0)), 2),
        "pct_underpredict": round(100.0 * float(np.mean(err < 0)), 2),
    }


# ---------------------------------------------------------------------------
# Persist / visualize
# ---------------------------------------------------------------------------

def save_predictions(source_results: dict[str, dict], path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(ML_DIR, "ml_predictions.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = []
    for src, res in source_results.items():
        meta = res["splits"]["test"]["meta"].reset_index(drop=True)
        frames.append(pd.DataFrame({
            "date": meta["date"].values,
            "source_dataset": src,
            "entity_id": meta["entity_id"].values,
            "product_key": meta["product_key"].values,
            "actual_units_sold": res["splits"]["test"]["y"],
            "predicted_units_sold": res["test_predictions"],
            "model": res["best_name"],
        }))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def save_models(source_results: dict[str, dict]) -> dict[str, str]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {}
    for src, res in source_results.items():
        best = res["best_name"]
        art = res["artifacts"][best]
        fname = f"{src.lower()}_best_model.joblib"
        path = os.path.join(MODELS_DIR, fname)
        # Ensure preprocessor pickles under src.ml_forecasting (not __main__)
        pre = art["preprocessor"]
        pre.__class__ = FeaturePreprocessor
        payload = {
            "source_dataset": src,
            "model_name": best,
            "model": art["model"],
            "preprocessor": pre,
            "feature_names": art["feature_names"],
            "numeric_features": res["numeric"],
            "categorical_features": res["categorical"],
            "excluded_fields": res["excluded"],
            "random_state": RANDOM_STATE,
            "baseline_benchmark": BASELINE_BENCHMARKS[src],
            "validation_metrics": art["val_metrics"],
            "test_metrics": res["test_metrics_table"]
            .query("model == @best")
            .iloc[0]
            .to_dict(),
        }
        joblib.dump(payload, path)
        paths[src] = path
        print(f"  Saved {path}")
    return paths


def create_importance_chart(importance: pd.DataFrame, out_dir: str = FIGURES_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "feature_importance.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, src in zip(axes, ["UCI", "SYNTHETIC"]):
        sub = importance[
            (importance["source_dataset"] == src)
            & (importance["importance_type"] == "native")
        ].sort_values("rank").head(20)
        if sub.empty:
            sub = importance[
                (importance["source_dataset"] == src)
                & (importance["importance_type"] == "permutation_mae")
            ].sort_values("rank").head(20)
        if sub.empty:
            continue
        ax.barh(sub["feature"], sub["importance"], color="#2563eb")
        ax.set_title(f"{src} — Top features ({sub['model'].iloc[0]})")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def create_forecast_charts(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    out_dir: str = FIGURES_DIR,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    def pick(source: str, by: str, ascending: bool = False):
        sub = df[df["source_dataset"] == source]
        ranked = (
            sub.groupby(["entity_id", "product_key"], observed=True)[by]
            .sum()
            .sort_values(ascending=ascending)
        )
        return ranked.index[0] if len(ranked) else (None, None)

    specs = [
        ("uci_high_volume", "UCI", "units_sold", False),
        ("uci_high_revenue", "UCI", "revenue", False),
        ("syn_store_sku", "SYNTHETIC", "units_sold", False),
        ("syn_high_revenue", "SYNTHETIC", "revenue", False),
    ]
    for name, src, by, asc in specs:
        ent, pk = pick(src, by, asc)
        if ent is None:
            continue
        g = preds[
            (preds["source_dataset"] == src)
            & (preds["entity_id"] == ent)
            & (preds["product_key"] == pk)
        ].sort_values("date").tail(60)
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(g["date"], g["actual_units_sold"], label="Actual", lw=2, color="#111827")
        ax.plot(g["date"], g["predicted_units_sold"], label=f"ML ({g['model'].iloc[0]})", alpha=0.9)
        ax.set_title(f"{name}: {src} | {ent} | {pk}")
        ax.set_ylabel("units_sold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(out_dir, f"{name}.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

    # WAPE comparison vs baseline
    path = os.path.join(out_dir, "ml_vs_baseline_wape.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, base_v, ml_v = [], [], []
    for src in ["UCI", "SYNTHETIC"]:
        labels.append(src)
        base_v.append(BASELINE_BENCHMARKS[src]["WAPE"])
    # filled by caller via preds model metrics — compute from preds
    for src in ["UCI", "SYNTHETIC"]:
        sub = preds[preds["source_dataset"] == src]
        m = evaluate_model(
            sub["actual_units_sold"].to_numpy(),
            sub["predicted_units_sold"].to_numpy(),
            "ml",
        )
        ml_v.append(m["WAPE"])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, base_v, 0.35, label="Phase 7 baseline")
    ax.bar(x + 0.18, ml_v, 0.35, label="Phase 8 best ML")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("TEST WAPE %")
    ax.set_title("ML vs Phase 7 Baseline (TEST WAPE)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(path)
    return paths


def write_metadata(source_results: dict[str, dict], model_paths: dict[str, str]) -> str:
    os.makedirs(DOCS_DIR, exist_ok=True)
    meta = {
        "phase": 8,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "features_path": FEATURES_PATH,
        "features_rows": None,
        "random_state": RANDOM_STATE,
        "baseline_benchmarks": BASELINE_BENCHMARKS,
        "excluded_fields": EXCLUDED_FIELDS,
        "sources": {},
    }
    for src, res in source_results.items():
        best = res["best_name"]
        test_row = res["test_metrics_table"].query("model == @best").iloc[0].to_dict()
        art = res["artifacts"][best]
        meta["sources"][src] = {
            "best_model": best,
            "model_path": model_paths.get(src),
            "train_rows": res["splits"]["train"]["n"],
            "validation_rows": res["splits"]["validation"]["n"],
            "test_rows": res["splits"]["test"]["n"],
            "train_start": str(res["splits"]["train"]["start"].date()),
            "train_end": str(res["splits"]["train"]["end"].date()),
            "validation_start": str(res["splits"]["validation"]["start"].date()),
            "validation_end": str(res["splits"]["validation"]["end"].date()),
            "test_start": str(res["splits"]["test"]["start"].date()),
            "test_end": str(res["splits"]["test"]["end"].date()),
            "feature_list": art["feature_names"],
            "parameters": _model_params(art["model"]),
            "validation_metrics": art["val_metrics"],
            "test_metrics": {
                k: test_row[k]
                for k in [
                    "MAE", "RMSE", "MAPE", "sMAPE", "WAPE",
                    "baseline_WAPE", "wape_improvement_pct", "n",
                ]
            },
            "training_time_sec": art["train_time"],
        }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return METADATA_PATH


def _model_params(model) -> dict:
    if hasattr(model, "get_params"):
        params = model.get_params()
        # keep JSON-safe scalars
        out = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
        return out
    return {"type": type(model).__name__}


def write_ml_report(
    df: pd.DataFrame,
    source_results: dict[str, dict],
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
    analyses: dict,
    chart_paths: list[str],
    validation_summary: str,
    files: list[str],
) -> str:
    path = os.path.join(DOCS_DIR, "ml_forecasting_report.md")
    lines = []
    lines.append("# Phase 8 — Machine Learning Demand Forecasting Report\n")
    lines.append(f"**Validation:** {validation_summary}\n")
    lines.append("## 1. Objective\n")
    lines.append(
        "Train source-specific ML demand models on Phase 6 features and beat "
        "Phase 7 baseline WAPE benchmarks on the untouched TEST window.\n"
    )
    lines.append("## 2. Input data\n")
    lines.append(f"- Path: `{FEATURES_PATH}`\n")
    lines.append(f"- Rows: **{len(df):,}** | Columns: **{df.shape[1]}**\n")
    lines.append("## 3. Feature groups\n")
    lines.append(
        "Numeric: calendar, cyclical, lag, rolling, demand trend, price; "
        "Synthetic extras: promo, inventory, store size. "
        "Categorical (frequency-encoded): season; Synthetic also category/"
        "sub_category/brand/region/store_type.\n"
    )
    lines.append("### Excluded fields\n")
    for k, v in EXCLUDED_FIELDS.items():
        lines.append(f"- `{k}`: {v}\n")
    lines.append("## 4. Models evaluated\n")
    lines.append(
        "random_forest, hist_gradient_boosting, lightgbm, xgboost "
        "(CatBoost not installed — skipped).\n"
    )
    lines.append("## 5. Training configuration\n")
    lines.append(f"- `random_state={RANDOM_STATE}`\n")
    lines.append("- Selection metric order: WAPE → MAE → RMSE → sMAPE on VALIDATION\n")
    lines.append("- TEST used only after selection\n")

    lines.append("## 6. Validation results\n")
    val = metrics[metrics["split"] == "validation"].sort_values(["source_dataset", "WAPE"])
    lines.append(val[
        ["source_dataset", "model", "MAE", "RMSE", "sMAPE", "WAPE", "training_time"]
    ].to_string(index=False))
    lines.append("\n")

    lines.append("## 7. Test results\n")
    test = metrics[metrics["split"] == "test"].sort_values(["source_dataset", "WAPE"])
    lines.append(test[
        ["source_dataset", "model", "MAE", "RMSE", "sMAPE", "WAPE",
         "baseline_WAPE", "wape_improvement_pct", "selected"]
    ].to_string(index=False))
    lines.append("\n")

    lines.append("## 8–11. Baseline comparison & best models\n")
    for src, res in source_results.items():
        best = res["best_name"]
        row = res["test_metrics_table"].query("model == @best").iloc[0]
        base = BASELINE_BENCHMARKS[src]
        beat = row["WAPE"] < base["WAPE"]
        lines.append(f"### {src}\n")
        lines.append(f"- Best ML model: **{best}**\n")
        lines.append(f"- Baseline: {base['model']} WAPE={base['WAPE']}\n")
        lines.append(
            f"- ML TEST WAPE={row['WAPE']:.4f} | MAE={row['MAE']:.4f} | "
            f"RMSE={row['RMSE']:.4f} | sMAPE={row['sMAPE']:.4f}\n"
        )
        lines.append(f"- WAPE improvement %: **{row['wape_improvement_pct']:.4f}**\n")
        lines.append(f"- Beat baseline on TEST: **{'YES' if beat else 'NO'}**\n")
        lines.append(
            f"OBSERVATION: {best} {'reduced' if beat else 'did not reduce'} TEST WAPE vs "
            f"{base['model']}.\n"
        )
        lines.append(
            f"EVIDENCE: baseline WAPE={base['WAPE']}, ML WAPE={row['WAPE']:.4f}, "
            f"improvement={row['wape_improvement_pct']:.4f}%.\n"
        )
        lines.append(
            "BUSINESS INTERPRETATION: "
            + (
                "Lag/calendar/price features add predictive structure beyond the baseline.\n"
                if beat
                else "Baseline remains competitive; revisit features/complexity in Phase 9.\n"
            )
        )
        lines.append(
            "BUSINESS ACTION: "
            + (
                "Prefer the selected ML model for planning with inventory constraints.\n"
                if beat
                else "Keep Phase 7 baseline as fallback until ML clearly outperforms.\n"
            )
        )

    lines.append("## 12. Product-level performance\n")
    for src, a in analyses.items():
        bp = a["by_product"].sort_values("WAPE")
        lines.append(f"### {src}\n")
        lines.append("Best WAPE SKUs:\n")
        for _, r in bp.head(3).iterrows():
            lines.append(
                f"- `{r['product_key']}` ({r['entity_id']}): WAPE={r['WAPE']:.2f}, "
                f"MAE={r['MAE']:.2f}\n"
            )
        lines.append("Worst WAPE SKUs:\n")
        for _, r in bp.tail(3).iloc[::-1].iterrows():
            lines.append(
                f"- `{r['product_key']}` ({r['entity_id']}): WAPE={r['WAPE']:.2f}, "
                f"MAE={r['MAE']:.2f}\n"
            )
        if not a["high_value"].empty:
            lines.append(a["high_value"].to_string(index=False))
            lines.append("\n")

    lines.append("## 13. Store-level performance (Synthetic)\n")
    be = analyses["SYNTHETIC"]["by_entity"].sort_values("WAPE")
    lines.append(be[["entity_id", "MAE", "RMSE", "WAPE"]].to_string(index=False))
    lines.append(
        f"\nBest store: {be.iloc[0]['entity_id']} WAPE={be.iloc[0]['WAPE']:.4f}; "
        f"Worst: {be.iloc[-1]['entity_id']} WAPE={be.iloc[-1]['WAPE']:.4f}\n"
    )

    lines.append("## 14. Feature importance (top 10 native)\n")
    for src in ["UCI", "SYNTHETIC"]:
        sub = importance[
            (importance.source_dataset == src) & (importance.importance_type == "native")
        ].sort_values("rank").head(10)
        lines.append(f"### {src}\n")
        lines.append(sub[["rank", "feature", "importance"]].to_string(index=False))
        lines.append("\n")
    lines.append("Feature importance is correlational — not causal.\n")

    lines.append("## 15. Error analysis\n")
    for src, a in analyses.items():
        ea = a["error"]
        lines.append(f"### {src}\n")
        for k, v in ea.items():
            lines.append(f"- {k}: {v}\n")

    lines.append("## 16. Model limitations\n")
    lines.append("- No walk-forward CV in this phase (deferred to Phase 9).\n")
    lines.append("- RF uses row subsampling on large Synthetic train for runtime.\n")
    lines.append("- Frequency encoding collapses rare categories.\n")
    lines.append("- Same-day average_unit_price treated as known price signal.\n")
    lines.append("- No prediction intervals.\n")

    lines.append("## 17. Phase 9 recommendations\n")
    lines.append("1. Walk-forward / rolling origin validation for stability.\n")
    lines.append("2. Residual diagnostics and horizon-specific metrics.\n")
    lines.append("3. Intermittent-demand methods for sparse UCI SKUs.\n")
    lines.append("4. Statistical comparison of ML vs baseline (Diebold-Mariano).\n")
    lines.append("5. Calibrate inventory-aware decision thresholds.\n")

    lines.append("## Charts\n")
    for p in chart_paths:
        lines.append(f"- `{p}`\n")
    lines.append("## Files\n")
    for p in files:
        lines.append(f"- `{p}`\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_ml_forecasting_pipeline(save: bool = True) -> dict:
    print("[Phase 8] Loading Phase 6 features...")
    df = load_feature_dataset()
    print(f"  shape={df.shape}")

    source_results = {}
    for src in ["UCI", "SYNTHETIC"]:
        source_results[src] = train_and_select_for_source(df, src)

    metrics = pd.concat(
        [r["all_metrics"] for r in source_results.values()], ignore_index=True
    )
    # Ensure selected flag on validation rows too
    if "selected" not in metrics.columns:
        metrics["selected"] = False
    for src, res in source_results.items():
        metrics.loc[
            (metrics["source_dataset"] == src)
            & (metrics["model"] == res["best_name"])
            & (metrics["split"] == "validation"),
            "selected",
        ] = True

    print("[Phase 8] Feature importance...")
    importance = pd.concat(
        [calculate_feature_importance(r) for r in source_results.values()],
        ignore_index=True,
    )

    print("[Phase 8] Product / store / error analysis...")
    analyses = {}
    for src, res in source_results.items():
        bp, be, hv = product_store_analysis(res)
        analyses[src] = {
            "by_product": bp,
            "by_entity": be,
            "high_value": hv,
            "error": error_analysis(res),
        }

    files = []
    chart_paths = []
    model_paths = {}
    preds = None
    report_path = None
    if save:
        os.makedirs(ML_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)

        metrics_path = os.path.join(ML_DIR, "ml_model_metrics.parquet")
        metrics.to_parquet(metrics_path, index=False)
        files.append(metrics_path)

        imp_path = os.path.join(ML_DIR, "feature_importance.parquet")
        importance.to_parquet(imp_path, index=False)
        files.append(imp_path)

        for src, a in analyses.items():
            p = os.path.join(ML_DIR, f"ml_metrics_by_product_{src.lower()}.parquet")
            a["by_product"].to_parquet(p, index=False)
            files.append(p)
            p = os.path.join(ML_DIR, f"ml_metrics_by_entity_{src.lower()}.parquet")
            a["by_entity"].to_parquet(p, index=False)
            files.append(p)
            if not a["high_value"].empty:
                p = os.path.join(ML_DIR, f"ml_high_value_{src.lower()}.parquet")
                a["high_value"].to_parquet(p, index=False)
                files.append(p)

        err_df = pd.DataFrame([a["error"] for a in analyses.values()])
        err_path = os.path.join(ML_DIR, "ml_error_analysis.parquet")
        err_df.to_parquet(err_path, index=False)
        files.append(err_path)

        preds = save_predictions(source_results)
        files.append(os.path.join(ML_DIR, "ml_predictions.parquet"))

        model_paths = save_models(source_results)
        files.extend(model_paths.values())

        meta_path = write_metadata(source_results, model_paths)
        # patch features_rows
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["features_rows"] = int(len(df))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        files.append(meta_path)

        chart_paths = create_forecast_charts(df, preds)
        chart_paths.insert(0, create_importance_chart(importance))
        files.extend(chart_paths)

        report_path = write_ml_report(
            df, source_results, metrics, importance, analyses, chart_paths,
            validation_summary="(pending validate_ml_forecasting.py)",
            files=files,
        )
        files.append(report_path)

    return {
        "features": df,
        "source_results": source_results,
        "metrics": metrics,
        "importance": importance,
        "analyses": analyses,
        "predictions": preds,
        "model_paths": model_paths,
        "chart_paths": chart_paths,
        "files": files,
        "report_path": report_path,
    }


if __name__ == "__main__":
    result = run_ml_forecasting_pipeline(save=True)
    print("\n[Phase 8] Complete.")
    for src, res in result["source_results"].items():
        best = res["best_name"]
        row = res["test_metrics_table"].query("model == @best").iloc[0]
        print(
            f"  {src}: {best} TEST WAPE={row['WAPE']:.4f} "
            f"vs baseline {row['baseline_WAPE']} "
            f"(improvement {row['wape_improvement_pct']:.2f}%)"
        )
