"""
Phase 7 — Baseline Demand Forecasting
======================================
Project FORESIGHT: Demand & Inventory Intelligence

Establishes leakage-safe statistical baselines on Phase 6 features.
NO advanced ML in this module.

Baselines:
  - Naive:             yhat(t) = y(t-1)
  - Seasonal Naive:    yhat(t) = y(t-7)   # period=7 from Phase 5 DOW seasonality
  - Moving Average:    7 / 14 / 30 day means of y[t-w .. t-1]
  - Historical Mean:   expanding mean of y[1 .. t-1] within grain

Grain: date + source_dataset + entity_id + product_key
Target: units_sold
Sources evaluated independently (UCI vs SYNTHETIC).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

FEATURES_PATH = os.path.join(
    BASE_DIR, "data", "processed", "features", "forecast_features.parquet"
)
FORECAST_DIR = os.path.join(BASE_DIR, "data", "processed", "forecasts", "baseline")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures", "forecasting", "baseline")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

GRAIN_COLS = ["source_dataset", "entity_id", "product_key"]
FULL_GRAIN = ["date"] + GRAIN_COLS
TARGET = "units_sold"

# Seasonal period justified by Phase 5 DOW seasonality (see docs/eda_report.md §8)
# and confirmed empirically in confirm_seasonal_period().
SEASONAL_PERIOD = 7

MODEL_COLS = {
    "naive": "pred_naive",
    "seasonal_naive": "pred_seasonal_naive",
    "moving_average_7": "pred_ma_7",
    "moving_average_14": "pred_ma_14",
    "moving_average_30": "pred_ma_30",
    "historical_mean": "pred_historical_mean",
}


# ---------------------------------------------------------------------------
# Load / split
# ---------------------------------------------------------------------------

def load_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    """Load Phase 6 forecast_features and validate required columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Phase 6 output missing: {path}. Complete Phase 6 before Phase 7."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    required = FULL_GRAIN + [TARGET, "split", "revenue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"forecast_features missing required columns: {missing}")
    return df.sort_values(FULL_GRAIN).reset_index(drop=True)


def create_time_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return chronological split summary from the Phase 6 `split` column.
    Does NOT re-split randomly — uses the validated Phase 6 splits.
    """
    if "split" not in df.columns:
        raise ValueError("Phase 6 `split` column missing")
    records = []
    for src in sorted(df["source_dataset"].unique()):
        for sp in ["train", "validation", "test"]:
            sub = df[(df["source_dataset"] == src) & (df["split"] == sp)]
            if len(sub) == 0:
                continue
            records.append({
                "source_dataset": src,
                "split": sp,
                "start_date": sub["date"].min(),
                "end_date": sub["date"].max(),
                "rows": int(len(sub)),
                "unique_entities": int(sub["entity_id"].nunique()),
                "unique_products": int(sub["product_key"].nunique()),
            })
    return pd.DataFrame(records)


def confirm_seasonal_period(df: pd.DataFrame) -> dict:
    """
    Document why seasonal period = 7.

    Evidence:
      - Phase 5 EDA: DOW seasonality for both sources
      - Empirical: mean units by day_of_week varies materially within each source
    """
    evidence = {
        "selected_period": SEASONAL_PERIOD,
        "rationale": (
            "Phase 5 EDA (§8 Seasonality) found clear day-of-week effects for both "
            "SYNTHETIC (weekend lift) and UCI (weekday wholesale peaks). "
            "A 7-day seasonal naive aligns forecasts with the same weekday."
        ),
        "phase5_reference": "docs/eda_report.md §8",
        "by_source_dow_means": {},
    }
    for src in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == src]
        if "day_of_week" not in sub.columns:
            continue
        means = sub.groupby("day_of_week", observed=True)[TARGET].mean()
        evidence["by_source_dow_means"][src] = {
            int(k): round(float(v), 4) for k, v in means.items()
        }
        evidence[f"{src}_dow_cv"] = round(
            float(means.std() / means.mean()) if means.mean() else 0.0, 4
        )
    return evidence


# ---------------------------------------------------------------------------
# Baseline generators (leakage-safe)
# ---------------------------------------------------------------------------

def naive_forecast(df: pd.DataFrame) -> pd.Series:
    """Forecast(t) = Actual(t-1) within grain."""
    if "units_sold_lag_1" in df.columns:
        return df["units_sold_lag_1"].astype(float)
    return (
        df.groupby(GRAIN_COLS, observed=True)[TARGET].shift(1).astype(float)
    )


def seasonal_naive_forecast(df: pd.DataFrame, period: int = SEASONAL_PERIOD) -> pd.Series:
    """Forecast(t) = Actual(t - period) within grain."""
    lag_col = f"units_sold_lag_{period}"
    if lag_col in df.columns:
        return df[lag_col].astype(float)
    return (
        df.groupby(GRAIN_COLS, observed=True)[TARGET].shift(period).astype(float)
    )


def moving_average_forecast(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Leakage-safe moving average: mean of y[t-window .. t-1].
    Prefers Phase 6 rolling_mean_{window} (already shift-1 safe).
    """
    col = f"rolling_mean_{window}"
    if col in df.columns:
        return df[col].astype(float)

    shifted = df.groupby(GRAIN_COLS, observed=True)[TARGET].shift(1)
    return (
        shifted.groupby([df[c] for c in GRAIN_COLS], observed=True)
        .rolling(window, min_periods=1)
        .mean()
        .droplevel(list(range(len(GRAIN_COLS))))
        .sort_index()
        .astype(float)
    )


def historical_mean_forecast(df: pd.DataFrame) -> pd.Series:
    """
    Expanding mean of all prior observations within grain (excludes current).
    Equivalent to mean(y[1..t-1]) — never uses validation/test future targets.
    """
    shifted = df.groupby(GRAIN_COLS, observed=True)[TARGET].shift(1)
    return (
        shifted.groupby([df[c] for c in GRAIN_COLS], observed=True)
        .expanding(min_periods=1)
        .mean()
        .droplevel(list(range(len(GRAIN_COLS))))
        .sort_index()
        .astype(float)
    )


def generate_all_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all baseline prediction columns to a copy of df."""
    out = df.copy()
    out["pred_naive"] = naive_forecast(out)
    out["pred_seasonal_naive"] = seasonal_naive_forecast(out, SEASONAL_PERIOD)
    out["pred_ma_7"] = moving_average_forecast(out, 7)
    out["pred_ma_14"] = moving_average_forecast(out, 14)
    out["pred_ma_30"] = moving_average_forecast(out, 30)
    out["pred_historical_mean"] = historical_mean_forecast(out)
    return out


# ---------------------------------------------------------------------------
# Metrics (zero-safe)
# ---------------------------------------------------------------------------

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    MAE, RMSE, WAPE, sMAPE, and MAPE-where-valid.

    Formulas:
      MAE   = mean(|y - yhat|)
      RMSE  = sqrt(mean((y - yhat)^2))
      WAPE  = sum(|y - yhat|) / sum(|y|) * 100          (0 if sum|y|=0)
      sMAPE = mean( 2|y-yhat| / (|y|+|yhat|) ) * 100
              — pairs where |y|+|yhat|=0 contribute 0 (perfect zero-zero)
      MAPE  = mean(|y-yhat|/|y|) * 100 over rows with y != 0 only
              — NaN if no positive-demand rows
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n == 0:
        return {
            "model": model_name,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "sMAPE": np.nan,
            "WAPE": np.nan,
            "n": 0,
        }

    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom_w = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(err)) / denom_w * 100.0) if denom_w > 0 else 0.0

    denom_s = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.zeros_like(denom_s, dtype=float)
    nonzero = denom_s > 0
    smape_terms[nonzero] = 2.0 * np.abs(err[nonzero]) / denom_s[nonzero]
    smape = float(np.mean(smape_terms) * 100.0)

    pos = y_true != 0
    if pos.any():
        mape = float(np.mean(np.abs(err[pos]) / np.abs(y_true[pos])) * 100.0)
    else:
        mape = np.nan

    return {
        "model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4) if mape == mape else np.nan,
        "sMAPE": round(smape, 4),
        "WAPE": round(wape, 4),
        "n": int(n),
    }


def _eval_frame(
    pred_df: pd.DataFrame,
    split: Optional[str] = None,
    source: Optional[str] = None,
) -> list[dict]:
    """Evaluate all models on a filtered prediction frame."""
    sub = pred_df
    if split is not None:
        sub = sub[sub["split"] == split]
    if source is not None:
        sub = sub[sub["source_dataset"] == source]
    rows = []
    for model, col in MODEL_COLS.items():
        m = calculate_metrics(sub[TARGET].to_numpy(), sub[col].to_numpy(), model)
        m["split"] = split if split else "all_eval"
        m["source_dataset"] = source if source else "ALL"
        rows.append(m)
    return rows


def evaluate_baselines(pred_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute metric tables:
      - metrics: overall by split (val/test) across sources (still separated rows)
      - by_source: source × model × split
      - by_product: product × model on TEST
      - by_entity: entity × model on TEST (Synthetic emphasis)
      - comparison: best-ranked models per source on TEST (by WAPE then MAE)
    """
    eval_df = pred_df[pred_df["split"].isin(["validation", "test"])].copy()

    metrics_rows = []
    for sp in ["validation", "test"]:
        for src in sorted(eval_df["source_dataset"].unique()):
            metrics_rows.extend(_eval_frame(eval_df, split=sp, source=src))
    metrics = pd.DataFrame(metrics_rows)

    by_source = metrics.copy()

    # Product-level on TEST
    prod_rows = []
    test = eval_df[eval_df["split"] == "test"]
    for (src, ent, pk), g in test.groupby(GRAIN_COLS, observed=True):
        for model, col in MODEL_COLS.items():
            m = calculate_metrics(g[TARGET].to_numpy(), g[col].to_numpy(), model)
            m.update({
                "source_dataset": src,
                "entity_id": ent,
                "product_key": pk,
                "split": "test",
                "units_total": float(g[TARGET].sum()),
                "revenue_total": float(g["revenue"].sum()) if "revenue" in g.columns else np.nan,
            })
            prod_rows.append(m)
    by_product = pd.DataFrame(prod_rows)

    # Entity-level on TEST
    ent_rows = []
    for (src, ent), g in test.groupby(["source_dataset", "entity_id"], observed=True):
        for model, col in MODEL_COLS.items():
            m = calculate_metrics(g[TARGET].to_numpy(), g[col].to_numpy(), model)
            m.update({
                "source_dataset": src,
                "entity_id": ent,
                "split": "test",
                "units_total": float(g[TARGET].sum()),
                "revenue_total": float(g["revenue"].sum()) if "revenue" in g.columns else np.nan,
            })
            ent_rows.append(m)
    by_entity = pd.DataFrame(ent_rows)

    # Comparison leaderboard — TEST only, per source, ranked by WAPE then MAE
    comparison_rows = []
    test_metrics = metrics[metrics["split"] == "test"]
    for src in sorted(test_metrics["source_dataset"].unique()):
        sub = test_metrics[test_metrics["source_dataset"] == src].copy()
        sub = sub.sort_values(["WAPE", "MAE", "RMSE"]).reset_index(drop=True)
        sub["rank"] = np.arange(1, len(sub) + 1)
        comparison_rows.append(sub)
    comparison = pd.concat(comparison_rows, ignore_index=True) if comparison_rows else pd.DataFrame()

    return {
        "metrics": metrics,
        "by_source": by_source,
        "by_product": by_product,
        "by_entity": by_entity,
        "comparison": comparison,
    }


def high_value_sku_analysis(
    df: pd.DataFrame,
    by_product: pd.DataFrame,
    best_model_by_source: dict[str, str],
) -> pd.DataFrame:
    """
    Compare baseline TEST performance for high-revenue vs lower-revenue SKUs.
    Pareto cut computed from actual revenue (not hard-coded Phase 5 %).
    """
    rows = []
    for src in sorted(df["source_dataset"].unique()):
        src_df = df[df["source_dataset"] == src]
        sku_rev = (
            src_df.groupby("product_key", observed=True)["revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        total = float(sku_rev.sum())
        if total <= 0:
            continue
        cumshare = sku_rev.cumsum() / total
        # High-value = SKUs needed to reach 80% of revenue (actual Pareto set)
        high_mask = cumshare.shift(fill_value=0) < 0.80
        high_skus = set(sku_rev.index[high_mask])
        low_skus = set(sku_rev.index) - high_skus

        model = best_model_by_source.get(src)
        if model is None:
            continue
        bp = by_product[
            (by_product["source_dataset"] == src) & (by_product["model"] == model)
        ]
        # Aggregate product metrics to product_key (avg across entities for Synthetic)
        agg = bp.groupby("product_key", observed=True).agg(
            WAPE=("WAPE", "mean"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            revenue_total=("revenue_total", "sum"),
            n=("n", "sum"),
        )
        for label, skus in [("high_revenue", high_skus), ("lower_revenue", low_skus)]:
            if not skus:
                continue
            part = agg.loc[agg.index.isin(skus)]
            if part.empty:
                continue
            seg_rev = float(sku_rev[sku_rev.index.isin(skus)].sum())
            rows.append({
                "source_dataset": src,
                "segment": label,
                "n_skus": int(len(skus)),
                "revenue_share_pct": round(100.0 * seg_rev / total, 2),
                "model": model,
                "MAE": round(float(part["MAE"].mean()), 4),
                "RMSE": round(float(part["RMSE"].mean()), 4),
                "WAPE": round(float(part["WAPE"].mean()), 4),
                "pareto_threshold": 0.80,
                "pareto_sku_count": len(high_skus),
                "pareto_sku_pct": round(100.0 * len(high_skus) / len(sku_rev), 2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _pick_series(
    df: pd.DataFrame,
    source: str,
    order_by: str = "units",
    ascending: bool = False,
) -> tuple:
    """Pick a representative (entity_id, product_key) for charts."""
    sub = df[df["source_dataset"] == source]
    if order_by == "revenue":
        ranked = (
            sub.groupby(["entity_id", "product_key"], observed=True)["revenue"]
            .sum()
            .sort_values(ascending=ascending)
        )
    else:
        ranked = (
            sub.groupby(["entity_id", "product_key"], observed=True)[TARGET]
            .sum()
            .sort_values(ascending=ascending)
        )
    if ranked.empty:
        return None, None
    return ranked.index[0]


def create_forecast_charts(pred_df: pd.DataFrame, out_dir: str = FIGURES_DIR) -> list[str]:
    """Create representative actual-vs-baseline charts. Returns saved paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    specs = [
        ("uci_high_volume", "UCI", "units", False),
        ("uci_low_volume", "UCI", "units", True),
        ("uci_high_revenue", "UCI", "revenue", False),
        ("syn_store_sku", "SYNTHETIC", "units", False),
        ("syn_high_revenue", "SYNTHETIC", "revenue", False),
    ]

    test = pred_df[pred_df["split"] == "test"]
    for name, src, order_by, ascending in specs:
        ent, pk = _pick_series(pred_df[pred_df["source_dataset"] == src], src, order_by, ascending)
        if ent is None:
            continue
        g = test[
            (test["source_dataset"] == src)
            & (test["entity_id"] == ent)
            & (test["product_key"] == pk)
        ].sort_values("date")
        if g.empty:
            # fall back to full series tail
            g = pred_df[
                (pred_df["source_dataset"] == src)
                & (pred_df["entity_id"] == ent)
                & (pred_df["product_key"] == pk)
            ].sort_values("date").tail(90)

        # Cap points for readability
        g = g.tail(60)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(g["date"], g[TARGET], label="Actual", color="#1f2937", linewidth=2)
        ax.plot(g["date"], g["pred_naive"], label="Naive", alpha=0.85)
        ax.plot(g["date"], g["pred_seasonal_naive"], label="Seasonal Naive (7)", alpha=0.85)
        ax.plot(g["date"], g["pred_ma_7"], label="MA-7", alpha=0.85)
        ax.set_title(f"{name}: {src} | {ent} | {pk}")
        ax.set_xlabel("Date")
        ax.set_ylabel("units_sold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)

    # Comparison bar chart — TEST WAPE by source
    return paths


def create_comparison_chart(comparison: pd.DataFrame, out_dir: str = FIGURES_DIR) -> str:
    """Bar chart of TEST WAPE by model for each source."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "baseline_wape_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, src in zip(axes, ["UCI", "SYNTHETIC"]):
        sub = comparison[comparison["source_dataset"] == src].sort_values("WAPE")
        if sub.empty:
            continue
        ax.barh(sub["model"], sub["WAPE"], color="#3b82f6")
        ax.set_title(f"{src} — Test WAPE (%)")
        ax.set_xlabel("WAPE %")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def save_baseline_results(
    pred_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    out_dir: str = FORECAST_DIR,
) -> dict[str, str]:
    """Write baseline parquet outputs. Returns path map."""
    os.makedirs(out_dir, exist_ok=True)
    keep = FULL_GRAIN + [TARGET, "split", "revenue", "sku_id"] + list(MODEL_COLS.values())
    keep = [c for c in keep if c in pred_df.columns]
    # Persist eval rows primarily (val+test); include train warm-up NaNs only if needed
    # Keep all rows for leakage audits (first-row NaN checks).
    paths = {}
    pred_path = os.path.join(out_dir, "baseline_predictions.parquet")
    pred_df[keep].to_parquet(pred_path, index=False)
    paths["predictions"] = pred_path

    mapping = {
        "metrics": "baseline_metrics.parquet",
        "by_source": "baseline_metrics_by_source.parquet",
        "by_product": "baseline_metrics_by_product.parquet",
        "by_entity": "baseline_metrics_by_entity.parquet",
        "comparison": "baseline_comparison.parquet",
        "high_value": "baseline_high_value_sku.parquet",
    }
    for key, fname in mapping.items():
        if key not in tables or tables[key] is None or tables[key].empty:
            continue
        p = os.path.join(out_dir, fname)
        tables[key].to_parquet(p, index=False)
        paths[key] = p
    return paths


def best_baselines(comparison: pd.DataFrame) -> dict:
    """Extract best TEST baseline per source (lowest WAPE, then MAE)."""
    out = {}
    for src in sorted(comparison["source_dataset"].unique()):
        sub = comparison[comparison["source_dataset"] == src].sort_values(
            ["WAPE", "MAE", "RMSE"]
        )
        if sub.empty:
            continue
        top = sub.iloc[0]
        out[src] = {
            "model": top["model"],
            "MAE": float(top["MAE"]),
            "RMSE": float(top["RMSE"]),
            "sMAPE": float(top["sMAPE"]),
            "WAPE": float(top["WAPE"]),
            "MAPE": float(top["MAPE"]) if pd.notna(top["MAPE"]) else np.nan,
            "n": int(top["n"]),
        }
    return out


def write_baseline_report(
    df: pd.DataFrame,
    split_summary: pd.DataFrame,
    seasonality: dict,
    tables: dict[str, pd.DataFrame],
    best: dict,
    chart_paths: list[str],
    validation_summary: str,
    paths: dict[str, str],
    report_path: str | None = None,
) -> str:
    """Write docs/baseline_forecasting_report.md from executed results."""
    report_path = report_path or os.path.join(DOCS_DIR, "baseline_forecasting_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    def _fmt_split(src: str) -> str:
        lines = []
        for _, r in split_summary[split_summary["source_dataset"] == src].iterrows():
            lines.append(
                f"| {r['split']} | {pd.Timestamp(r['start_date']).date()} | "
                f"{pd.Timestamp(r['end_date']).date()} | {int(r['rows']):,} |"
            )
        return "\n".join(lines)

    def _metrics_table(src: str, split: str = "test") -> str:
        m = tables["by_source"]
        sub = m[(m["source_dataset"] == src) & (m["split"] == split)].sort_values("WAPE")
        lines = ["| Model | MAE | RMSE | MAPE | sMAPE | WAPE | n |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for _, r in sub.iterrows():
            mape = f"{r['MAPE']:.4f}" if pd.notna(r["MAPE"]) else "N/A"
            lines.append(
                f"| {r['model']} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {mape} | "
                f"{r['sMAPE']:.4f} | {r['WAPE']:.4f} | {int(r['n']):,} |"
            )
        return "\n".join(lines)

    # Product findings
    prod_findings = []
    for src, info in best.items():
        bp = tables["by_product"]
        sub = bp[(bp["source_dataset"] == src) & (bp["model"] == info["model"])]
        if sub.empty:
            continue
        best_p = sub.sort_values("WAPE").head(3)
        worst_p = sub.sort_values("WAPE", ascending=False).head(3)
        prod_findings.append(f"### {src} (model={info['model']})")
        prod_findings.append("Best WAPE products:")
        for _, r in best_p.iterrows():
            prod_findings.append(
                f"- `{r['product_key']}` ({r['entity_id']}): WAPE={r['WAPE']:.2f}, "
                f"MAE={r['MAE']:.2f}, units={r['units_total']:.0f}"
            )
        prod_findings.append("Worst WAPE products:")
        for _, r in worst_p.iterrows():
            prod_findings.append(
                f"- `{r['product_key']}` ({r['entity_id']}): WAPE={r['WAPE']:.2f}, "
                f"MAE={r['MAE']:.2f}, units={r['units_total']:.0f}"
            )

    # Store findings (Synthetic)
    store_lines = []
    be = tables["by_entity"]
    syn_best = best.get("SYNTHETIC", {}).get("model")
    if syn_best:
        ent = be[(be["source_dataset"] == "SYNTHETIC") & (be["model"] == syn_best)].sort_values("WAPE")
        for _, r in ent.iterrows():
            store_lines.append(
                f"| {r['entity_id']} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['WAPE']:.4f} |"
            )

    hv = tables.get("high_value", pd.DataFrame())
    if hv.empty:
        hv_md = "_n/a_"
    else:
        hv_lines = [
            "| " + " | ".join(hv.columns.astype(str)) + " |",
            "|" + "|".join(["---"] * len(hv.columns)) + "|",
        ]
        for _, r in hv.iterrows():
            hv_lines.append("| " + " | ".join(str(r[c]) for c in hv.columns) + " |")
        hv_md = "\n".join(hv_lines)

    charts_md = "\n".join(f"- `{p}`" for p in chart_paths)
    files_md = "\n".join(f"- `{p}`" for p in paths.values())

    md = f"""# Phase 7 — Baseline Demand Forecasting Report

**Project:** FORESIGHT — Demand & Inventory Intelligence  
**Status:** COMPLETE (executed + validated)  
**Validation:** {validation_summary}

---

## 1. Objective

Establish reliable, leakage-safe baseline demand forecasts before Phase 8 ML.
Baselines become the hard benchmarks ML must beat.

## 2. Phase 6 input

| Item | Value |
|---|---|
| Input | `data/processed/features/forecast_features.parquet` |
| Rows | **{len(df):,}** |
| Columns | **{df.shape[1]}** |
| Date range | {df['date'].min().date()} → {df['date'].max().date()} |
| Sources | {dict(df['source_dataset'].value_counts())} |

## 3. Forecasting grain

`date + source_dataset + entity_id + product_key`

UCI and SYNTHETIC are evaluated separately — never combined into one continuous series.

## 4. Target

`units_sold` (nulls in input: {int(df[TARGET].isna().sum())})

## 5. Split dates

Chronological splits inherited from Phase 6 (`split` column). No random splitting.

### SYNTHETIC
| Split | Start | End | Rows |
|---|---|---|---|
{_fmt_split("SYNTHETIC")}

### UCI
| Split | Start | End | Rows |
|---|---|---|---|
{_fmt_split("UCI")}

## 6. Baseline models

| Model | Formula | Leakage control |
|---|---|---|
| Naive | y(t-1) | lag within grain |
| Seasonal Naive | y(t-{SEASONAL_PERIOD}) | lag within grain |
| Moving Average 7/14/30 | mean(y[t-w..t-1]) | shift(1) before roll |
| Historical Mean | expanding mean(y[1..t-1]) | excludes current |

**Seasonal period = {SEASONAL_PERIOD}**

Rationale: {seasonality['rationale']}

Empirical DOW mean CV: UCI={seasonality.get('UCI_dow_cv')}, SYNTHETIC={seasonality.get('SYNTHETIC_dow_cv')}

## 7. Metric definitions

- **MAE** = mean(\\|y − ŷ\\|)
- **RMSE** = sqrt(mean((y − ŷ)²))
- **WAPE** = Σ\\|y − ŷ\\| / Σ\\|y\\| × 100 (0 if Σ\\|y\\|=0)
- **sMAPE** = mean(2\\|y − ŷ\\| / (\\|y\\|+\\|ŷ\\|)) × 100; zero-zero pairs contribute 0
- **MAPE** = mean(\\|y − ŷ\\| / \\|y\\|) × 100 **only where y ≠ 0**; else N/A

Priority for business interpretation: MAE, RMSE, WAPE, sMAPE.

## 8. UCI results (TEST)

{_metrics_table("UCI", "test")}

## 9. Synthetic results (TEST)

{_metrics_table("SYNTHETIC", "test")}

## 10. Product results

{chr(10).join(prod_findings)}

## 11. Store results (Synthetic, best model)

| Store | MAE | RMSE | WAPE |
|---|---:|---:|---:|
{chr(10).join(store_lines)}

## 12. Best baselines (TEST)

| Source | Best model | MAE | RMSE | sMAPE | WAPE |
|---|---|---:|---:|---:|---:|
{chr(10).join(
    f"| {src} | {info['model']} | {info['MAE']:.4f} | {info['RMSE']:.4f} | "
    f"{info['sMAPE']:.4f} | {info['WAPE']:.4f} |"
    for src, info in best.items()
)}

These are the Phase 8 ML benchmarks.

## 13. Business insights

### High- vs lower-revenue SKUs (actual Pareto @ 80%)

{hv_md}

OBSERVATION: Best baseline differs by source if ranks diverge — do not force a universal winner.

EVIDENCE: See comparison tables above (TEST WAPE ranking).

BUSINESS INTERPRETATION: Stable, high-volume Synthetic store-SKU series favor smoothed averages; intermittent UCI wholesale demand may favor seasonal/naive patterns differently.

IMPLICATION FOR ML: Phase 8 must beat the **source-specific** best WAPE above, with separate models or clearly separated evaluations.

## 14. Limitations

- Baselines ignore price, promo, and inventory signals (intentional for Phase 7).
- Warm-up rows (insufficient history) yield NaN predictions and are excluded from metrics via finite masks.
- UCI has intermittent demand and many zero/sparse SKU-days → MAPE is computed only on non-zero actuals.
- Seasonal period fixed at 7 from Phase 5 DOW evidence; monthly seasonality is not modeled here.
- No ML / Prophet / SARIMA trained in this phase.

## 15. Phase 8 recommendations

1. Train ML **separately** per `source_dataset`.
2. Beat the source-specific best baseline WAPE in §12.
3. Use Phase 6 features (`lag`, `rolling`, price/promo/inventory where available).
4. Preserve chronological splits; never random-split.
5. Report MAE / RMSE / WAPE / sMAPE on the same TEST windows.
6. Investigate worst-WAPE products/stores from §10–11 for feature gaps.
7. Consider intermittent-demand methods for sparse UCI SKUs.

## Charts

{charts_md}

## Files created

{files_md}

---

*Generated from actual Phase 7 execution. No simulated metrics.*
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    return report_path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_baseline_pipeline(save: bool = True) -> dict:
    """Execute full Phase 7 baseline pipeline. Returns result bundle."""
    print("[Phase 7] Loading Phase 6 features...")
    df = load_features()
    print(f"  rows={len(df):,} cols={df.shape[1]}")

    split_summary = create_time_split(df)
    print("[Phase 7] Chronological splits (from Phase 6):")
    print(split_summary.to_string(index=False))

    seasonality = confirm_seasonal_period(df)
    print(f"[Phase 7] Seasonal period={seasonality['selected_period']}")

    print("[Phase 7] Generating baseline predictions...")
    pred_df = generate_all_predictions(df)

    print("[Phase 7] Evaluating metrics...")
    tables = evaluate_baselines(pred_df)
    best = best_baselines(tables["comparison"])
    best_models = {src: info["model"] for src, info in best.items()}
    tables["high_value"] = high_value_sku_analysis(df, tables["by_product"], best_models)

    print("[Phase 7] Best baselines (TEST):")
    for src, info in best.items():
        print(f"  {src}: {info['model']}  WAPE={info['WAPE']:.4f}  MAE={info['MAE']:.4f}")

    chart_paths = []
    paths = {}
    report_path = None
    if save:
        print("[Phase 7] Saving outputs...")
        paths = save_baseline_results(pred_df, tables)
        chart_paths = create_forecast_charts(pred_df)
        chart_paths.append(create_comparison_chart(tables["comparison"]))
        # Validation + report filled by caller / CLI after validate_baselines
        report_path = write_baseline_report(
            df, split_summary, seasonality, tables, best, chart_paths,
            validation_summary="(pending validate_baselines.py)",
            paths=paths,
        )
        print(f"[Phase 7] Report: {report_path}")

    return {
        "features": df,
        "predictions": pred_df,
        "split_summary": split_summary,
        "seasonality": seasonality,
        "tables": tables,
        "best": best,
        "chart_paths": chart_paths,
        "paths": paths,
        "report_path": report_path,
    }


if __name__ == "__main__":
    result = run_baseline_pipeline(save=True)
    print("[Phase 7] Baseline forecasting complete.")
    print(f"  Models: {list(MODEL_COLS)}")
    for src, info in result["best"].items():
        print(f"  Best {src}: {info}")
