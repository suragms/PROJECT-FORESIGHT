"""
Phase 9.2 / 9.4 / 9.5 — Residual, zero-demand, and store/entity analysis.

Uses frozen Phase 8 TEST predictions. Does not retrain models.
Residual convention: residual = actual - prediction
Bias convention (Phase 8): bias = mean(prediction - actual)
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.phase9_common import (
    FIGURES_DIR,
    ML_PRED_PATH,
    PHASE9_DIR,
    apply_mpl_style,
    ensure_dirs,
    evaluate_extended,
)


def load_phase8_predictions() -> pd.DataFrame:
    if not os.path.exists(ML_PRED_PATH):
        raise FileNotFoundError(f"Phase 8 predictions missing: {ML_PRED_PATH}")
    df = pd.read_parquet(ML_PRED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    required = [
        "date", "source_dataset", "entity_id", "product_key",
        "actual_units_sold", "predicted_units_sold", "model",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Phase 8 prediction schema missing: {missing}")
    df["residual"] = df["actual_units_sold"] - df["predicted_units_sold"]
    df["abs_error"] = df["residual"].abs()
    return df.sort_values(["source_dataset", "entity_id", "product_key", "date"]).reset_index(drop=True)


def demand_regime(actual: pd.Series) -> pd.Series:
    """
    Data-driven regimes per source (applied after grouping):
      zero: actual == 0
      low / medium / high: tertiles of strictly positive demand
    """
    out = pd.Series(index=actual.index, dtype="object")
    zero = actual == 0
    out.loc[zero] = "zero"
    pos = actual[~zero]
    if pos.empty:
        return out.fillna("zero")
    q1, q2 = pos.quantile(1 / 3), pos.quantile(2 / 3)
    out.loc[~zero & (actual <= q1)] = "low"
    out.loc[~zero & (actual > q1) & (actual <= q2)] = "medium"
    out.loc[~zero & (actual > q2)] = "high"
    return out


def overall_residual_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, g in df.groupby("source_dataset", observed=True):
        m = evaluate_extended(
            g["actual_units_sold"].to_numpy(),
            g["predicted_units_sold"].to_numpy(),
            model_name=str(g["model"].iloc[0]),
        )
        rows.append({
            "source_dataset": src,
            "n": int(len(g)),
            **{k: m[k] for k in [
                "MAE", "RMSE", "sMAPE", "WAPE", "MAPE",
                "bias", "mean_residual", "median_residual", "residual_std",
                "overprediction_pct", "underprediction_pct",
            ]},
        })
    return pd.DataFrame(rows)


def regime_table(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for src, g in df.groupby("source_dataset", observed=True):
        g = g.copy()
        g["demand_regime"] = demand_regime(g["actual_units_sold"])
        for regime, sg in g.groupby("demand_regime"):
            m = evaluate_extended(
                sg["actual_units_sold"].to_numpy(),
                sg["predicted_units_sold"].to_numpy(),
                "lightgbm",
            )
            parts.append({
                "source_dataset": src,
                "demand_regime": regime,
                "n": int(len(sg)),
                "n_share_pct": round(100.0 * len(sg) / len(g), 2),
                "mean_actual": round(float(sg["actual_units_sold"].mean()), 4),
                "mean_prediction": round(float(sg["predicted_units_sold"].mean()), 4),
                **{k: m[k] for k in [
                    "MAE", "RMSE", "sMAPE", "WAPE", "bias",
                    "overprediction_pct", "underprediction_pct",
                ]},
                "q33_positive": round(float(g.loc[g["actual_units_sold"] > 0, "actual_units_sold"].quantile(1 / 3)), 4)
                if (g["actual_units_sold"] > 0).any() else np.nan,
                "q66_positive": round(float(g.loc[g["actual_units_sold"] > 0, "actual_units_sold"].quantile(2 / 3)), 4)
                if (g["actual_units_sold"] > 0).any() else np.nan,
            })
    return pd.DataFrame(parts)


def zero_demand_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, g in df.groupby("source_dataset", observed=True):
        z = g[g["actual_units_sold"] == 0]
        nz = g[g["actual_units_sold"] != 0]
        zm = evaluate_extended(
            z["actual_units_sold"].to_numpy() if len(z) else np.array([]),
            z["predicted_units_sold"].to_numpy() if len(z) else np.array([]),
            "lightgbm",
        )
        nzm = evaluate_extended(
            nz["actual_units_sold"].to_numpy() if len(nz) else np.array([]),
            nz["predicted_units_sold"].to_numpy() if len(nz) else np.array([]),
            "lightgbm",
        )
        rows.append({
            "source_dataset": src,
            "n_total": int(len(g)),
            "n_zero": int(len(z)),
            "n_nonzero": int(len(nz)),
            "zero_share_pct": round(100.0 * len(z) / len(g), 2) if len(g) else np.nan,
            "zero_actual_prediction_mae": zm["MAE"],
            "zero_actual_mean_prediction": round(float(z["predicted_units_sold"].mean()), 4) if len(z) else np.nan,
            "zero_actual_median_prediction": round(float(z["predicted_units_sold"].median()), 4) if len(z) else np.nan,
            "zero_actual_positive_prediction_rate": round(
                100.0 * float((z["predicted_units_sold"] > 0).mean()), 2
            ) if len(z) else np.nan,
            "zero_WAPE": zm["WAPE"],
            "zero_sMAPE": zm["sMAPE"],
            "nonzero_MAE": nzm["MAE"],
            "nonzero_RMSE": nzm["RMSE"],
            "nonzero_WAPE": nzm["WAPE"],
            "nonzero_sMAPE": nzm["sMAPE"],
            "nonzero_mean_actual": round(float(nz["actual_units_sold"].mean()), 4) if len(nz) else np.nan,
            "nonzero_mean_prediction": round(float(nz["predicted_units_sold"].mean()), 4) if len(nz) else np.nan,
            "counts_reconcile": int(len(z) + len(nz) == len(g)),
        })
    return pd.DataFrame(rows)


def residual_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Heteroscedasticity / systematic residual vs actual and vs prediction."""
    rows = []
    for src, g in df.groupby("source_dataset", observed=True):
        abs_e = g["abs_error"]
        pred_q = pd.qcut(g["predicted_units_sold"], 5, duplicates="drop")
        by_pred = g.groupby(pred_q, observed=True).agg(
            mean_residual=("residual", "mean"),
            mae=("abs_error", "mean"),
            n=("residual", "size"),
        )
        actual_q = pd.qcut(g["actual_units_sold"], 5, duplicates="drop")
        by_act = g.groupby(actual_q, observed=True).agg(
            mean_residual=("residual", "mean"),
            mae=("abs_error", "mean"),
        )
        rows.append({
            "source_dataset": src,
            "n": int(len(g)),
            "corr_abs_error_actual": round(float(g["actual_units_sold"].corr(abs_e)), 4),
            "corr_abs_error_prediction": round(float(g["predicted_units_sold"].corr(abs_e)), 4),
            "corr_residual_actual": round(float(g["actual_units_sold"].corr(g["residual"])), 4),
            "corr_residual_prediction": round(float(g["predicted_units_sold"].corr(g["residual"])), 4),
            "low_pred_mean_residual": round(float(by_pred["mean_residual"].iloc[0]), 4)
            if len(by_pred) else np.nan,
            "high_pred_mean_residual": round(float(by_pred["mean_residual"].iloc[-1]), 4)
            if len(by_pred) else np.nan,
            "low_actual_mae": round(float(by_act["mae"].iloc[0]), 4) if len(by_act) else np.nan,
            "high_actual_mae": round(float(by_act["mae"].iloc[-1]), 4) if len(by_act) else np.nan,
        })
    return pd.DataFrame(rows)


def entity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (src, ent), g in df.groupby(["source_dataset", "entity_id"], observed=True):
        m = evaluate_extended(
            g["actual_units_sold"].to_numpy(),
            g["predicted_units_sold"].to_numpy(),
            "lightgbm",
        )
        rows.append({
            "source_dataset": src,
            "entity_id": ent,
            "n": int(len(g)),
            **{k: m[k] for k in ["MAE", "RMSE", "sMAPE", "WAPE", "bias"]},
            "units_total": float(g["actual_units_sold"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["source_dataset", "WAPE"])


def create_residual_charts(df: pd.DataFrame, entity_df: pd.DataFrame, zero_df: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt

    apply_mpl_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = []
    rng = np.random.default_rng(42)

    for src in ["UCI", "SYNTHETIC"]:
        g = df[df["source_dataset"] == src]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        axes[0].hist(g["residual"], bins=80, color="#1d4ed8", alpha=0.85, density=True)
        axes[0].axvline(0, color="#111827", lw=1.2, ls="--", label="zero residual")
        axes[0].axvline(
            g["residual"].mean(), color="#dc2626", lw=1.2,
            label=f"mean={g['residual'].mean():.3f}",
        )
        axes[0].set_xlabel("Residual (actual - prediction)")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Histogram / density")
        axes[0].legend()
        axes[1].boxplot(
            g["residual"], tick_labels=["residual"], patch_artist=True,
            boxprops=dict(facecolor="#bfdbfe"), showfliers=False,
        )
        axes[1].axhline(0, color="#111827", lw=1, ls="--")
        axes[1].set_ylabel("Residual (actual - prediction)")
        axes[1].set_title("Box plot (outliers hidden)")
        fig.suptitle(f"{src} residual distribution (Phase 8 TEST)")
        fig.tight_layout()
        p = os.path.join(FIGURES_DIR, f"residual_distribution_{src.lower()}.png")
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

        sample_n = min(8000, len(g))
        idx = rng.choice(len(g), size=sample_n, replace=False)
        gs = g.iloc[idx]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        axes[0].scatter(
            gs["actual_units_sold"], gs["residual"],
            s=8, alpha=0.25, color="#1d4ed8",
        )
        axes[0].axhline(0, color="#111827", lw=1, ls="--")
        axes[0].set_xlabel("Actual units_sold")
        axes[0].set_ylabel("Residual (actual - prediction)")
        axes[0].set_title("Residual vs actual")
        axes[1].scatter(
            gs["predicted_units_sold"], gs["residual"],
            s=8, alpha=0.25, color="#059669",
        )
        axes[1].axhline(0, color="#111827", lw=1, ls="--")
        axes[1].set_xlabel("Predicted units_sold")
        axes[1].set_ylabel("Residual (actual - prediction)")
        axes[1].set_title("Residual vs prediction")
        fig.suptitle(f"{src} residual vs actual / prediction (n={sample_n:,} sample)")
        fig.tight_layout()
        p = os.path.join(FIGURES_DIR, f"residual_vs_actual_{src.lower()}.png")
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

        daily = g.groupby("date", as_index=False).agg(
            mean_residual=("residual", "mean"),
            mae=("abs_error", "mean"),
        )
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(daily["date"], daily["mean_residual"], color="#1d4ed8", lw=1.4, label="Mean residual")
        ax.axhline(0, color="#111827", lw=1, ls="--")
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean residual (actual - prediction)")
        ax.set_title(f"{src} residual over time (daily mean)")
        ax.legend()
        p = os.path.join(FIGURES_DIR, f"residual_over_time_{src.lower()}.png")
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    # Zero-demand
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, src in zip(axes, ["UCI", "SYNTHETIC"]):
        row = zero_df[zero_df["source_dataset"] == src].iloc[0]
        if int(row["n_zero"]) == 0:
            ax.bar(["Non-zero MAE"], [row["nonzero_MAE"]], color=["#1d4ed8"])
            ax.set_ylabel("MAE (units_sold)")
            ax.set_title(f"{src}: no zero-demand TEST rows")
            ax.text(
                0.5, 0.92, f"zero share=0% | n={int(row['n_total']):,}",
                transform=ax.transAxes, ha="center", fontsize=8,
            )
        else:
            labels = ["Zero-demand MAE", "Non-zero MAE"]
            vals = [row["zero_actual_prediction_mae"], row["nonzero_MAE"]]
            ax.bar(labels, vals, color=["#94a3b8", "#1d4ed8"])
            ax.set_ylabel("MAE (units_sold)")
            ax.set_title(f"{src} zero vs non-zero demand")
            ax.text(
                0.5, 0.92,
                f"zero share={row['zero_share_pct']:.1f}% | "
                f"P(pred>0|actual=0)={row['zero_actual_positive_prediction_rate']:.1f}%",
                transform=ax.transAxes, ha="center", fontsize=8,
            )
    fig.suptitle("Zero-demand vs positive-demand MAE")
    fig.tight_layout()
    p = os.path.join(FIGURES_DIR, "zero_demand_analysis.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    syn = entity_df[entity_df["source_dataset"] == "SYNTHETIC"].sort_values("WAPE")
    if not syn.empty:
        fig, ax = plt.subplots(figsize=(11, 5.2))
        colors = ["#059669" if i == 0 else "#dc2626" if i == len(syn) - 1 else "#1d4ed8"
                  for i in range(len(syn))]
        ax.bar(syn["entity_id"], syn["WAPE"], color=colors)
        ax.set_xlabel("Store / entity_id")
        ax.set_ylabel("TEST WAPE (%)")
        ax.set_title("SYNTHETIC store stability (Phase 8 TEST, recalculated)")
        ax.tick_params(axis="x", rotation=35)
        p = os.path.join(FIGURES_DIR, "store_stability.png")
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def run_residual_analysis(save: bool = True) -> dict[str, Any]:
    ensure_dirs()
    print("[Phase 9.2] Residual / zero-demand / store analysis...")
    df = load_phase8_predictions()
    overall = overall_residual_table(df)
    regimes = regime_table(df)
    zero = zero_demand_table(df)
    entities = entity_table(df)
    diagnostics = residual_diagnostics(df)
    print(overall.to_string(index=False))
    print(zero.to_string(index=False))
    print(diagnostics.to_string(index=False))
    syn = entities[entities["source_dataset"] == "SYNTHETIC"].sort_values("WAPE")
    if not syn.empty:
        print(
            f"  Best store {syn.iloc[0]['entity_id']} WAPE={syn.iloc[0]['WAPE']:.4f}; "
            f"Worst {syn.iloc[-1]['entity_id']} WAPE={syn.iloc[-1]['WAPE']:.4f}"
        )
    charts = []
    if save:
        overall.to_parquet(os.path.join(PHASE9_DIR, "residual_overall.parquet"), index=False)
        regimes.to_parquet(os.path.join(PHASE9_DIR, "residual_by_regime.parquet"), index=False)
        zero.to_parquet(os.path.join(PHASE9_DIR, "zero_demand.parquet"), index=False)
        entities.to_parquet(os.path.join(PHASE9_DIR, "entity_metrics.parquet"), index=False)
        diagnostics.to_parquet(os.path.join(PHASE9_DIR, "residual_diagnostics.parquet"), index=False)
        charts = create_residual_charts(df, entities, zero)
    return {
        "predictions": df,
        "overall": overall,
        "regimes": regimes,
        "zero_demand": zero,
        "entities": entities,
        "diagnostics": diagnostics,
        "charts": charts,
    }


if __name__ == "__main__":
    run_residual_analysis(save=True)
