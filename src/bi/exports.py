"""Build compact Power BI-ready exports from existing FORESIGHT outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.bi.kpis import executive_kpi_row, load_risk
from src.bi.quality import freshness, quality_scorecard
from src.bi.recommendations import build_recommendations, risk_matrix_cell
from src.bi.rules import (
    DEMAND_HIGH_QUANTILE,
    INVENTORY_HIGH_QUANTILE,
    classify_growth,
)
from src.config import (
    FINAL_FORECASTS_PATH,
    INVENTORY_RISK_PATH,
    OUTPUTS_BI_DIR,
    OUTPUTS_MONITORING_DIR,
    PHASE11_META_PATH,
    REGISTRY_PATH,
)
from src.forecasting.registry import load_registry, verify_hash
from src.production.simulation import EXPECTED_SYN_HASH, EXPECTED_UCI_HASH

EXPORT_FILES = (
    "executive_kpis.parquet",
    "product_demand.parquet",
    "forecast_performance.parquet",
    "inventory_risk.parquet",
    "recommendations.parquet",
    "system_health.parquet",
)

OPTIONAL_FILES = ("seasonality.parquet",)

REQUIRED_COLUMNS = {
    "executive_kpis.parquet": (
        "layer",
        "decision_support_only",
        "forecast_mae",
        "forecast_rmse",
        "forecast_wape",
        "forecast_bias",
        "inventory_n_rows",
        "inventory_stockout_critical_high",
        "inventory_extract_note",
    ),
    "product_demand.parquet": (
        "sku_id",
        "store_id",
        "total_recent_units",
        "demand_share",
        "growth_class",
        "stockout_risk_level",
        "demand_rank_label",
        "extract_note",
    ),
    "forecast_performance.parquet": (
        "forecast_date",
        "source_dataset",
        "horizon",
        "actual",
        "forecast",
        "error",
        "absolute_error",
        "p10",
        "p90",
        "grain",
    ),
    "inventory_risk.parquet": (
        "sku_id",
        "store_id",
        "stockout_risk_level",
        "overstock_risk_level",
        "reorder_triggered",
        "risk_matrix_cell",
        "extract_note",
    ),
    "recommendations.parquet": (
        "sku_id",
        "recommended_review",
        "evidence",
        "reason",
        "confidence_limitation",
        "autonomous_decision",
    ),
    "system_health.parquet": (
        "uci_h1_hash",
        "synthetic_h1_hash",
        "uci_hash_matches_phase12",
        "synthetic_hash_matches_phase12",
        "live_data",
        "monitoring_snapshot",
    ),
}


def _json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write(df: pd.DataFrame, name: str) -> Path:
    OUTPUTS_BI_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_BI_DIR / name
    df.to_parquet(path, index=False)
    return path


def forecast_performance_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = _json(OUTPUTS_MONITORING_DIR / "accuracy_monitoring_report.json") or {}
    rows = acc.get("by_dataset_horizon") or []
    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics["grain"] = "dataset_horizon"
        metrics["source"] = "outputs/monitoring/accuracy_monitoring_report.json"
        metrics["note"] = "Held-out TEST actuals in final_predictions.parquet — not future unknown actuals."

    fc = pd.read_parquet(
        FINAL_FORECASTS_PATH,
        columns=["forecast_date", "source_dataset", "horizon", "actual", "prediction", "lower_bound", "upper_bound"],
    )
    fc["forecast_date"] = pd.to_datetime(fc["forecast_date"])
    daily = fc.groupby(["source_dataset", "horizon", "forecast_date"], as_index=False).agg(
        actual=("actual", "mean"),
        forecast=("prediction", "mean"),
        p10=("lower_bound", "mean"),
        p90=("upper_bound", "mean"),
        n=("prediction", "size"),
    )
    daily["error"] = daily["forecast"] - daily["actual"]
    daily["absolute_error"] = daily["error"].abs()
    daily["grain"] = "daily"
    daily["concept_actual"] = "Actual (held-out TEST)"
    daily["concept_forecast"] = "Forecast (model point prediction)"
    daily["concept_p10"] = "P10 interval companion — not an observation"
    daily["concept_p90"] = "P90 interval companion — not an observation"
    daily["note"] = "Do not treat forecast or P10/P90 as actuals. Metrics are not computed on unknown future dates."
    return metrics, daily


def seasonality_table(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    d = daily.copy()
    d["month"] = pd.to_datetime(d["forecast_date"]).dt.month
    d["dow"] = pd.to_datetime(d["forecast_date"]).dt.dayofweek
    monthly = d.groupby(["source_dataset", "horizon", "month"], as_index=False).agg(
        mean_actual=("actual", "mean"), n=("n", "sum")
    )
    monthly["pattern"] = "monthly"
    monthly["period"] = monthly["month"]
    weekly = d.groupby(["source_dataset", "horizon", "dow"], as_index=False).agg(
        mean_actual=("actual", "mean"), n=("n", "sum")
    )
    weekly["pattern"] = "weekly"
    weekly["period"] = weekly["dow"]
    out = pd.concat([monthly.drop(columns=["month"]), weekly.drop(columns=["dow"])], ignore_index=True)
    notes = []
    for ds, g in monthly.groupby("source_dataset"):
        mu = g["mean_actual"].mean()
        sd = g["mean_actual"].std(ddof=0)
        cv = float(sd / mu) if mu else None
        weak = cv is not None and cv < 0.10
        peak = int(g.loc[g["mean_actual"].idxmax(), "month"]) if len(g) else None
        low = int(g.loc[g["mean_actual"].idxmin(), "month"]) if len(g) else None
        notes.append({
            "source_dataset": ds,
            "horizon": None,
            "pattern": "monthly_summary",
            "period": None,
            "mean_actual": mu,
            "n": int(g["n"].sum()),
            "seasonality_cv": None if cv is None else round(cv, 4),
            "weak_or_uncertain": weak,
            "seasonal_peak_month": peak,
            "seasonal_low_month": low,
            "note": "Computed from held-out TEST actuals. Weak if monthly CV < 0.10. Not a causal seasonality model.",
        })
    return pd.concat([out, pd.DataFrame(notes)], ignore_index=True)


def product_forecast_h1() -> pd.DataFrame:
    h1 = pd.read_parquet(
        FINAL_FORECASTS_PATH,
        columns=["forecast_date", "source_dataset", "entity_id", "product_key", "horizon",
                 "actual", "prediction", "lower_bound", "upper_bound"],
        filters=[("horizon", "==", 1)],
    )
    h1["forecast_date"] = pd.to_datetime(h1["forecast_date"])
    mid = h1.groupby(["source_dataset", "entity_id", "product_key"])["forecast_date"].transform("median")
    h1["half"] = np.where(h1["forecast_date"] <= mid, "hist", "recent")
    agg = h1.groupby(["source_dataset", "entity_id", "product_key", "half"], as_index=False).agg(
        demand=("actual", "mean"), n=("actual", "size"), forecast=("prediction", "mean"),
        p10=("lower_bound", "mean"), p90=("upper_bound", "mean"),
    )
    hist = agg[agg["half"] == "hist"].rename(columns={"demand": "historical_demand", "n": "n_hist"})
    recent = agg[agg["half"] == "recent"].rename(columns={"demand": "recent_demand", "n": "n_recent"})
    keys = ["source_dataset", "entity_id", "product_key"]
    merged = pd.merge(
        hist[keys + ["historical_demand", "n_hist", "forecast", "p10", "p90"]],
        recent[keys + ["recent_demand", "n_recent"]],
        on=keys,
        how="outer",
    )
    denom = merged["historical_demand"].abs().clip(lower=1e-9)
    rate = (merged["recent_demand"] - merged["historical_demand"]) / denom
    merged["growth_rate"] = rate
    merged["growth_class"] = [
        classify_growth(
            None if pd.isna(r) else float(r),
            int(nh) if pd.notna(nh) else 0,
            int(nr) if pd.notna(nr) else 0,
        )
        for r, nh, nr in zip(merged["growth_rate"], merged["n_hist"], merged["n_recent"])
    ]
    width = (merged["p90"] - merged["p10"]).abs() / merged["forecast"].abs().clip(lower=1e-9)
    merged["relative_interval_width"] = width
    merged["horizon"] = 1
    merged["growth_rule"] = "TEST actuals split at median forecast_date; |rate|<0.05 Stable; n<10 Insufficient Evidence"
    merged["low_demand_label"] = "LOW DEMAND"
    return merged


def inventory_bi(risk: pd.DataFrame, product_fc: pd.DataFrame) -> pd.DataFrame:
    out = risk.copy()
    d_cut = out["avg_daily_demand"].quantile(DEMAND_HIGH_QUANTILE) if "avg_daily_demand" in out.columns else None
    i_cut = out["ending_inventory"].quantile(INVENTORY_HIGH_QUANTILE) if "ending_inventory" in out.columns else None
    out["demand_high"] = out["avg_daily_demand"] > d_cut if d_cut is not None else False
    out["inventory_high"] = out["ending_inventory"] > i_cut if i_cut is not None else False
    out["risk_matrix_cell"] = [
        risk_matrix_cell(bool(d), bool(i)) for d, i in zip(out["demand_high"], out["inventory_high"])
    ]
    out["extract_note"] = "1000-row reference extract"
    out["risk_matrix_rule"] = (
        "demand_high if avg_daily_demand > extract median; "
        "inventory_high if ending_inventory > extract median (strict). "
        "Median ending_inventory is 0 on this extract; strict '>' treats zero stock as Low."
    )
    syn = product_fc[product_fc["source_dataset"] == "SYNTHETIC"].copy()
    syn["sku_id"] = syn["product_key"].astype(str).str.replace("^SYN_", "", regex=True)
    syn_small = syn.rename(columns={"forecast": "forecast_mean_h1", "entity_id": "store_id"})
    join_cols = ["sku_id", "store_id"]
    keep = join_cols + ["forecast_mean_h1", "growth_class", "growth_rate", "p10", "p90", "relative_interval_width"]
    keep = [c for c in keep if c in syn_small.columns]
    out = out.merge(syn_small[keep], on=["sku_id", "store_id"], how="left")
    for col in ("forecast_mean_h1", "growth_class", "growth_rate", "p10", "p90", "relative_interval_width"):
        if col not in out.columns:
            out[col] = np.nan
    out["forecast_mean_h1"] = out["forecast_mean_h1"].where(out["forecast_mean_h1"].notna(), other=None)
    out["growth_class"] = out["growth_class"].fillna("Insufficient Evidence")
    share = out["total_recent_units"] / max(float(out["total_recent_units"].sum()), 1e-9) if "total_recent_units" in out.columns else np.nan
    out["demand_share"] = share
    out["demand_rank_label"] = np.where(
        out["total_recent_revenue"].rank(ascending=False, method="min") <= 10,
        "TOP_REVENUE",
        np.where(out["total_recent_revenue"].rank(ascending=True, method="min") <= 10, "LOW DEMAND", "MID"),
    )
    return out


def system_health_row() -> dict[str, Any]:
    recs = load_registry()
    hashes = {r["model_id"]: verify_hash(r) for r in recs}
    fresh = freshness()
    q = quality_scorecard()
    mon = _json(OUTPUTS_MONITORING_DIR / "monitoring_summary.json") or {}
    return {
        "api_health": "see GET /health (not called in export)",
        "readiness": "see GET /ready (not called in export)",
        "uci_h1_hash": hashes.get("uci_h1_phase8_lightgbm"),
        "synthetic_h1_hash": hashes.get("synthetic_h1_hurdle_th050"),
        "uci_hash_matches_phase12": hashes.get("uci_h1_phase8_lightgbm") == EXPECTED_UCI_HASH,
        "synthetic_hash_matches_phase12": hashes.get("synthetic_h1_hurdle_th050") == EXPECTED_SYN_HASH,
        "n_registry": len(recs),
        "monitoring_snapshot": fresh.get("monitoring_snapshot"),
        "forecast_generated": fresh.get("forecast_generated"),
        "data_as_of": fresh.get("data_as_of"),
        "live_data": False,
        "retraining": mon.get("retraining", "disabled"),
        "quality_schema_validity": q.get("schema_validity"),
        "n_alerts": mon.get("n_alerts"),
        "phase11_meta": str(PHASE11_META_PATH.name),
        "registry_path": str(Path(REGISTRY_PATH).name),
    }


def build_bi_exports() -> dict[str, str]:
    risk = load_risk()
    if risk is None:
        raise FileNotFoundError(INVENTORY_RISK_PATH)
    metrics, daily = forecast_performance_tables()
    season = seasonality_table(daily)
    products_fc = product_forecast_h1()
    inv = inventory_bi(risk, products_fc)

    kpi = pd.DataFrame([executive_kpi_row()])
    _write(kpi, "executive_kpis.parquet")

    demand_cols = [
        "sku_id", "sku_name", "store_id", "total_recent_units", "total_recent_revenue",
        "demand_share", "avg_daily_demand", "growth_class", "growth_rate",
        "forecast_mean_h1", "stockout_risk_level", "overstock_risk_level",
        "demand_rank_label", "extract_note",
    ]
    demand_cols = [c for c in demand_cols if c in inv.columns]
    _write(inv[demand_cols], "product_demand.parquet")

    daily_cols = [c for c in REQUIRED_COLUMNS["forecast_performance.parquet"] if c in daily.columns]
    extra = [c for c in daily.columns if c not in daily_cols]
    _write(daily[daily_cols + extra], "forecast_performance.parquet")
    if not metrics.empty:
        _write(metrics, "forecast_metrics.parquet")
    _write(inv, "inventory_risk.parquet")

    recs = build_recommendations(inv)
    _write(recs, "recommendations.parquet")
    _write(pd.DataFrame([system_health_row()]), "system_health.parquet")
    _write(season, "seasonality.parquet")

    schema = {name: str(OUTPUTS_BI_DIR / name) for name in EXPORT_FILES}
    (OUTPUTS_BI_DIR / "schema.json").write_text(
        json.dumps({
            "generated": freshness(),
            "files": schema,
            "required_columns": {k: list(v) for k, v in REQUIRED_COLUMNS.items()},
            "notes": [
                "Compact BI extracts. Full final_predictions.parquet is not duplicated.",
                "Inventory tables are the 1000-row reference extract.",
                "File snapshots are not live data.",
                "forecast_metrics.parquet is optional compact MAE/RMSE/WAPE by dataset-horizon.",
                "seasonality.parquet is optional monthly/weekly TEST actuals.",
            ],
        }, indent=2, default=str),
        encoding="utf-8",
    )
    return schema


def schema_issues(tables: dict[str, pd.DataFrame] | None = None) -> list[str]:
    issues: list[str] = []
    for name, cols in REQUIRED_COLUMNS.items():
        path = OUTPUTS_BI_DIR / name
        if not path.exists():
            issues.append(f"missing {name}")
            continue
        df = tables[name] if tables and name in tables else pd.read_parquet(path)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            issues.append(f"{name} missing columns: {missing}")
        if df.empty:
            issues.append(f"{name} is empty")
    return issues


def load_export(name: str) -> pd.DataFrame:
    path = OUTPUTS_BI_DIR / name
    if not path.exists():
        build_bi_exports()
    return pd.read_parquet(path)


def load_bi_tables(rebuild: bool = False) -> dict[str, pd.DataFrame]:
    missing = [name for name in EXPORT_FILES if not (OUTPUTS_BI_DIR / name).exists()]
    if rebuild or missing:
        build_bi_exports()
    tables = {name: pd.read_parquet(OUTPUTS_BI_DIR / name) for name in EXPORT_FILES}
    season_path = OUTPUTS_BI_DIR / "seasonality.parquet"
    if season_path.exists():
        tables["seasonality.parquet"] = pd.read_parquet(season_path)
    return tables


if __name__ == "__main__":
    written = build_bi_exports()
    problems = schema_issues()
    print("wrote", json.dumps(written, indent=2))
    print("schema_issues", problems or "none")
