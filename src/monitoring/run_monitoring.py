"""Run monitoring reports against Phase 11 final forecasts (and optional live batches)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    FEATURES_PATH,
    FINAL_FORECASTS_PATH,
    OUTPUTS_MONITORING_DIR,
)
from src.forecasting.logging_utils import configure_logging
from src.monitoring.data_quality import data_quality_report
from src.monitoring.forecast_monitor import evaluate_alerts, forecast_distribution, ks_stat, psi
from src.monitoring.metrics import accuracy_table, by_group, by_regime

logger = logging.getLogger("forecast_service.monitoring")

REF_FEATURES = ["units_sold_lag_1", "rolling_mean_7", "average_unit_price"]


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def run_monitoring(
    forecast_path: Path | None = None,
    features_path: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    configure_logging()
    out_dir = Path(out_dir or OUTPUTS_MONITORING_DIR)
    fc_path = Path(forecast_path or FINAL_FORECASTS_PATH)
    ft_path = Path(features_path or FEATURES_PATH)
    fc = pd.read_parquet(fc_path)
    quality = data_quality_report(fc, required=["forecast_date", "source_dataset", "entity_id", "product_key", "prediction"])
    dist = forecast_distribution(fc)
    acc_overall = accuracy_table(fc) if "actual" in fc.columns else {"n_with_actuals": 0, "metrics": None}
    by_ds_h = by_group(fc, ["source_dataset", "horizon"]) if "actual" in fc.columns else []
    by_ent = by_group(fc, ["source_dataset", "entity_id"]) if "actual" in fc.columns else []
    regimes = by_regime(fc) if "actual" in fc.columns else []

    drift = {"features": {}, "note": "Reference = Phase 6 train split; current = TEST origins in final forecasts joined to features."}
    if ft_path.exists() and "forecast_date" in fc.columns:
        feat = pd.read_parquet(ft_path)
        feat["date"] = pd.to_datetime(feat["date"])
        train = feat[feat["split"] == "train"] if "split" in feat.columns else feat
        # Join h=1 forecasts back to origin features on grain
        h1 = fc[fc["horizon"] == 1].copy()
        h1["forecast_date"] = pd.to_datetime(h1["forecast_date"])
        merged = h1.merge(
            feat,
            left_on=["forecast_date", "source_dataset", "entity_id", "product_key"],
            right_on=["date", "source_dataset", "entity_id", "product_key"],
            how="inner",
        )
        for col in REF_FEATURES:
            if col not in train.columns or col not in merged.columns:
                continue
            drift["features"][col] = {
                "ref_mean": round(float(train[col].mean()), 4) if train[col].notna().any() else None,
                "cur_mean": round(float(merged[col].mean()), 4) if merged[col].notna().any() else None,
                "ref_std": round(float(train[col].std(ddof=1)), 4) if train[col].notna().sum() > 1 else None,
                "cur_std": round(float(merged[col].std(ddof=1)), 4) if merged[col].notna().sum() > 1 else None,
            }
            p = psi(train[col].to_numpy(), merged[col].to_numpy())
            k = ks_stat(train[col].to_numpy(), merged[col].to_numpy())
            drift["features"][col]["psi"] = None if p != p else round(p, 4)
            drift["features"][col]["ks"] = None if k != k else round(k, 4)

    alerts = []
    for row in by_ds_h:
        alerts.extend(evaluate_alerts(
            quality=quality, dist=dist,
            accuracy={"metrics": row},
            drift=drift,
            dataset=row.get("source_dataset"),
            horizon=int(row["horizon"]) if row.get("horizon") is not None else None,
        ))
    if not by_ds_h:
        alerts.extend(evaluate_alerts(quality=quality, dist=dist, accuracy=acc_overall, drift=drift))

    seen = set()
    uniq = []
    for a in alerts:
        key = (a["code"], a["detail"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)

    accuracy_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": acc_overall,
        "by_dataset_horizon": by_ds_h,
        "by_entity": by_ent[:50],
        "by_regime": regimes,
        "note": "Accuracy is computed only where actual is present. Future rows without actuals are excluded.",
    }
    quality_report = {"generated_at": datetime.now(timezone.utc).isoformat(), **quality}
    dist_report = {"generated_at": datetime.now(timezone.utc).isoformat(), **dist, "alerts": uniq}
    drift_report = {"generated_at": datetime.now(timezone.utc).isoformat(), **drift}
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "forecast_file": str(fc_path),
        "n_forecasts": int(len(fc)),
        "n_alerts": len(uniq),
        "alerts": uniq,
        "accuracy_available": bool(acc_overall.get("n_with_actuals")),
    }
    _write(out_dir / "data_quality_report.json", quality_report)
    _write(out_dir / "forecast_monitoring_report.json", dist_report)
    _write(out_dir / "accuracy_monitoring_report.json", accuracy_report)
    _write(out_dir / "drift_report.json", drift_report)
    _write(out_dir / "monitoring_summary.json", summary)
    logger.info("monitoring_done n=%s alerts=%s out=%s", len(fc), len(uniq), out_dir)
    return summary


if __name__ == "__main__":
    run_monitoring()
