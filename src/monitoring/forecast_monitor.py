"""Forecast-distribution monitoring and evidence-based alerts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import MONITORING
from src.monitoring.metrics import accuracy_table, by_group, by_regime


def forecast_distribution(df: pd.DataFrame) -> dict[str, Any]:
    p = pd.to_numeric(df["prediction"], errors="coerce")
    out = {
        "n": int(len(df)),
        "prediction_mean": round(float(p.mean()), 6) if len(p) else None,
        "prediction_median": round(float(p.median()), 6) if len(p) else None,
        "prediction_std": round(float(p.std(ddof=1)), 6) if len(p) > 1 else 0.0,
        "prediction_min": round(float(p.min()), 6) if len(p) else None,
        "prediction_max": round(float(p.max()), 6) if len(p) else None,
        "zero_prediction_rate_pct": round(100.0 * float((p == 0).mean()), 4) if len(p) else None,
        "forecast_volume": int(len(df)),
    }
    if "horizon" in df.columns:
        out["horizon_distribution"] = {
            str(int(k)): int(v) for k, v in df["horizon"].value_counts().sort_index().items()
        }
    if "source_dataset" in df.columns and "SYNTHETIC" in set(df["source_dataset"].astype(str)):
        syn = df[df["source_dataset"].astype(str) == "SYNTHETIC"]
        if "horizon" in syn.columns:
            syn_h1 = syn[syn["horizon"].astype(int) == 1]
        else:
            syn_h1 = syn
        if len(syn_h1):
            ps = pd.to_numeric(syn_h1["prediction"], errors="coerce")
            out["synthetic_h1_n"] = int(len(syn_h1))
            out["synthetic_zero_prediction_rate_pct"] = round(100.0 * float((ps == 0).mean()), 4)
            if "actual" in syn_h1.columns and syn_h1["actual"].notna().any():
                z = syn_h1["actual"] == 0
                if z.any():
                    out["synthetic_zero_false_positive_rate_pct"] = round(
                        100.0 * float((ps[z] > 0).mean()), 4
                    )
    return out


def evaluate_alerts(
    *,
    quality: dict,
    dist: dict,
    accuracy: dict | None,
    drift: dict | None,
    dataset: str | None = None,
    horizon: int | None = None,
) -> list[dict[str, Any]]:
    alerts = []
    cfg = MONITORING
    if quality.get("missing_required_columns"):
        alerts.append({
            "code": "data_quality_degradation",
            "severity": "warning",
            "detail": f"Missing columns {quality['missing_required_columns']}",
        })
    if quality.get("n_duplicates", 0) > 0:
        alerts.append({
            "code": "data_quality_degradation",
            "severity": "warning",
            "detail": f"duplicate_rate_pct={quality.get('duplicate_rate_pct')}",
        })
    for col, info in (quality.get("category_changes") or {}).items():
        if info.get("unseen_rate_pct", 0) > cfg["unseen_category_rate_warn"]:
            alerts.append({
                "code": "feature_drift",
                "severity": "warning",
                "detail": f"{col} unseen_rate_pct={info['unseen_rate_pct']} (warn>{cfg['unseen_category_rate_warn']})",
            })
    zp = dist.get("synthetic_zero_prediction_rate_pct")
    if zp is not None and (zp < cfg["synthetic_zero_pred_rate_min"] or zp > cfg["synthetic_zero_pred_rate_max"]):
        alerts.append({
            "code": "forecast_distribution_drift",
            "severity": "warning",
            "detail": f"SYNTHETIC zero-prediction rate {zp} outside {cfg['synthetic_zero_pred_rate_min']}-{cfg['synthetic_zero_pred_rate_max']}",
        })
    fp = dist.get("synthetic_zero_false_positive_rate_pct")
    if fp is not None and fp > cfg["synthetic_zero_fp_warn"]:
        alerts.append({
            "code": "zero_demand_false_positive_increase",
            "severity": "warning",
            "detail": f"P(pred>0|actual=0)={fp} > {cfg['synthetic_zero_fp_warn']}",
        })
    if accuracy and accuracy.get("metrics"):
        wape = accuracy["metrics"].get("WAPE")
        if dataset == "UCI" and horizon == 1 and wape is not None:
            if wape > cfg["uci_h1_wape_fold2"]:
                alerts.append({
                    "code": "accuracy_degradation",
                    "severity": "warning",
                    "detail": f"UCI h=1 WAPE {wape} > fold-2 threshold {cfg['uci_h1_wape_fold2']}",
                })
            if wape > cfg["uci_h1_wape_1p5x"]:
                alerts.append({
                    "code": "accuracy_degradation",
                    "severity": "warning",
                    "detail": f"UCI h=1 WAPE {wape} > 1.5x Phase 11 TEST ({cfg['uci_h1_wape_1p5x']})",
                })
        if dataset == "SYNTHETIC" and horizon == 1 and wape is not None and wape > cfg["synthetic_h1_wape_1p5x"]:
            alerts.append({
                "code": "accuracy_degradation",
                "severity": "warning",
                "detail": f"SYNTHETIC h=1 WAPE {wape} > 1.5x TEST ({cfg['synthetic_h1_wape_1p5x']})",
            })
    if drift:
        for feat, rec in (drift.get("features") or {}).items():
            if rec.get("psi") is not None and rec["psi"] > cfg["psi_warn"]:
                alerts.append({
                    "code": "feature_drift",
                    "severity": "warning",
                    "detail": f"{feat} PSI={rec['psi']} > {cfg['psi_warn']}",
                })
    return alerts


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, qs))
    if len(edges) < 3:
        return 0.0
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e = np.clip(e_hist / max(e_hist.sum(), 1), 1e-6, None)
    a = np.clip(a_hist / max(a_hist.sum(), 1), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_stat(expected: np.ndarray, actual: np.ndarray) -> float:
    from scipy.stats import ks_2samp
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    return float(ks_2samp(expected, actual, alternative="two-sided").statistic)
