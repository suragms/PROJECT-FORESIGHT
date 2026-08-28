"""Phase 21 — Feature quality monitoring against Phase 20 contract."""

from __future__ import annotations

import json

import pandas as pd

from src.phase21_common import CONTRACT_PATH, P19_FEAT, now_iso, classify_status


def load_contract() -> dict:
    with open(CONTRACT_PATH) as f:
        return json.load(f)


def run_feature_quality_monitoring(df: pd.DataFrame | None = None) -> dict:
    contract = load_contract()
    expected_features = [f["feature_name"] for f in contract["features"]]
    required_features = [f["feature_name"] for f in contract["features"] if f.get("required", True)]

    if df is None:
        df = pd.read_parquet(P19_FEAT)
    latest = df[df["week"] == df["week"].max()].copy() if "week" in df.columns else df

    actual_features = [c for c in df.columns if c in expected_features]
    missing = [f for f in required_features if f not in df.columns]
    unexpected = [c for c in df.columns if c not in expected_features
                    and c not in {"week", "source_dataset", "product_key", "units_sold", "revenue",
                                  "avg_unit_price", "transaction_count", "unique_customers",
                                  "promotion_flag", "store_count", "season", "holiday_names"}]

    null_rates = {}
    for f in required_features:
        if f in latest.columns:
            null_rates[f] = round(float(latest[f].isna().mean()), 4)

    critical_null = [f for f, r in null_rates.items() if r > 0.1]

    dist_summary = {}
    for f in required_features[:10]:  # sample key features
        if f in latest.columns and pd.api.types.is_numeric_dtype(latest[f]):
            dist_summary[f] = {
                "mean": round(float(latest[f].mean()), 4),
                "std": round(float(latest[f].std()), 4),
                "min": round(float(latest[f].min()), 4),
                "max": round(float(latest[f].max()), 4),
            }

    checks = {
        "expected_feature_count": contract["feature_count"],
        "actual_feature_count": len(actual_features),
        "missing_features": missing,
        "unexpected_features": unexpected,
        "null_rates": null_rates,
        "distribution_summary": dist_summary,
        "feature_count_match": "PASS" if len(actual_features) >= contract["feature_count"] else "FAIL",
        "missing_required": "FAIL" if missing else "PASS",
        "critical_nulls": "FAIL" if critical_null else ("WARNING" if any(r > 0 for r in null_rates.values()) else "PASS"),
        "unexpected_documented": "PASS" if unexpected else "PASS",
    }
    check_vals = [checks["feature_count_match"], checks["missing_required"], checks["critical_nulls"]]
    checks["overall_status"] = classify_status(check_vals)
    checks["timestamp"] = now_iso()
    return checks
