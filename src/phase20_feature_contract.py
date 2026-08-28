"""Phase 20 — Build feature contract from Phase 19 features."""

import json
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_PATH = os.path.join(BASE, "data", "phase19", "features", "synthetic_weekly_features.parquet")
AUDIT_PATH = os.path.join(BASE, "data", "phase19", "features", "leakage_audit.json")
OUT = os.path.join(BASE, "docs", "phase20_feature_contract.json")

SKIP = {"week", "source_dataset", "product_key", "units_sold", "revenue",
        "avg_unit_price", "transaction_count", "unique_customers",
        "promotion_flag", "store_count", "season", "holiday_names"}


def build_contract():
    df = pd.read_parquet(FEAT_PATH)
    audit_lookup = {}
    if os.path.exists(AUDIT_PATH):
        audit_lookup = {a["feature"]: a for a in json.load(open(AUDIT_PATH))}

    feature_names = [
        c for c in df.columns if c.startswith(("lag_", "rolling_", "ewm_", "sin_", "cos_", "season_"))
        or c in ("week_of_year", "month", "quarter", "year", "price_lag1", "promo_lag1",
                 "is_holiday_week", "holiday_count", "weeks_to_next_holiday",
                 "weeks_since_last_holiday", "holiday_x_promo")
    ]

    features = []
    for name in feature_names:
        if name in SKIP:
            continue
        audit = audit_lookup.get(name, {})
        dtype = str(df[name].dtype)
        features.append({
            "feature_name": name,
            "dtype": dtype,
            "required": True,
            "source": audit.get("source", "phase19_feature_pipeline"),
            "available_at_prediction_time": audit.get("available_at_prediction_time", True),
            "default_behavior": "reject_if_missing",
            "leakage_status": audit.get("leakage_status", "PASS"),
        })

    contract = {
        "model_id": "phase20_synthetic_lightgbm",
        "feature_version": "phase19_weekly_holiday",
        "feature_count": len(features),
        "features": features,
    }
    with open(OUT, "w") as f:
        json.dump(contract, f, indent=2)
    print(f"Feature contract: {len(features)} features -> {OUT}")
    return contract


if __name__ == "__main__":
    build_contract()
