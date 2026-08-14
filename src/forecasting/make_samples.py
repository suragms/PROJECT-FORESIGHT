"""Write small TEST feature samples for API/inference tests. Does not retrain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import FEATURES_PATH, SAMPLES_DIR
from src.forecasting.preprocessing import expected_features
from src.ml_forecasting import prepare_features
from src.phase10_hurdle_forecasting import _usable


def write_samples(n: int = 12) -> dict[str, Path]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for src in ["UCI", "SYNTHETIC"]:
        sub, _, _ = prepare_features(df, src)
        test = _usable(sub)
        test = test[test["split"] == "test"].head(n).copy()
        path = SAMPLES_DIR / f"{src.lower()}_h1_sample.parquet"
        test.to_parquet(path, index=False)
        out[src] = path
        # JSON payload for API docs/examples (no units_sold as a predictor)
        feats = expected_features(src)
        cols = ["date", "source_dataset", "entity_id", "product_key"] + feats["numeric"] + feats["categorical"]
        cols = [c for c in cols if c in test.columns]
        rec = test[cols].iloc[0].to_dict()
        rec["date"] = str(pd.to_datetime(rec["date"]).date())
        for k, v in list(rec.items()):
            if hasattr(v, "item"):
                rec[k] = v.item()
            if isinstance(rec[k], float) and (pd.isna(rec[k]) or rec[k] != rec[k]):
                rec[k] = None
        with open(SAMPLES_DIR / f"{src.lower()}_h1_sample.json", "w", encoding="utf-8") as f:
            json.dump({"source_dataset": src, "horizon": 1, "record": rec}, f, indent=2, default=str)
    return out


if __name__ == "__main__":
    print(write_samples())
