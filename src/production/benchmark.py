"""Measure inference latency for Phase 13. Does not retrain or alter models."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.config import SAMPLES_DIR
from src.forecasting.inference import ForecastEngine
from src.forecasting.make_samples import write_samples


def _unique_batch(df: pd.DataFrame, n: int) -> pd.DataFrame:
    base = df.reset_index(drop=True)
    rows = []
    for i in range(n):
        rec = base.iloc[i % len(base)].to_dict()
        rec["product_key"] = f"{rec['product_key']}__bench{i}"
        rows.append(rec)
    return pd.DataFrame(rows)


def run_benchmark() -> dict[str, Any]:
    sample_path = SAMPLES_DIR / "uci_h1_sample.parquet"
    if not sample_path.exists():
        write_samples()
    df = pd.read_parquet(sample_path)
    t0 = time.perf_counter()
    engine = ForecastEngine("UCI", 1)
    load_s = time.perf_counter() - t0

    def timed(frame: pd.DataFrame) -> float:
        start = time.perf_counter()
        out = engine.predict(frame, include_actual=False)
        elapsed = time.perf_counter() - start
        if len(out) != len(frame):
            raise RuntimeError("benchmark row count mismatch")
        return elapsed

    single_s = timed(df.head(1))
    batch10_s = timed(_unique_batch(df, 10))
    batch100_s = timed(_unique_batch(df, 100))
    batch500_s = timed(_unique_batch(df, 500))
    return {
        "dataset": "UCI",
        "horizon": 1,
        "load_s": round(load_s, 4),
        "single_s": round(single_s, 4),
        "batch10_s": round(batch10_s, 4),
        "batch100_s": round(batch100_s, 4),
        "batch500_s": round(batch500_s, 4),
        "throughput_batch10_rows_per_s": round(10 / batch10_s, 2) if batch10_s else None,
        "throughput_batch100_rows_per_s": round(100 / batch100_s, 2) if batch100_s else None,
        "throughput_batch500_rows_per_s": round(500 / batch500_s, 2) if batch500_s else None,
        "error_rate": 0.0,
        "phase12_baseline": {
            "load_s": 0.019,
            "single_s": 0.0681,
            "batch10_s": 0.0607,
            "throughput_rows_per_s": 164.8,
            "note": "Phase 12 recorded baseline; Phase 13 reports measured values, not invented targets.",
        },
    }
