"""
Deterministic batch forecast pipeline.

Usage:
  python -m src.forecasting.batch_forecast --input <path> --output <path> --dataset UCI --horizon 1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import OUTPUTS_FORECASTS_DIR, PROJECT_ROOT, SUPPORTED_DATASETS, SUPPORTED_HORIZONS
from src.forecasting.inference import ForecastEngine
from src.forecasting.logging_utils import configure_logging
from src.forecasting.validation import InputValidationError, validate_dataset_horizon

logger = logging.getLogger("forecast_service.batch")


def run_batch(
    input_path: str | Path,
    output_path: str | Path,
    dataset: str,
    horizon: int,
    *,
    include_actual: bool = False,
) -> Path:
    validate_dataset_horizon(dataset, horizon)
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")
    t0 = time.perf_counter()
    if inp.suffix.lower() == ".parquet":
        df = pd.read_parquet(inp)
    elif inp.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(inp)
    else:
        raise InputValidationError("Input must be .parquet or .csv")
    logger.info("batch_start input=%s n=%s dataset=%s horizon=%s", inp, len(df), dataset, horizon)
    engine = ForecastEngine(dataset, int(horizon))
    logger.info(
        "batch_model model_id=%s version=%s hash=%s",
        engine.record["model_id"], engine.record.get("code_version"), engine.hash,
    )
    out = engine.predict(df, include_actual=include_actual)
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.suffix.lower() == ".csv":
        out.to_csv(outp, index=False)
    else:
        out.to_parquet(outp, index=False)
    elapsed = time.perf_counter() - t0
    logger.info(
        "batch_done output=%s n_out=%s duration_s=%.3f rows_per_s=%.1f",
        outp, len(out), elapsed, (len(out) / elapsed if elapsed else 0.0),
    )
    return outp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.forecasting.batch_forecast",
        description="Run Phase 11 registered models on a feature file. Does not retrain.",
    )
    p.add_argument("--input", required=True, help="Parquet/CSV of forecast features")
    p.add_argument("--output", required=True, help="Output parquet/csv path")
    p.add_argument("--dataset", required=True, choices=list(SUPPORTED_DATASETS))
    p.add_argument("--horizon", required=True, type=int, choices=list(SUPPORTED_HORIZONS))
    p.add_argument(
        "--include-actual",
        action="store_true",
        help="Keep actual demand when the column is present (evaluation only)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_batch(
            args.input, args.output, args.dataset, args.horizon,
            include_actual=args.include_actual,
        )
        return 0
    except (InputValidationError, FileNotFoundError, ValueError) as exc:
        logger.error("batch_failed err=%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
