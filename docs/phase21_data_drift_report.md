# Phase 21 — Data Drift Report

## Method

Data drift compares a **baseline window** (first 70% of weeks in reference data) against a **current window** (most recent ~20% of weeks) using Population Stability Index (PSI).

## Thresholds (Project-Specific)

These thresholds are documented for this monitoring layer only — they are not universal standards.

| PSI Range | Classification |
|-----------|----------------|
| < 0.10 | STABLE |
| 0.10 – 0.20 | WATCH |
| 0.20 – 0.25 | DRIFT |
| ≥ 0.25 | CRITICAL_DRIFT |

## Overall Status Logic

- **FAIL** — 3+ features at CRITICAL_DRIFT
- **WARNING** — 1+ CRITICAL_DRIFT or 5+ DRIFT
- **PASS** — otherwise

## Monitored Dimensions

- Numeric contract features (up to 20 per run)
- Weekly demand distribution (`units_sold`)
- SKU coverage (baseline vs current week window)

## Baseline Source

`data/phase19/features/synthetic_weekly_features.parquet` — validated Phase 19/20 reference data. No production history is fabricated.

## Recommended Actions

- **WATCH** — review recent data ingestion and feature pipeline
- **DRIFT** — investigate feature engineering changes or business events
- **CRITICAL_DRIFT** — escalate; verify data source integrity before trusting forecasts
