# Phase 21 — Prediction Monitoring

## Scope

Monitors production forecast distributions from `data/phase20/production_forecasts.parquet` against Phase 19 backtest forecast distributions.

## Tracked Statistics

- Mean, median, standard deviation
- Quantiles (p25, p75)
- Zero forecast rate
- Detected shifts

## Shift Detection Rules

| Signal | Condition |
|--------|-----------|
| Mean shift | >30% change vs baseline mean |
| Variance explosion | Current std > 2× baseline std |
| Excessive zeros | Zero forecast rate > 50% |

Seasonal behavior is not automatically classified as drift; shifts are relative to the approved backtest baseline.

## Forecast Performance

Separated into:

- **FORECAST AVAILABLE** — production forecasts exist
- **ACTUALS AVAILABLE** — measured only when actual demand exists

Production actuals status: `PENDING_ACTUALS` until live demand is collected.

Validation reference metrics (from backtest) are labeled `VALIDATION_REFERENCE_NOT_LIVE_PRODUCTION`.

## Horizon Monitoring

Each horizon h1–h6 reported independently. h7–h8 labeled `EXTENDED_PARTIAL` and excluded from primary KPI.

## Baselines (Not Thresholds)

- Overall WAPE: 13.96%
- h1–h6 WAPE: 11.03%

Performance WARNING if validation reference exceeds baseline by >15%; FAIL if >30%.
