# Phase 22 — Technical Documentation

## Data Pipeline

1. **Raw data** — UCI Online Retail II and Synthetic retail relational dataset
2. **Cleaning** — `src/data_cleaning.py` with documented quality rules
3. **Integration** — Phase 16/17 dataset alignment and provenance tracking
4. **Weekly aggregation** — Synthetic data aggregated to weekly SKU grain for production forecasting

## Feature Engineering

- **45-feature contract** — `docs/phase20_feature_contract.json`
- Features include: lags (1, 2, 4, 7, 13, 26, 52), rolling statistics, calendar/season/holiday indicators
- **Leakage prevention** — each feature audited with `leakage_status: PASS`
- Phase 19 added holiday features to reduce Nov–Dec bias

## Forecast Model

| Property | Value |
|----------|-------|
| Algorithm | LightGBM |
| Model ID | `phase20_synthetic_lightgbm` |
| Parent | `phase19_synthetic_lightgbm` |
| Grain | Weekly SKU-level |
| Horizon | 6 weeks (h7–h8 extended partial) |
| Source | SYNTHETIC only |

## Validation Method

- **Rolling-origin backtest** — train on expanding window, forecast forward
- Metrics computed only where actuals exist
- WAPE = Σ|actual − forecast| / Σ|actual|

## Metrics

- **WAPE** — weighted absolute percentage error (primary)
- **Bias** — mean(forecast − actual); positive = over-forecasting

## Risk Engine

`src/phase20_risk_adapter.py` computes:

- Lead-time demand, weeks of supply, projected balance
- Stockout and overstock risk scores/levels
- Recommended actions (REORDER NOW, WATCH, HEALTHY, MARKDOWN)

## API Adapters

- `src/phase20_api_adapter.py` — forecast generation with source validation
- `src/phase20_risk_adapter.py` — risk matrix and explain endpoint
- `src/api/phase20_routes.py` — `/phase20/*` endpoints
- `src/api/phase21_routes.py` — `/phase21/*` monitoring endpoints

## Monitoring (Phase 21)

| Module | Function |
|--------|----------|
| `phase21_data_quality.py` | Schema, nulls, duplicates |
| `phase21_feature_quality.py` | Contract compliance |
| `phase21_drift_detection.py` | PSI-based drift |
| `phase21_forecast_monitoring.py` | Horizon performance |
| `phase21_risk_monitoring.py` | Risk consistency |
| `phase21_integrity_monitoring.py` | SHA-256 verification |

## Model Integrity

- 12 frozen models in `docs/final_model_registry.json`
- Phase 20 production model in separate `docs/phase20_production_registry.json`
- SHA-256 verified on every monitoring run

## Promotion Lineage

```
Phase 17 candidate → Phase 18 promotion gate → Phase 19 hardening → Phase 20 production copy
```

Copy verified in `docs/phase20_promotion_provenance.json`.

## Testing

```bash
python -m pytest tests -q
```

214 tests including Phase 17–22 coverage.

## Legacy Artifact

`models/lightgbm_forecaster.joblib` — **LEGACY NON-PRODUCTION ARTIFACT ISSUE**. Not part of frozen production stack.
