# Phase 22 — Model Card

## Model Name

**phase20_synthetic_lightgbm**

## Version

Phase 20 Production Promotion (2026)

## Parent Candidate

`phase19_synthetic_lightgbm` (copy promoted, not retrained)

## Dataset

**SYNTHETIC** — Multi-store relational retail intelligence dataset (weekly SKU aggregation)

UCI Online Retail II is a **RESEARCH CANDIDATE** — not used for production forecasting.

## Forecast Grain

Weekly SKU-level

## Features

45 features per `docs/phase20_feature_contract.json`:

- Lag features (1, 2, 4, 7, 13, 26, 52 weeks)
- Rolling statistics (mean, std, min, max over 4/8 weeks)
- Calendar features (season, holiday indicators)
- Price and promotion features

## Training / Validation Method

- Rolling-origin backtest on Synthetic weekly data
- Phase 19 hardening with holiday features
- Leakage audit: all features PASS

## Metrics (Validation / Backtest)

| Metric | Value |
|--------|-------|
| Overall WAPE | 13.96% |
| Validated h1–h6 WAPE | 11.03% |
| Seasonal Naive Baseline (Synthetic) | 25.51% |

**These are validation metrics, not live production performance.**

## Supported Horizon

6 weeks (h1–h6) — default production forecast

## Extended Horizon

Weeks 7–8 — **EXTENDED / PARTIAL ACCURACY** — not default production KPI

## Known Limitations

1. Holiday bias in Nov–Dec partially unresolved
2. h7–h8 have partial accuracy
3. Production actual performance: PENDING_ACTUALS
4. Quantile / hurdle companion models not implemented for this production path

## Intended Use

- Weekly demand forecasting for Synthetic retail SKUs
- Inventory risk scoring and decision support
- API and dashboard serving in academic/reference deployment

## Out-of-Scope Use

- UCI dataset production forecasting
- Automated purchase order placement
- Financial guarantee or ROI claims
- Real-time cloud autoscaling deployment (not implemented)

## Monitoring

Phase 21 observability layer monitors data quality, feature contract, drift, integrity, and risk consistency. Run: `python src/run_phase21.py`

## Rollback

See `docs/phase20_rollback_plan.md`. Phase 19 candidate artifact remains unchanged at `models/phase19/synthetic/phase19_synthetic_lightgbm.joblib`.

## Artifact Location

`models/final/phase20/phase20_synthetic_lightgbm.joblib`

SHA-256: `96a88f1dbb8e1904f2c0b79877afe7bfe30ef5336f8d4598dc07d6adf895e086`
