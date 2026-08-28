# Phase 21 — Monitoring Architecture

## Objective

Phase 21 provides **additive, non-destructive observability** for the promoted Phase 20 forecasting system. It does not retrain models, modify frozen artifacts, or overwrite production forecasts.

## Components

| Module | Responsibility |
|--------|----------------|
| `src/phase21_common.py` | Paths, utilities, validation baselines |
| `src/phase21_integrity_monitoring.py` | SHA-256 baseline and ongoing integrity checks |
| `src/phase21_data_quality.py` | Schema, nulls, duplicates, SKU coverage |
| `src/phase21_feature_quality.py` | 45-feature contract validation |
| `src/phase21_drift_detection.py` | Data drift (PSI) and prediction drift |
| `src/phase21_forecast_monitoring.py` | Performance and horizon monitoring |
| `src/phase21_risk_monitoring.py` | Risk consistency and distribution |
| `src/phase21_holiday_monitoring.py` | Holiday vs non-holiday comparison |
| `src/phase21_monitoring.py` | Orchestrator, alerts, health score, history |
| `src/run_phase21.py` | Pipeline entry point |

## Outputs

```
data/phase21/monitoring/
├── data_quality_report.json
├── feature_quality_report.json
├── data_drift_report.json
├── prediction_drift_report.json
├── forecast_performance_report.json
├── risk_consistency_report.json
├── model_integrity_report.json
├── monitoring_summary.json
├── alerts.json
└── history/snapshot_<timestamp>.json
```

## Health Score Logic

Rule-based (not AI confidence):

- **CRITICAL** — any component FAIL
- **DEGRADED** — 2+ WARNING/PARTIAL
- **WATCH** — 1 WARNING/PARTIAL
- **HEALTHY** — all PASS

## API (Additive)

- `GET /phase21/health`
- `GET /phase21/monitoring/latest`
- `GET /phase21/alerts`
- `GET /phase21/integrity`

## Dashboard

`dashboard/phase21_monitoring.py` — observability dashboard (does not replace `dashboard/phase20_production.py`).

## Reference Data

- Features: `data/phase19/features/synthetic_weekly_features.parquet`
- Forecasts: `data/phase20/production_forecasts.parquet`
- Risk: `data/phase20/production_risk.parquet`
- Feature contract: `docs/phase20_feature_contract.json`
- Integrity baseline: `docs/phase21_production_integrity_baseline.json`
