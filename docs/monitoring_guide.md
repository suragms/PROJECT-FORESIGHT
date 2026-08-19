# Monitoring guide

Implementation of `docs/forecast_monitoring_plan.md`. Thresholds are from Phases 9–11, not invented business KPIs.

## Run

```bash
python -m src.monitoring.run_monitoring
```

## Outputs (`outputs/monitoring/`)

| File | Contents |
| --- | --- |
| `data_quality_report.json` | row count, missingness, duplicates, invalid negatives, category changes, date gaps |
| `forecast_monitoring_report.json` | prediction mean/median/std, zero rate, volume, horizon mix, alerts |
| `accuracy_monitoring_report.json` | MAE/RMSE/WAPE/sMAPE/bias by dataset, horizon, entity, demand regime — **only where actuals exist** |
| `drift_report.json` | PSI and KS on `units_sold_lag_1`, `rolling_mean_7`, `average_unit_price` vs train split |
| `monitoring_summary.json` | alert count, pointers, in-process API counters |
| `api_metrics.json` | request count, error rate, auth failures, rate-limit events, mean latency, batch size |

API counters are **in-process** and empty unless that worker served traffic. They do not replace a production APM.

Monitoring **does not retrain** and **does not modify** frozen models. Automatic retraining is not enabled.

## Alert codes

| Code | Trigger (evidence) |
| --- | --- |
| `data_quality_degradation` | missing required columns or duplicates |
| `feature_drift` | unseen category rate > 5%, or PSI > 0.20 |
| `forecast_distribution_drift` | SYNTHETIC **h=1** zero-prediction rate outside 50–75% (hurdle TEST ~62%). Long-horizon direct models are excluded because they are not the hurdle. |
| `zero_demand_false_positive_increase` | SYNTHETIC **h=1** P(pred>0\|actual=0) > 10% |
| `accuracy_degradation` | UCI h=1 WAPE > 105 (fold 2) or > 119.2 (1.5× TEST); SYNTHETIC h=1 WAPE > 39.4 |

PSI > 0.20 is an engineering default commonly used as a “shift worth reviewing” cutoff, documented as such.

Accuracy is never computed for rows without actuals.
