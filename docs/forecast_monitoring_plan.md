# Forecast Monitoring Plan

Applies after deploying the Phase 11 final models. Thresholds are taken from Phases 8–10 evidence, not invented business KPIs.

## Data drift

| Signal | Warning | Evidence |
| --- | --- | --- |
| Missing required-feature rate | > 0% in a batch (pipeline should reject) | Inference is fail-closed; any accepted missingness is a contract break |
| Unseen category rate (freq→0) | > 5% of rows in a day | Frequency encoder maps unknowns to 0; large spikes mean assortment/store change |
| Feature mean shift (lags, rolling, price) | \|z\|-score vs train > 3 on a daily aggregate | Phase 6 features are the training distribution |
| SYNTHETIC inventory/promo missing | any | Required at origin for the hurdle/direct SYNTHETIC models |

## Forecast drift

| Signal | Warning | Evidence |
| --- | --- | --- |
| Mean prediction vs TEST mean | relative change > 25% over 7 days | UCI residuals correlate with actual level (Phase 9) |
| SYNTHETIC zero-prediction rate | outside 50–75% (TEST predicted zero rate ~62%) | Hurdle TEST predicted_zero_rate |
| SYNTHETIC P(pred>0 \| later actual=0) | > 10% | Phase 8 was 82.64%; hurdle 1.42% — 10% is an early-regression tripwire |
| Forecast row count vs expected grain | ±2% vs entity×product calendar | Missing keys silently drop volume |

## Accuracy (when actuals arrive)

| Dataset | Metric | Warning | Evidence |
| --- | --- | --- | --- |
| UCI h=1 | WAPE | > 105 | Phase 9 fold-2 WAPE 105.31 |
| UCI h=1 | WAPE | > 1.5 × 79.47 ≈ 119.2 | Phase 9 Stable max/min cap 1.50 |
| SYNTHETIC h=1 | WAPE | > 1.5 × 26.25 ≈ 39.4 | same range-ratio rule |
| Both | MAE / RMSE / bias | rolling 28-day bias sign flip persisting 14 days | Phase 8 bias convention mean(pred-actual) |
| Direct h≥7 | WAPE | worse than the Phase 9 recursive WAPE for that horizon | Phase 10 selected direct because it beat recursive |

## Business / operational signals

No dollar stockout or holding-cost thresholds exist in this project. Monitor proxy signals only:

| Signal | Warning | Why |
| --- | --- | --- |
| High-regime under-prediction | high-demand MAE rising vs TEST high-regime | Phase 9 SYNTHETIC high-demand bias was negative |
| Horizon degradation | h=30 WAPE approaching 100 | Phase 9 recursive and Phase 10 direct both degrade |
| Interval coverage | P10–P90 coverage < 70% or > 95% over 28 days | TEST coverage UCI 82%, SYNTHETIC 90%; not calibrated |
| UCI January–April window | extra review | Fold-2 instability |

## Review cadence

- Daily: batch rejects, row counts, missing-feature rate, zero-prediction rate.
- Weekly: WAPE/MAE/bias on newly arrived actuals; interval coverage.
- Seasonal: UCI post-holiday window; SYNTHETIC promo calendar changes.
- Do not auto-retrain over frozen Phase 8 files. Candidate retrains belong in a new versioned model_id.

