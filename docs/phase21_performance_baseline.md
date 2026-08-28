# Phase 21 — Performance Baseline

## Production Model

- **Model ID:** `phase20_synthetic_lightgbm`
- **Location:** `models/final/phase20/phase20_synthetic_lightgbm.joblib`
- **Source Dataset:** SYNTHETIC (weekly SKU-level)
- **Validated Production Horizon:** 6 weeks (h1–h6)

## Validation Baselines (Phase 19/20 Evidence)

These metrics come from **controlled backtest validation**, not live production measurement:

| Metric | Value | Source |
|--------|-------|--------|
| Overall WAPE | 13.96% | Phase 19 rolling-origin backtest |
| Validated h1–h6 WAPE | 11.03% | Phase 19 supported-horizon backtest |

## Important Distinction

**Validation performance ≠ production performance.**

- Backtests use historical splits with known actuals.
- Production forecasts are generated forward; actuals arrive with lag.
- Phase 21 reports `PENDING_ACTUALS` until real production demand is observed.
- Deviations from validation baselines in production are **investigation signals**, not automatic rollback triggers.

## Known Limitation

Holiday bias during Nov–Dec is reduced but not fully eliminated. Holiday-period monitoring is tracked separately.

## Extended Horizon (h7–h8)

Weeks 7–8 may be generated with `EXTENDED_PARTIAL` accuracy status. They are **not** included in the primary validated production KPI.
