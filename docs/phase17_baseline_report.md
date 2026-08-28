# Phase 17 — Seasonal-Naive Baseline Report

**Methodology:** Same-week-last-year prediction at weekly SKU granularity.
**Horizon:** 8 weeks (Zidio spec: 6-8 weeks).
**Validation:** Rolling-origin cross-validation with up to 5 folds.

## Results

| Source | WAPE | Bias | Predictions | SKUs |
|--------|------|------|-------------|------|
| SYNTHETIC | 25.51% | +14.75 | From backtest | 100 |
| UCI | 91.63% | +5.92 | From backtest | 4,917 |

## Interpretation

- **SYNTHETIC (25.51%)**: Reasonable baseline for a well-structured 4-year dataset with clear seasonality. The positive bias indicates the seasonal-naive tends to under-forecast (actual exceeds forecast).
- **UCI (91.63%)**: Very high WAPE reflects the extreme sparsity and long-tail nature of UCI's 4,917 StockCodes. Many SKUs have zero demand in the seasonal reference week. This is a genuinely hard forecasting problem.

## Baseline Artifacts

- `data/phase17/backtests/seasonal_naive_results.parquet`
