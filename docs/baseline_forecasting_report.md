# Phase 7 — Baseline Demand Forecasting Report

**Project:** FORESIGHT — Demand & Inventory Intelligence  
**Status:** COMPLETE (executed + validated)  
**Validation:** 69/69 PASS

---

## 1. Objective

Establish reliable, leakage-safe baseline demand forecasts before Phase 8 ML.
Baselines become the hard benchmarks ML must beat.

## 2. Phase 6 input

| Item | Value |
|---|---|
| Input | `data/processed/features/forecast_features.parquet` |
| Rows | **1,995,496** |
| Columns | **62** |
| Date range | 2009-12-01 → 2025-12-31 |
| Sources | {'SYNTHETIC': 1461000, 'UCI': 534496} |

## 3. Forecasting grain

`date + source_dataset + entity_id + product_key`

UCI and SYNTHETIC are evaluated separately — never combined into one continuous series.

## 4. Target

`units_sold` (nulls in input: 0)

## 5. Split dates

Chronological splits inherited from Phase 6 (`split` column). No random splitting.

### SYNTHETIC
| Split | Start | End | Rows |
|---|---|---|---|
| train | 2022-01-01 | 2025-03-13 | 1,168,000 |
| validation | 2025-03-14 | 2025-08-06 | 146,000 |
| test | 2025-08-07 | 2025-12-31 | 147,000 |

### UCI
| Split | Start | End | Rows |
|---|---|---|---|
| train | 2009-12-01 | 2011-07-13 | 401,604 |
| validation | 2011-07-14 | 2011-09-25 | 53,174 |
| test | 2011-09-26 | 2011-12-09 | 79,718 |

## 6. Baseline models

| Model | Formula | Leakage control |
|---|---|---|
| Naive | y(t-1) | lag within grain |
| Seasonal Naive | y(t-7) | lag within grain |
| Moving Average 7/14/30 | mean(y[t-w..t-1]) | shift(1) before roll |
| Historical Mean | expanding mean(y[1..t-1]) | excludes current |

**Seasonal period = 7**

Rationale: Phase 5 EDA (§8 Seasonality) found clear day-of-week effects for both SYNTHETIC (weekend lift) and UCI (weekday wholesale peaks). A 7-day seasonal naive aligns forecasts with the same weekday.

Empirical DOW mean CV: UCI=0.173, SYNTHETIC=0.2241

## 7. Metric definitions

- **MAE** = mean(\|y − ŷ\|)
- **RMSE** = sqrt(mean((y − ŷ)²))
- **WAPE** = Σ\|y − ŷ\| / Σ\|y\| × 100 (0 if Σ\|y\|=0)
- **sMAPE** = mean(2\|y − ŷ\| / (\|y\|+\|ŷ\|)) × 100; zero-zero pairs contribute 0
- **MAPE** = mean(\|y − ŷ\| / \|y\|) × 100 **only where y ≠ 0**; else N/A

Priority for business interpretation: MAE, RMSE, WAPE, sMAPE.

## 8. UCI results (TEST)

| Model | MAE | RMSE | MAPE | sMAPE | WAPE | n |
|---|---:|---:|---:|---:|---:|---:|
| moving_average_30 | 18.8542 | 72.0799 | 317.0346 | 84.3796 | 86.3870 | 79,545 |
| moving_average_14 | 19.0853 | 72.6680 | 309.3890 | 83.0470 | 87.4460 | 79,545 |
| moving_average_7 | 19.5614 | 74.2491 | 301.1717 | 82.4668 | 89.6276 | 79,545 |
| historical_mean | 20.5473 | 75.7927 | 379.6150 | 89.2190 | 94.1446 | 79,545 |
| naive | 23.6822 | 99.8604 | 313.5110 | 88.7279 | 108.5081 | 79,545 |
| seasonal_naive | 24.2402 | 91.2787 | 336.7925 | 90.0892 | 110.7666 | 78,419 |

## 9. Synthetic results (TEST)

| Model | MAE | RMSE | MAPE | sMAPE | WAPE | n |
|---|---:|---:|---:|---:|---:|---:|
| naive | 5.2717 | 10.4688 | 97.0020 | 45.1750 | 72.8181 | 147,000 |
| historical_mean | 7.3879 | 10.0161 | 58.6245 | 150.1693 | 102.0497 | 147,000 |
| moving_average_30 | 7.4157 | 10.1194 | 59.8022 | 150.7727 | 102.4335 | 147,000 |
| moving_average_7 | 7.4786 | 11.0391 | 69.9000 | 114.9469 | 103.3033 | 147,000 |
| moving_average_14 | 7.8060 | 10.8799 | 64.5076 | 141.2780 | 107.8252 | 147,000 |
| seasonal_naive | 9.1952 | 14.9194 | 83.9620 | 89.5054 | 127.0150 | 147,000 |

## 10. Product results

### SYNTHETIC (model=naive)
Best WAPE products:
- `SYN_SKU_00068` (STORE_005): WAPE=29.21, MAE=10.27, units=5170
- `SYN_SKU_00057` (STORE_008): WAPE=29.29, MAE=8.12, units=4073
- `SYN_SKU_00060` (STORE_008): WAPE=30.40, MAE=7.07, units=3421
Worst WAPE products:
- `SYN_SKU_00064` (STORE_001): WAPE=200.29, MAE=4.66, units=342
- `SYN_SKU_00064` (STORE_010): WAPE=200.00, MAE=4.65, units=342
- `SYN_SKU_00090` (STORE_006): WAPE=200.00, MAE=6.67, units=490
### UCI (model=moving_average_30)
Best WAPE products:
- `UCI_S` (ONLINE): WAPE=0.00, MAE=0.00, units=1
- `UCI_15060b` (ONLINE): WAPE=0.00, MAE=0.00, units=2
- `UCI_47518f` (ONLINE): WAPE=0.00, MAE=0.00, units=2
Worst WAPE products:
- `UCI_79063D` (ONLINE): WAPE=22973.40, MAE=306.31, units=4
- `UCI_16046` (ONLINE): WAPE=15780.00, MAE=315.60, units=4
- `UCI_23702` (ONLINE): WAPE=4520.83, MAE=60.28, units=104

## 11. Store results (Synthetic, best model)

| Store | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| STORE_003 | 4.8736 | 9.6567 | 70.0813 |
| STORE_007 | 5.2118 | 10.3496 | 70.6595 |
| STORE_008 | 5.3418 | 10.6853 | 71.4517 |
| STORE_005 | 5.2991 | 10.4596 | 71.8580 |
| STORE_006 | 5.4589 | 10.8128 | 73.4901 |
| STORE_009 | 5.2926 | 10.3810 | 73.6173 |
| STORE_002 | 5.2571 | 10.4302 | 73.7236 |
| STORE_004 | 5.4638 | 10.7709 | 73.7492 |
| STORE_001 | 5.3914 | 10.8265 | 74.3060 |
| STORE_010 | 5.1264 | 10.2624 | 75.3708 |

## 12. Best baselines (TEST)

| Source | Best model | MAE | RMSE | sMAPE | WAPE |
|---|---|---:|---:|---:|---:|
| SYNTHETIC | naive | 5.2717 | 10.4688 | 45.1750 | 72.8181 |
| UCI | moving_average_30 | 18.8542 | 72.0799 | 84.3796 | 86.3870 |

These are the Phase 8 ML benchmarks.

## 13. Business insights

### High- vs lower-revenue SKUs (actual Pareto @ 80%)

| source_dataset | segment | n_skus | revenue_share_pct | model | MAE | RMSE | WAPE | pareto_threshold | pareto_sku_count | pareto_sku_pct |
|---|---|---|---|---|---|---|---|---|---|---|
| SYNTHETIC | high_revenue | 35 | 80.55 | naive | 5.1281 | 9.421 | 63.4077 | 0.8 | 35 | 35.0 |
| SYNTHETIC | lower_revenue | 65 | 19.45 | naive | 5.349 | 9.9656 | 104.2339 | 0.8 | 35 | 35.0 |
| UCI | high_revenue | 1009 | 80.02 | moving_average_30 | 22.3422 | 38.4259 | 104.0568 | 0.8 | 1009 | 20.24 |
| UCI | lower_revenue | 3975 | 19.98 | moving_average_30 | 11.2273 | 17.4009 | 145.9175 | 0.8 | 1009 | 20.24 |

OBSERVATION: Best baseline differs by source if ranks diverge — do not force a universal winner.

EVIDENCE: See comparison tables above (TEST WAPE ranking).

BUSINESS INTERPRETATION: Stable, high-volume Synthetic store-SKU series favor smoothed averages; intermittent UCI wholesale demand may favor seasonal/naive patterns differently.

IMPLICATION FOR ML: Phase 8 must beat the **source-specific** best WAPE above, with separate models or clearly separated evaluations.

## 14. Limitations

- Baselines ignore price, promo, and inventory signals (intentional for Phase 7).
- Warm-up rows (insufficient history) yield NaN predictions and are excluded from metrics via finite masks.
- UCI has intermittent demand and many zero/sparse SKU-days → MAPE is computed only on non-zero actuals.
- Seasonal period fixed at 7 from Phase 5 DOW evidence; monthly seasonality is not modeled here.
- No ML / Prophet / SARIMA trained in this phase.

## 15. Phase 8 recommendations

1. Train ML **separately** per `source_dataset`.
2. Beat the source-specific best baseline WAPE in §12.
3. Use Phase 6 features (`lag`, `rolling`, price/promo/inventory where available).
4. Preserve chronological splits; never random-split.
5. Report MAE / RMSE / WAPE / sMAPE on the same TEST windows.
6. Investigate worst-WAPE products/stores from §10–11 for feature gaps.
7. Consider intermittent-demand methods for sparse UCI SKUs.

## Charts

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\baseline\baseline_wape_comparison.png`
- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\baseline\syn_high_revenue.png`
- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\baseline\syn_store_sku.png`
- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\baseline\uci_high_revenue.png`
- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\baseline\uci_high_volume.png`
- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\baseline\uci_low_volume.png`

## Files created

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\baseline\baseline_predictions.parquet`
- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\baseline\baseline_metrics.parquet`
- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\baseline\baseline_metrics_by_source.parquet`
- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\baseline\baseline_metrics_by_product.parquet`
- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\baseline\baseline_metrics_by_entity.parquet`
- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\baseline\baseline_comparison.parquet`
- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\baseline\baseline_high_value_sku.parquet`

---

*Generated from actual Phase 7 execution. No simulated metrics.*
