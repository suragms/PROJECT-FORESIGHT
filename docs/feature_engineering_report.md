# Phase 6 — Feature Engineering Report

**Project:** FORESIGHT — Demand & Inventory Intelligence  
**Status:** COMPLETE (executed + validated)  
**Primary output:** `data/processed/features/forecast_features.parquet`  
**Validation:** 77/77 PASS

---

## 1. Input row count

| Metric | Value |
|---|---|
| Input table | `data/processed/integrated/forecast_base.parquet` |
| Input rows | **1,995,496** |
| Input columns | **12** |

Input columns: `date, source_dataset, entity_id, entity_type, product_key, sku_id, units_sold, revenue, average_unit_price, transaction_count, unique_customers, promotion_flag`

## 2. Output row count

| Metric | Value |
|---|---|
| Output rows | **1,995,496** |
| Output columns | **62** |
| Engineered / derived columns (excl. raw CAM) | **50** |

## 3. Input columns

Preserved from CAM `forecast_base`:

- `date`
- `source_dataset`
- `entity_id`
- `entity_type`
- `product_key`
- `sku_id`
- `units_sold`
- `revenue`
- `average_unit_price`
- `transaction_count`
- `unique_customers`
- `promotion_flag`

## 4. Output feature count

Total columns in `forecast_features.parquet`: **62**  
(Includes grain keys, target, engineered features, split, and flags.)

## 5. Feature groups

| Group | Features |
|---|---|
| target | 1 — units_sold |
| calendar | 8 — year, month, quarter, week_of_year, day_of_week, day_of_month, day_of_year, is_weekend |
| cyclical | 4 — month_sin, month_cos, dow_sin, dow_cos |
| lag | 8 — units_sold_lag_1, units_sold_lag_2, units_sold_lag_3, units_sold_lag_7, units_sold_lag_14, units_sold_lag_21, units_sold_lag_28, units_sold_lag_30 |
| rolling | 6 — rolling_mean_7, rolling_mean_14, rolling_mean_30, rolling_std_7, rolling_std_14, rolling_std_30 |
| demand_trend | 4 — demand_change_1, demand_change_7, demand_growth_7, demand_growth_30 |
| price | 5 — average_unit_price, base_price, discount_pct, price_lag_1, price_change |
| promotion | 3 — promotion_flag, promotion_available, promo_rolling_7 |
| product | 3 — category, sub_category, brand |
| entity | 3 — region, store_type, store_size_sqft |
| inventory | 4 — ending_inventory, on_order_qty, stockout_flag, historical_doi |
| calendar_dim | 2 — is_holiday, season |
| metadata | 2 — split, insufficient_history |

## 6. Train / validation / test dates

Chronological splits computed **per `source_dataset`** (80% / 10% / 10% of each source date span). No random splitting.

| Source | Split | Start | End | Rows |
|---|---|---|---|---|
| SYNTHETIC | train | 2022-01-01 | 2025-03-13 | 1,168,000 |
| SYNTHETIC | validation | 2025-03-14 | 2025-08-06 | 146,000 |
| SYNTHETIC | test | 2025-08-07 | 2025-12-31 | 147,000 |
| UCI | train | 2009-12-01 | 2011-07-13 | 401,604 |
| UCI | validation | 2011-07-14 | 2011-09-25 | 53,174 |
| UCI | test | 2011-09-26 | 2011-12-09 | 79,718 |

## 7. Train / validation / test row counts

| Split | Rows |
|---|---|
| train | **1,569,604** |
| validation | **199,174** |
| test | **226,718** |

## 8. Missing-value summary

**Strategy:** Lag / rolling / demand-trend NaNs at the start of each series are **left as NaN** (not zero-filled). Zero would invent false history. An `insufficient_history` flag marks rows where `units_sold_lag_1` is NaN. Source-specific NaNs (UCI promotions, UCI categories, UCI inventory) are preserved as unknown — never fabricated.

Top missing engineered features:

| Feature | Missing count | Missing % |
|---|---|---|
| promo_rolling_7 | 535,496 | 26.84% |
| units_sold_lag_30 | 149,115 | 7.47% |
| demand_growth_30 | 149,115 | 7.47% |
| units_sold_lag_28 | 140,421 | 7.04% |
| units_sold_lag_21 | 108,741 | 5.45% |
| units_sold_lag_14 | 75,156 | 3.77% |
| demand_change_7 | 44,567 | 2.23% |
| units_sold_lag_7 | 39,275 | 1.97% |
| demand_growth_7 | 39,275 | 1.97% |
| units_sold_lag_3 | 17,425 | 0.87% |
| units_sold_lag_2 | 11,757 | 0.59% |
| rolling_std_7 | 11,757 | 0.59% |
| rolling_std_14 | 11,757 | 0.59% |
| rolling_std_30 | 11,757 | 0.59% |
| demand_change_1 | 11,757 | 0.59% |

## 9. Leakage validation

Automated checks in `src/validate_features.py` cover:

1. Lag features use previous observations only  
2. Rolling features exclude the current target (`shift(1)` before rolling)  
3. No future target enters features (first-row lag/rolling = NaN)  
4. Features never cross source / entity / product boundaries  
5. Inventory features are lag-1 shifted (ending inventory depends on same-day sales)  
6. Train dates precede validation; validation precedes test (per source)

**Result:** 77/77 PASS

## 10. Validation result

```
77/77 PASS
```

Also verified: output exists & readable, row count matches input, no duplicate grain, no null grain keys, no infinite values, UCI/Synthetic separation maintained.

## 11. ML compatibility

Existing ML engine (`src/forecasting.py`) expects legacy column names (`sin_month`, `units_sold_rolling_mean_7`, EWM, etc.).

| Item | Detail |
|---|---|
| Adapter | `src/feature_adapter.py` |
| Renames | {'dow_sin': 'sin_day_of_week', 'dow_cos': 'cos_day_of_week', 'month_sin': 'sin_month', 'month_cos': 'cos_month', 'rolling_mean_7': 'units_sold_rolling_mean_7', 'rolling_std_7': 'units_sold_rolling_std_7', 'rolling_mean_14': 'units_sold_rolling_mean_14', 'rolling_mean_30': 'units_sold_rolling_mean_30', 'rolling_std_14': 'units_sold_rolling_std_14', 'rolling_std_30': 'units_sold_rolling_std_30'} |
| Legacy-only (added by adapter if needed) | ['units_sold_ewm_7', 'units_sold_ewm_28'] |
| Phase 7 recommendation | Consume `forecast_features.parquet` directly for baselines |
| Legacy ML path | Keep using `build_forecasting_feature_matrix` via `feature_adapter` + `cam_adapter` |

Phase 6 does not duplicate the legacy SKU-level feature matrix. Use adapt_phase6_to_legacy_ml() to bridge Phase 6 columns, or build_forecasting_feature_matrix() on a legacy sales frame for the existing ML training path. Phase 7 baselines should consume forecast_features.parquet directly.

## 12. Files created

| File | Role |
|---|---|
| `data/processed/features/forecast_features.parquet` | Primary feature store |
| `notebooks/05_feature_engineering.ipynb` | Executed Phase 6 notebook |
| `src/feature_engineering.py` | Phase 6 pipeline |
| `src/validate_features.py` | Leakage + integrity tests |
| `src/feature_adapter.py` | Legacy ML compatibility |
| `docs/feature_registry.csv` | Feature catalog |
| `docs/feature_quality_report.csv` | Quality metrics |
| `docs/feature_engineering_report.md` | This report |

## 13. Limitations

- UCI has no verified promotions, categories, brands, store attributes, or inventory — those fields remain NaN.
- Inventory features use **prior-day** ending inventory; day-0 of each Synthetic series is NaN after the shift.
- Rolling std is NaN when fewer than 2 prior observations exist (not zero-filled).
- Phase 6 does not emit legacy EWM columns; use `feature_adapter.adapt_phase6_to_legacy_ml` if needed.
- Calendar holiday/season join is on `date` only (shared enrichment across sources for overlapping calendar logic via unique dates).

## 14. Recommendations for Phase 7

1. Train baselines **separately** for UCI and SYNTHETIC on `forecast_features.parquet`.
2. Use the `split` column — do not re-split randomly.
3. Prefer rows with `insufficient_history == 0` for metrics that require lags (or evaluate NaN-tolerant metrics).
4. Baselines (Naive, Seasonal Naive, Moving Average, Historical Mean) can start from `units_sold` + grain + `split` without waiting on ML features.
5. Do **not** mix UCI and Synthetic series when fitting or scoring.
6. Keep the existing ML engine untouched until Phase 8; bridge via `feature_adapter` if needed.

---

*Generated from actual pipeline execution. Do not treat as simulated.*
