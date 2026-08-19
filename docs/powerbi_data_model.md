# Power BI data model (preparation)

This document describes the **prepared** analytical extracts under `outputs/bi/`. It is not evidence that a Power BI workspace, app, or gateway has been deployed. **No Power BI deployment was performed.** Tests and `python src/validate_phase15.py` do not require Power BI Desktop.

Refresh assumption: regenerate extracts with `python -m src.bi.exports` from repository files. There is no incremental cloud refresh.

All timestamps on these files are **snapshots**, not live warehouse clocks.

## Export files

| File | Grain | Role |
| --- | --- | --- |
| `executive_kpis.parquet` | 1 row | KPI snapshot (demand, forecast, inventory, ops) |
| `product_demand.parquet` | store × SKU (extract) | Demand, share, growth, rank, risk labels |
| `forecast_performance.parquet` | dataset × horizon × date | Daily mean actual vs forecast vs P10/P90 |
| `forecast_metrics.parquet` | dataset × horizon | Optional monitoring MAE/RMSE/WAPE/bias (not one of the six required files) |
| `inventory_risk.parquet` | store × SKU (extract) | Existing risk fields + 2×2 overlay |
| `recommendations.parquet` | store × SKU (extract) | Decision-support reviews |
| `system_health.parquet` | 1 row | Hash/freshness snapshot |
| `seasonality.parquet` | dataset × pattern × period | Optional TEST seasonality |
| `schema.json` | n/a | Column contract written at export time |

Compact extracts **do not** copy all 957,949 forecast rows.

## Fact tables

### FactDemand (`product_demand.parquet`)

Primary key: (`store_id`, `sku_id`) on the 1,000-row extract.

| Column | Meaning |
| --- | --- |
| sku_id, sku_name, store_id | Entity keys from the extract |
| total_recent_units, total_recent_revenue | Existing extract measures |
| demand_share | units / extract unit total |
| avg_daily_demand | Existing extract field |
| growth_class, growth_rate | Documented TEST split after SYNTHETIC join; else Insufficient Evidence |
| forecast_mean_h1 | Mean h=1 prediction when joined |
| stockout_risk_level, overstock_risk_level | Existing scorer |
| demand_rank_label | TOP_REVENUE / LOW DEMAND / MID via `total_recent_revenue` |
| extract_note | Always states 1000-row reference extract |

### FactForecastDaily (`forecast_performance.parquet`)

Primary key: (`source_dataset`, `horizon`, `forecast_date`).

| Column | Meaning |
| --- | --- |
| actual | Held-out TEST actual (not a future unknown) |
| forecast | Model point prediction — **not** an observation |
| p10, p90 | Interval companions — **not** observations |
| error | forecast − actual |
| absolute_error | \|error\| |
| n | Row count in the daily mean |
| grain | `daily` |

Do not compute accuracy on dates without actuals. Aggregate MAE/RMSE/WAPE/bias live in `executive_kpis.parquet` / monitoring JSON.

### FactInventoryRisk (`inventory_risk.parquet`)

Primary key: (`store_id`, `sku_id`). Same 1,000 extract rows. Includes original scorer columns plus:

* `demand_high` / `inventory_high` / `risk_matrix_cell` — documented strict-median overlay
* `risk_matrix_rule` — text of the overlay
* `extract_note`

## Dimensions (logical)

Power BI can create these from the facts; physical dim tables are not exported to avoid duplication.

| Dimension | Key | Source |
| --- | --- | --- |
| Date | forecast_date | FactForecastDaily |
| Product | sku_id / product_key | extract / forecasts |
| Entity | store_id / entity_id | extract (`STORE_*`) or UCI `ONLINE` |
| Dataset | source_dataset | UCI, SYNTHETIC |

UCI products are **not** rows in FactDemand/FactInventoryRisk. Do not invent inventory measures for them.

## Relationships (intended)

```
DimDate[forecast_date]     1—*  FactForecastDaily[forecast_date]
DimProduct[sku_id]         1—*  FactDemand[sku_id]
DimProduct[sku_id]         1—*  FactInventoryRisk[sku_id]
DimEntity[store_id]        1—*  FactDemand[store_id]
FactDemand[sku_id, store_id] 1—1 FactInventoryRisk[sku_id, store_id]
FactDemand[sku_id, store_id] 1—1 Recommendations[sku_id, store_id]
```

`executive_kpis` and `system_health` are disconnected snapshot tables (no relationship required).

## Measures (suggested; not deployed)

| Measure | Rule |
| --- | --- |
| Total Recent Units | SUM(FactDemand[total_recent_units]) |
| Critical Stockout Count | COUNTROWS filter stockout_risk_level = "CRITICAL / HIGH" |
| Reorder Review Count | SUM(reorder_triggered) — shelf at/below ROP on the extract |
| TEST MAE | VALUE from executive_kpis[forecast_mae] (do not re-average daily means unweighted) |
| Recommendation Mix | COUNTROWS(Recommendations) by recommended_review |

Do not average WAPE across horizons without weights. Prefer the monitoring snapshot values.

## Refresh assumptions

1. Frozen models are **not** refreshed by Power BI.
2. Re-run `python -m src.bi.exports` after monitoring or extract updates.
3. Label visuals with Data as of / Forecast generated / Monitoring snapshot from `system_health.parquet`.
4. Never mark the dataset as DirectQuery to a production warehouse — these files are snapshots.
