# Phase 15 — Business intelligence report

Phase 15 adds a **read-only** business intelligence layer on frozen FORESIGHT outputs. Models, features, and SHA-256 hashes were not changed.

The layer answers four decision-support questions:

| Question | Source |
| --- | --- |
| WHAT HAPPENED? | Extract demand (`total_recent_units` / `total_recent_revenue`) and held-out TEST actuals |
| WHY DID IT HAPPEN? | Seasonality of TEST actuals, documented growth split, existing risk labels |
| WHAT IS LIKELY TO HAPPEN? | Frozen registry forecasts and P10/P90 interval companions |
| WHAT SHOULD THE BUSINESS REVIEW? | Recommendation mapping (not purchase orders) |

## Inputs consumed (not recomputed as a new model)

* `data/processed/forecasts/final/final_predictions.parquet` (957,949 TEST rows)
* `outputs/risk_scores/inventory_risk_matrix.parquet` (**1000-row reference extract**)
* `outputs/monitoring/*.json`
* `docs/phase11_metadata.json` and `docs/final_model_registry.json`

Full forecast grain is **not** duplicated into `outputs/bi/`. Daily means and product-level h=1 aggregates are exported instead.

## KPI layer (measured)

Demand KPIs are computed on the inventory extract (100 SKUs in the extract):

| KPI | Value | Notes |
| --- | --- | --- |
| total recent units | 215,418 | extract `total_recent_units` sum |
| average daily demand | 6.949 | extract mean |
| demand volatility (CV) | 1.2706 | mean std / mean avg daily demand |
| n products (extract) | 100 | not the full catalog |
| growth on extract | Insufficient Evidence | extract has no independent YoY window |

Forecast KPIs are copied from the monitoring snapshot on held-out TEST actuals (not unknown future dates):

| KPI | Value |
| --- | --- |
| n with actuals | 957,949 |
| MAE | 8.8075 |
| RMSE | 37.7012 |
| WAPE | 73.4244 |
| bias | -0.2285 |
| interval coverage | NOT AVAILABLE |

P10/P90 exist on h=1 rows only (`lower_bound`/`upper_bound` missing 76.351% overall). Coverage is **not** claimed.

Inventory KPIs (extract only):

| KPI | Value |
| --- | --- |
| rows | 1,000 |
| CRITICAL / HIGH stockout | 733 |
| MEDIUM (REORDER) | 188 |
| LOW / SAFE | 79 |
| SEVERE OVERSTOCK | 0 |
| MODERATE OVERSTOCK | 1 |
| reorder_triggered (shelf at/below ROP) | 887 |
| LOW/SAFE and OPTIMAL | 78 |
| mean days of supply | 3.5399 |

Operations KPIs are in-process API counters from the last monitoring write (includes Phase 14 negative tests). They are **not** live APM:

| KPI | Value |
| --- | --- |
| request_count | 38 |
| error_rate | 0.4211 |
| mean_latency_ms | 30.83 |
| auth_failures | 2 |
| retraining | disabled |

## Product ranking

Top and bottom products use the **existing** extract ranking field `total_recent_revenue` (same methodology as Phase 13 business validation). Bottom ranks are labelled **LOW DEMAND**, never “bad product”.

| Rank label | n |
| --- | --- |
| TOP_REVENUE | 10 |
| LOW DEMAND | 10 |
| MID | 980 |

## Growth rule (explicit)

On h=1 TEST actuals, each series is split at its median `forecast_date`:

* n_hist < 10 or n_recent < 10 or missing rate → **Insufficient Evidence**
* |rate| < 0.05 → **Stable**
* rate ≥ 0.05 → **Growing**
* rate ≤ −0.05 → **Declining**

Join to the extract is SYNTHETIC only (`SYN_` prefix stripped). UCI warehouse fields remain **NOT AVAILABLE**.

Measured extract growth_class after join: Stable 469, Declining 346, Growing 185.

## Seasonality

Monthly means of TEST actuals (not a fitted seasonal model):

| Dataset | monthly CV | weak_or_uncertain (CV < 0.10) | peak month | low month |
| --- | --- | --- | --- | --- |
| SYNTHETIC | 0.0429 | True | 8 | 10 |
| UCI | 0.3534 | False | 10 | 9 |

SYNTHETIC monthly variation is weak. UCI TEST dates cover a short 2011 window; treat peaks as sample description, not a causal calendar model.

## Forecast vs actual

`outputs/bi/forecast_performance.parquet` is daily mean Actual / Forecast / P10 / P90 by dataset and horizon (952 rows). Concepts are labelled separately. Aggregate MAE/RMSE/WAPE/bias are **not** recomputed on empty future actuals; they come from `accuracy_monitoring_report.json`.

## Risk matrix overlay

Primary risk remains `stockout_risk_level` / `overstock_risk_level` from the existing scorer.

An executive 2×2 is added with a documented strict-median split:

```
                    DEMAND
                 Low            High
Inventory Low    Normal         Stockout Review
Position  High   Overstock Review  Critical Review
```

`ending_inventory` median on the extract is **0** (682 zero-stock rows). A `>= median` rule would mark every row inventory-high. The overlay therefore uses **strict greater than** the median so zero stock is Low inventory. This overlay does **not** replace the 733 CRITICAL / HIGH labels.

Measured overlay occupancy after the strict split:

| Cell | n |
| --- | --- |
| Normal | 422 |
| Stockout Review | 260 |
| Critical Review | 234 |
| Overstock Review | 84 |

## Recommendations (decision support)

Every row includes Evidence, Reason, Recommended Review, Confidence / Limitation. `autonomous_decision` is always false.

Measured mapping on the 1,000 extract rows (stockout label takes precedence):

| Recommended review | n |
| --- | --- |
| Review replenishment | 733 |
| Review forecast uncertainty | 266 |
| Review inventory exposure | 1 |

No purchase orders, supplier messages, or financial commitments are generated.

## Power BI preparation

Parquet exports under `outputs/bi/` are schema-documented in `docs/powerbi_data_model.md`. **Power BI Desktop was not required to validate them, and no Power BI workspace was deployed.**
