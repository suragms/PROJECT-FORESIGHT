# Phase 14 — Business workflow report

Layers are never mixed:

| Layer | Meaning |
| --- | --- |
| MODEL FORECAST | Registered joblib output (point + P10/P90 on h=1) |
| INVENTORY RISK | Scores from `inventory_risk_matrix.parquet` |
| BUSINESS RECOMMENDATION | Human-facing interpretation; **not** a purchase order |

## Forecast → inventory for the simulation SKU

The production simulation forecasted **UCI / ONLINE / UCI_10135**.

| Field | Value |
| --- | --- |
| forecast | 10.749200294243758 |
| P10 / P90 | 0.9803840782755803 / 12.034108979147309 |
| ending inventory | NOT AVAILABLE |
| lead time | NOT AVAILABLE |
| safety stock | NOT AVAILABLE |
| reorder point | NOT AVAILABLE |
| stockout risk | NOT AVAILABLE |
| overstock risk | NOT AVAILABLE |
| recommendation | NOT AVAILABLE |

UCI product keys are not rows in the synthetic store-SKU risk extract. Phase 14 does not invent ending inventory, lead time, or ROP for that SKU.

## Decision-support scenarios (from the risk extract, not auto-executed)

| Scenario | Status | Example | Interpretation |
| --- | --- | --- | --- |
| A Stockout | PASS | STORE_001 / SKU_00001, ending inventory 0, CRITICAL / HIGH | Inventory may be insufficient relative to expected demand. Review replenishment requirements. Not an automatic purchase order. |
| B Overstock | PASS (moderate fallback) | STORE_009 / SKU_00004, ending inventory 677, MODERATE OVERSTOCK | Potential excess inventory exposure. Review demand and inventory alignment. Severe overstock was **NOT AVAILABLE** on this extract. |
| C Stable | PASS | STORE_001 / SKU_00007, ending inventory 50, LOW / SAFE + OPTIMAL | No exceptional intervention indicated. |
| D Increasing demand | PASS | `demand_growth_30` on the UCI sample features | Review future supply capacity and inventory requirements. This is a **feature**, not a warehouse action. |
| E High uncertainty | PASS | P10 0.98 / P90 12.03 around prediction 10.75 | Additional business review recommended. |

All five are **decision-support outputs**. `autonomous_decision=false`.

## Original 10 questions

| # | Question | Data source | Method | Output | Business interpretation | Limitations |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Top products | risk matrix `total_recent_revenue` | historical aggregation | SKU ids in Phase 13 evidence | Ranked revenue, not a forecast | 1000-row extract, not a live census |
| 2 | Bottom products | same | historical aggregation | bottom SKU ids | Slow movers / review deadstock | same extract |
| 3 | Demand dynamics | `src/risk_scoring.py` + CAM sales | channel/region aggregates | implemented | Historical dynamics | not live POS |
| 4 | Seasonality | calendar-joined sales | monthly / holiday uplift | implemented | Seasonal pattern, not live | historical only |
| 5 | Demand growth | year-over-year units in risk_scoring | growth ranking | implemented | Historical growth | not a causal model |
| 6 | Future demand | `final_predictions.parquet` + API | Phase 11 frozen models | 957,949 forecast rows; API pred 10.75 for UCI_10135 h=1 | MODEL FORECAST | frozen models; no retraining |
| 7 | Stockout risk | risk matrix `stockout_risk_level` | scoring rules | 733 CRITICAL / HIGH on 1000-row extract | INVENTORY RISK | extract, not warehouse telemetry |
| 8 | Overstock risk | `overstock_risk_level` | scoring rules | 0 severe; moderate rows exist | INVENTORY RISK | severe overstock not on extract |
| 9 | Replenishment review | `reorder_triggered` | ROP flag | 0 true flags on this extract | Review only; no PO sent | flag may under-trigger |
| 10 | Actionable recommendations | `src/risk_scoring.py` text | derived narrative | present | BUSINESS RECOMMENDATION | decision support only |

## Dashboard

Forecast analytics loads Phase 11 parquet, shows dataset and horizon, labels point vs P10/P90, and marks monitoring as a **file snapshot**, not real-time telemetry. Inventory risk and recommendations are labelled on the executive app (`dashboard/app.py`). The executive app’s in-session trainer does not write `models/final`.
