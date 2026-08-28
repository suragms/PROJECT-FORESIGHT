# Phase 17 — Risk Scoring Report

## Methodology

Phase 17 replaces the Phase 10 historical-demand risk approach with **forecast-driven risk scoring**.

| Component | Phase 10 (Production) | Phase 17 (Candidate) |
|-----------|----------------------|---------------------|
| Demand input | avg_daily_demand from last 30 days | forecast_weekly_demand from LightGBM |
| Inventory | Latest snapshot ending_inventory | Latest weekly snapshot ending_inventory |
| Lead time | sku_master.lead_time_days | Same (converted to weeks) |
| Safety stock | sku_master.safety_stock | Same |
| Reorder point | sku_master.reorder_point | Same |

## Stockout Risk Logic

1. `on_hand_units <= 0` → score = 100 (CRITICAL)
2. `on_hand_units < safety_stock` → score = 80 (CRITICAL)
3. `weeks_of_supply < lead_time_weeks` → score = 60 (MEDIUM)
4. `inventory_position <= reorder_point` → score = 40 (MEDIUM)
5. Otherwise → LOW

## Overstock Risk Logic

1. `weeks_of_supply > 2 × horizon` → score = 80 (SEVERE)
2. `weeks_of_supply > horizon` → score = 50 (MODERATE)
3. Amplified if `forecast_weekly_demand < 0.5`

## Decision Grid

| Action | Condition |
|--------|-----------|
| REORDER NOW | stockout_risk_level = CRITICAL |
| MARKDOWN / CLEAR | overstock_risk_level = SEVERE |
| WATCH / VOLATILE | stockout = MEDIUM or overstock = MODERATE |
| HEALTHY | All other |

## Results (SYNTHETIC only — UCI has no inventory)

| Metric | Value |
|--------|-------|
| Total SKUs scored | 100 |
| REORDER NOW | 15 |
| MARKDOWN / CLEAR | 0 |
| WATCH / VOLATILE | 82 |
| HEALTHY | 3 |
| Demand source | FORECAST |

## UCI Risk Status

**NOT AVAILABLE** — UCI Online Retail II does not provide inventory, lead time, reorder point, or safety stock. No values were fabricated.
