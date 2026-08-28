# Phase 18 — Risk Engine Validation

## Risk Architecture Verification

| Component | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Demand source | FORECAST (not historical lookback) | FORECAST | PASS |
| Inventory source | `inventory_snapshots.parquet` latest week | Week 2025-12-30 | PASS |
| Lead time | From `sku_master.lead_time_days` ÷ 7 | Applied | PASS |
| Safety stock | From `sku_master.safety_stock` | Applied | PASS |
| Reorder point | From `sku_master.reorder_point` | Applied | PASS |
| UCI risk scoring | NOT AVAILABLE (no inventory) | Documented as NOT_AVAILABLE | PASS |

## Decision Grid Validation

| Action | Trigger Condition | Implementation Match | Status |
|--------|------------------|---------------------|--------|
| REORDER NOW | `stockout_risk_level == CRITICAL` | All REORDER NOW rows have CRITICAL stockout | PASS |
| MARKDOWN / CLEAR | `overstock_risk_level == SEVERE` | 0 SEVERE rows, 0 MARKDOWN rows — consistent | PASS |
| WATCH / VOLATILE | MEDIUM stockout OR MODERATE overstock | All WATCH rows match this condition | PASS |
| HEALTHY | Neither CRITICAL nor SEVERE | All HEALTHY rows verified LOW/OPTIMAL | PASS |

**Decision grid consistency: PASS**

## Internal Consistency Checks

| Check | Result |
|-------|--------|
| `weeks_of_supply = on_hand / forecast_demand` | PASS (tolerance 1%) |
| Risk scores ∈ [0, 100] | PASS |
| No negative financial metrics | PASS |
| `lead_time_demand = forecast_weekly_demand × lead_time_weeks` | PASS |
| `inventory_position = on_hand + on_order` | PASS |

## Sample High-Risk SKU Walkthrough (CRITICAL Stockout)

### SKU_00008

| Field | Value |
|-------|-------|
| Forecast weekly demand | 444.28 units |
| Lead time (weeks) | 2 |
| Lead-time demand | 888.57 units |
| On-hand units | **0** |
| On-order units | 990 |
| Inventory position | 990 |
| Safety stock | 9 |
| Weeks of supply | 0.0 |
| Stockout risk level | **CRITICAL** |
| Overstock risk level | OPTIMAL |
| Recommended action | **REORDER NOW** |
| Sales at risk (₹) | ₹12,467 |

**Explanation:** On-hand is zero (immediate stockout risk). Safety stock threshold (9 units) is already breached. Although on-order units (990) exceed lead-time demand (889), the zero on-hand means the system flags CRITICAL immediately. Action REORDER NOW is appropriate.

### SKU_00028

| Field | Value |
|-------|-------|
| Forecast weekly demand | 281.28 units |
| Lead time (weeks) | 4 |
| Lead-time demand | 1,125.13 units |
| On-hand units | **0** |
| On-order units | 1,080 |
| Inventory position | 1,080 |
| Safety stock | 7 |
| Weeks of supply | 0.0 |
| Stockout risk level | **CRITICAL** |
| Recommended action | **REORDER NOW** |
| Sales at risk (₹) | ₹203,558 |

**Explanation:** On-hand is zero with a 4-week lead time. Even with 1,080 on order, the gap between lead-time demand (1,125) and on-order (1,080) is 45 units, indicating a potential shortfall. High sales-at-risk due to elevated base price.

## Rupee Impact Verification

| Metric | Value | Source | Status |
|--------|-------|--------|--------|
| Total sales at risk | From forecast × lead_time × base_price | `sku_master.base_price` (actual) | VERIFIED |
| Total locked capital | ₹0.00 (no overstock detected) | `sku_master.cost_price` (actual) | VERIFIED |
| UCI financial impact | NOT AVAILABLE | UCI has no cost/inventory data | DOCUMENTED |

All monetary calculations use actual `base_price` and `cost_price` fields from `sku_master.csv`. No values were fabricated.
