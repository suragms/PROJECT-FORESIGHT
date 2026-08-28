# Phase 19 — Risk Validation

**Demand source:** PHASE19_FORECAST

**Supported horizon:** 6 weeks


## Decision Grid Validation

- REORDER NOW -> CRITICAL stockout: PASS

- HEALTHY -> LOW/OPTIMAL: PASS

- WoS consistency: PASS


## Stress Tests

| Scenario | Stockout | Overstock | Action | Expected | Pass |
|----------|----------|-----------|--------|----------|------|
| severe_stockout | CRITICAL | OPTIMAL | REORDER NOW | REORDER NOW | True |
| moderate_stockout | MEDIUM | OPTIMAL | WATCH / VOLATILE | WATCH / VOLATILE | True |
| healthy_inventory | LOW | OPTIMAL | HEALTHY | HEALTHY | True |
| moderate_overstock | LOW | MODERATE | WATCH / VOLATILE | WATCH / VOLATILE | True |
| severe_overstock | LOW | SEVERE | MARKDOWN / CLEAR | MARKDOWN / CLEAR | True |
| high_volatility | CRITICAL | OPTIMAL | REORDER NOW | REORDER NOW | True |

## Sample Critical SKUs

```json
[
  {
    "sku_id": "SKU_00008",
    "forecast_weekly_demand": 439.4815878186868,
    "lead_time_demand": 878.9631756373736,
    "on_hand_units": 0,
    "on_order_units": 990,
    "stockout_risk_level": "CRITICAL",
    "action": "REORDER NOW",
    "sales_at_risk": 12331.853354192352
  },
  {
    "sku_id": "SKU_00013",
    "forecast_weekly_demand": 337.74080937767417,
    "lead_time_demand": 675.4816187553483,
    "on_hand_units": 3,
    "on_order_units": 700,
    "stockout_risk_level": "CRITICAL",
    "action": "REORDER NOW",
    "sales_at_risk": 76065.98508803977
  },
  {
    "sku_id": "SKU_00028",
    "forecast_weekly_demand": 281.58411071708775,
    "lead_time_demand": 1126.336442868351,
    "on_hand_units": 0,
    "on_order_units": 1080,
    "stockout_risk_level": "CRITICAL",
    "action": "REORDER NOW",
    "sales_at_risk": 203776.78924374204
  },
  {
    "sku_id": "SKU_00035",
    "forecast_weekly_demand": 176.53116169994482,
    "lead_time_demand": 529.5934850998344,
    "on_hand_units": 0,
    "on_order_units": 430,
    "stockout_risk_level": "CRITICAL",
    "action": "REORDER NOW",
    "sales_at_risk": 36552.54234159057
  },
  {
    "sku_id": "SKU_00050",
    "forecast_weekly_demand": 177.46893547494278,
    "lead_time_demand": 709.8757418997711,
    "on_hand_units": 0,
    "on_order_units": 680,
    "stockout_risk_level": "CRITICAL",
    "action": "REORDER NOW",
    "sales_at_risk": 189153.490186613
  }
]
```


## Rupee Impact

- Sales at risk: 15075716.8

- Locked capital: 0.0
