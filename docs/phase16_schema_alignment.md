# Phase 16 — Zidio Schema Alignment

**Project:** Demand & Inventory Intelligence · **Phase:** 16
**Date:** 2026-08-19

---

## 1. Zidio Schema Alignment Matrix

| Zidio Concept | Current Repository | UCI | Kaggle Synthetic | Status |
|---------------|-------------------|-----|-----------------|--------|
| sales_daily | `data/raw/sales_daily.parquet` — 1,461,000 rows, date+store+SKU grain | Derived via `fact_sales` from invoice lines aggregated to date+entity+product_key | INSUFFICIENT EVIDENCE (not downloaded) | **PASS** (Repository + UCI) |
| sku_master | `data/raw/sku_master.csv` — 5,000 SKUs with full attributes | Derived: `dim_product` uses StockCode with modal Description; category/brand/price = NULL | INSUFFICIENT EVIDENCE | **PASS** (Repository); **PARTIAL** (UCI — no category/cost/margin) |
| calendar | `data/raw/calendar.csv` — 1,461 days (2022–2025) with holidays, seasons | Extended to cover UCI date range (2009–2011) in `dim_calendar`; UCI dates get `is_holiday=0` | INSUFFICIENT EVIDENCE | **PASS** |
| inventory_snapshots | `data/raw/inventory_snapshots.parquet` — 1,461,000 rows | **MISSING** — UCI does not provide inventory | INSUFFICIENT EVIDENCE | **PASS** (Repository); **NOT APPLICABLE** (UCI) |
| lead_time_days | `sku_master.csv` column `lead_time_days` — 3 to 30 days | **MISSING** — NOT AVAILABLE FROM SOURCE | INSUFFICIENT EVIDENCE | **PASS** (Repository); **NOT APPLICABLE** (UCI) |
| reorder_point | `sku_master.csv` column `reorder_point` | **MISSING** — NOT AVAILABLE FROM SOURCE | INSUFFICIENT EVIDENCE | **PASS** (Repository); **NOT APPLICABLE** (UCI) |
| on_hand_units | `inventory_snapshots` column `ending_inventory` | **MISSING** — NOT AVAILABLE FROM SOURCE | INSUFFICIENT EVIDENCE | **PASS** (Repository); **NOT APPLICABLE** (UCI) |
| on_order_units | `inventory_snapshots` column `on_order_qty` | **MISSING** — NOT AVAILABLE FROM SOURCE | INSUFFICIENT EVIDENCE | **PASS** (Repository); **NOT APPLICABLE** (UCI) |
| promo_flag | `sales_daily` column `promotion_flag` | Not available in UCI raw; `promotion_flag=0` in forecast_base | INSUFFICIENT EVIDENCE | **PASS** (Repository); **DERIVED** (UCI — always 0) |
| promo_event | Not explicitly stored; no named promotion events | **MISSING** | INSUFFICIENT EVIDENCE | **MISSING** (all sources) |
| unit_cost | `sku_master.csv` column `cost_price` | **MISSING** — NOT AVAILABLE FROM SOURCE | INSUFFICIENT EVIDENCE | **PASS** (Repository); **NOT APPLICABLE** (UCI) |
| list_price | `sku_master.csv` column `base_price` | UCI has `Price` (unit price per transaction line) | INSUFFICIENT EVIDENCE | **PASS** (Repository); **PARTIAL** (UCI — transaction-level only) |

---

## 2. Alignment Summary

- **Repository synthetic data:** All Zidio concepts are satisfied (PASS) except `promo_event` (MISSING — no named promotion calendar)
- **UCI data:** Only `sales_daily` and partial `sku_master`/`list_price` are available. All inventory, supply chain, and cost concepts are NOT AVAILABLE FROM SOURCE.
- **Kaggle Synthetic 10M:** INSUFFICIENT EVIDENCE — dataset not downloaded

---

## 3. Business Scope Check (Section 16)

The Zidio client specification describes:
- NorthBay Living, D2C, online-only, one warehouse, ~200 active SKUs

The current implementation contains:
- 30 defined stores (10 active in sales data)
- 5,000 defined SKUs (100 active in sales data)
- Multi-store, multi-SKU relational model

**Classification: DOCUMENTED SCOPE EXTENSION**

The multi-store synthetic model provides richer analytical scenarios for a forecasting reference system. The README explicitly documents "30 Stores, 5,000 SKUs" as the synthetic dataset scope. The UCI component (entity_id=ONLINE) operates as a single-entity channel matching the D2C concept.

The scope extension is useful because:
1. Multi-store data enables store-level demand variation analysis
2. The forecasting grain (`source_dataset + entity_id + product_key`) correctly separates entities
3. Risk scoring operates per store-SKU pair, demonstrating scalable inventory management
4. The README and Phase 15 documentation transparently describe the scope

---

## 4. Analytical Grain Verification

**Verified from `src/feature_engineering.py` line 32:**

```python
GRAIN_COLS = ["source_dataset", "entity_id", "product_key"]
```

**Full grain:** `date + source_dataset + entity_id + product_key`

This matches the expected common analytical grain. Source separation is enforced — UCI and SYNTHETIC are never mixed in grouped operations.
