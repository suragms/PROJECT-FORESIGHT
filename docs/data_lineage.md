# Data Lineage — Project FORESIGHT

**Project:** Demand & Inventory Intelligence · **Document:** `docs/data_lineage.md`
**Phase:** 4 (Data Integration & Common Analytical Model) · **Date:** 2026-08-12

This document records the end-to-end lineage of every major table in the
project: where it comes from, how it is transformed, its grain and primary key,
its purpose, and who consumes it.

## 1. End-to-end flow

```text
RAW DATA
   │  data/raw/ (never modified)
   ▼
PHASE 3 — CLEAN DATA                       data/processed/
   │  online_retail_clean/sales/returns/cancellations/invalid
   │  store_master_clean, sku_master_clean, customer_master_clean
   │  calendar_clean, sales_daily_clean, inventory_snapshots_clean
   ▼
PHASE 4 — COMMON ANALYTICAL MODEL (CAM)    data/processed/integrated/
   │  dim_calendar, dim_product, dim_entity, dim_customer
   │  fact_sales, fact_inventory, fact_returns, fact_cancellations
   │  inventory_analytics, customer_analytics, forecast_base
   ▼
FORECAST BASE                              forecast_base.parquet (no lag/rolling features)
   │
   ├──► PHASE 6 FEATURE ENGINEERING  ──►  ML FORECASTING        (src/forecasting.py)
   │                                          │                     │
   └──► INVENTORY ANALYTICS            ──►  INVENTORY RISK       (src/risk_scoring.py)
   │
   └──► POWER BI (star schema) ──► STREAMLIT (dashboard/app.py)
```

Every CAM table preserves the **`source_dataset`** discriminator (`UCI` /
`SYNTHETIC`) so the two source systems are never accidentally mixed.

---

## 2. Dimensions

### 2.1 `dim_calendar`

| Attribute | Value |
|---|---|
| Source | `calendar_clean.csv` (Phase 3, synthetic) extended over the UCI transaction date range |
| Transformation | Synthetic calendar attributes carried as-is; UCI dates get calendar-math attributes (year/month/quarter/day/day-of-week/week-of-year, month-derived season). `is_holiday = 0` and `holiday_name = NULL` for UCI dates — no fabricated holidays. |
| Grain | one row per `date` |
| Primary key | `date` |
| Purpose | Shared date lookup for time-based analysis, seasonality, holidays |
| Downstream | `fact_sales`, `fact_inventory`, `fact_returns`, `fact_cancellations`, feature engineering (calendar features) |

### 2.2 `dim_product`

| Attribute | Value |
|---|---|
| Source | `sku_master_clean.csv` (synthetic) + distinct `StockCode`/`Description` from UCI sales/returns/cancellations |
| Transformation | Synthetic: `product_key = SYN_<sku_id>`, full SKU master attributes. UCI: `product_key = UCI_<StockCode>`, `product_name` = modal description; all other attributes NULL (no fabrication). |
| Grain | one row per `product_key` |
| Primary key | `product_key` |
| Purpose | Product hierarchy (category/sub-category/brand) and replenishment parameters |
| Downstream | All facts via `product_key`; ML feature pipeline via `sku_id` |

### 2.3 `dim_entity`

| Attribute | Value |
|---|---|
| Source | `store_master_clean.csv` (synthetic) + a single UCI channel row |
| Transformation | Synthetic: `entity_id = store_id`, `entity_type = STORE`, physical store attributes. UCI: `entity_id = ONLINE`, `entity_type = CHANNEL`, store fields NULL — no fake stores. |
| Grain | one row per `(source_dataset, entity_id)` |
| Primary key | `source_dataset + entity_id` |
| Purpose | Entity dimension (physical stores + online channel) |
| Downstream | `fact_sales`, `fact_inventory` |

### 2.4 `dim_customer`

| Attribute | Value |
|---|---|
| Source | `customer_master_clean.csv` (synthetic) + identified `Customer ID` from UCI sales |
| Transformation | Synthetic: `customer_key = SYN_<customer_id>`, master attributes. UCI: `customer_key = UCI_<CustomerID>`, `country` = modal country; guest transactions have no row (NULL key) and are flagged in facts. |
| Grain | one row per identified `customer_key` |
| Primary key | `customer_key` |
| Purpose | Customer attributes for customer analytics |
| Downstream | `customer_analytics` |

---

## 3. Facts

### 3.1 `fact_sales`

| Attribute | Value |
|---|---|
| Source | `online_retail_sales.parquet` (UCI) + `sales_daily_clean.parquet` (synthetic) |
| Transformation | UCI: transaction lines aggregated to DATE + SKU (`entity_id = ONLINE`); guests counted in units/revenue but not in `unique_customers`. Synthetic: `store_id → entity_id`, `sku_id → product_key`; `total_revenue → revenue`, `avg_unit_price → average_unit_price`. Returns/cancellations excluded. |
| Grain | `date + source_dataset + entity_id + product_key` |
| Primary key | `date + source_dataset + entity_id + product_key` |
| Purpose | Standardized daily demand fact |
| Downstream | `forecast_base`; Power BI star schema; analytics |

### 3.2 `fact_inventory`

| Attribute | Value |
|---|---|
| Source | `inventory_snapshots_clean.parquet` (synthetic only) |
| Transformation | `store_id → entity_id`, `sku_id → product_key`; Phase 3 REVIEW semantic preserved (`ending = beginning − units_sold`; receipts never re-added); `beginning_inventory_pre_receipts` and `inventory_balance_ok` carried through. UCI gets no records. |
| Grain | `date + source_dataset + entity_id + product_key` |
| Primary key | `date + source_dataset + entity_id + product_key` |
| Purpose | Daily inventory fact for stockout/overstock/coverage analysis |
| Downstream | `inventory_analytics`; inventory risk engine (Phase 10) |

### 3.3 `fact_returns`

| Attribute | Value |
|---|---|
| Source | `online_retail_returns.csv` (UCI returns split) |
| Transformation | Aggregated to DATE + SKU; kept separate from demand. All Phase 3 UCI returns are guest, UK, price-zero lines → revenue impact 0.0 (documented). |
| Grain | `date + source_dataset + entity_id + product_key` |
| Primary key | `date + source_dataset + entity_id + product_key` |
| Purpose | Return analytics independent of demand |
| Downstream | Power BI; returns KPI analysis |

### 3.4 `fact_cancellations`

| Attribute | Value |
|---|---|
| Source | `online_retail_cancellations.csv` (UCI cancellations split) |
| Transformation | Aggregated to DATE + SKU; kept separate from demand. The anomalous cancellation line (Invoice `C496350`, StockCode `M`, qty +1, price 373.57) is preserved — appears as `UCI_M` on 2010-02-01 and is never silently removed. |
| Grain | `date + source_dataset + entity_id + product_key` |
| Primary key | `date + source_dataset + entity_id + product_key` |
| Purpose | Cancellation analytics independent of demand |
| Downstream | Power BI; cancellation-rate analysis |

---

## 4. Analytical tables

### 4.1 `inventory_analytics`

| Attribute | Value |
|---|---|
| Source | `fact_inventory` ⋈ `dim_product` |
| Transformation | Left join on `product_key` to bring category/sub-category/brand, lead time, reorder point, safety stock. **No risk scores computed** — prepares data for Phase 10. |
| Grain | `date + source_dataset + entity_id + product_key` |
| Primary key | `date + source_dataset + entity_id + product_key` |
| Purpose | Inventory risk engine input (via `src/cam_adapter.inventory_analytics_to_legacy_snapshots`) |
| Downstream | Inventory risk engine (Phase 10) |

### 4.2 `customer_analytics`

| Attribute | Value |
|---|---|
| Source | UCI sales (identified customers) + `customer_master_clean.csv` |
| Transformation | UCI: per-customer transaction metrics (count/units/revenue/first/last purchase), country = modal. Synthetic: master attributes only — no customer-grain transactions exist in the processed data, so transaction metrics are NULL (never fabricated). Guest transactions excluded from identified metrics. |
| Grain | one row per `customer_key` |
| Primary key | `customer_key` |
| Purpose | Customer behavior analytics |
| Downstream | Power BI customer view; Phase 5 EDA |

### 4.3 `forecast_base`

| Attribute | Value |
|---|---|
| Source | `fact_sales` (projection) |
| Transformation | Exact downstream column contract; deliberately **no lag / rolling / EWM features** (Phase 6). |
| Grain | `date + source_dataset + entity_id + product_key` |
| Primary key | `date + source_dataset + entity_id + product_key` |
| Purpose | Standardized input for forecasting |
| Downstream | ML forecasting (via `src/cam_adapter.forecast_base_to_legacy_sales`) → Phase 6 feature engineering |

---

## 5. Consumer contracts

| Consumer | Input table | Bridge |
|---|---|---|
| ML Forecasting Engine (`src/forecasting.py`) | `forecast_base` → legacy sales + `dim_product` → legacy SKU master | `src/cam_adapter.py` |
| Inventory Risk Engine (`src/risk_scoring.py`) | `inventory_analytics` → legacy snapshots | `src/cam_adapter.py` |
| Streamlit app (`dashboard/app.py`) | Phase 3 processed files (unchanged in Phase 4) | documented in `src/cam_adapter.check_app_compatibility()` |
| Power BI | CAM star schema (`dim_*` + `fact_*`) | direct |

## 6. Provenance & reproducibility

- Raw files (`data/raw/`) are never read by the CAM and never modified.
- All CAM tables are produced by `src/data_integration.run_integration_pipeline()`
  and reproduced by `notebooks/03_data_integration.ipynb` (executed, 0 errors).
- Quality gates: `docs/integration_quality_report.json/.csv` (11 tables, 0
  duplicate keys, 0 null keys, 0 FK orphans, 11/11 business rules) and the
  re-runnable harness `src/validate_integration.py`.
