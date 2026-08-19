# Phase 16 — Schema Contract

**Project:** Project FORESIGHT — Demand & Inventory Intelligence  
**Phase:** 16  

This document records every schema layer from raw ingestion through the BI export tier.
All fields are sourced from existing artifacts. No fields are invented.

---

## 1. Raw Schema

### 1.1 `data/raw/online_retail_II.csv` — UCI Online Retail II

| Field | Type | Nullable | Grain | Business Meaning |
|---|---|---|---|---|
| `Invoice` | int64 | No | transaction | Invoice number. C-prefixed = cancellation |
| `StockCode` | string | No | transaction | Product / stock-keeping code |
| `Description` | string | Yes | transaction | Product description (modal used in dim_product) |
| `Quantity` | int64 | No | transaction | Units; negative = return/cancellation |
| `InvoiceDate` | datetime | No | transaction | Transaction timestamp |
| `Price` | float64 | No | transaction | Unit price (GBP) |
| `Customer ID` | float64 (nullable) | Yes (22.77 %) | transaction | Customer identifier; guest = null |
| `Country` | string | No | transaction | Shipping country |

**Missing concepts (NOT AVAILABLE — not fabricated):**
- Inventory on-hand, on-order quantity, lead time, reorder point, safety stock, promotion flag, promotion event

---

### 1.2 `data/raw/sales_daily.parquet` — Synthetic daily sales

| Field | Type | Nullable | Grain | Source | Business Meaning |
|---|---|---|---|---|---|
| `date` | string (YYYY-MM-DD) | No | date + store + SKU | Generator | Observation date |
| `store_id` | string | No | date + store + SKU | Generator | Store identifier (STORE_001 – STORE_010 in data) |
| `sku_id` | string | No | date + store + SKU | Generator | SKU identifier (SKU_00001 – SKU_00100 in data) |
| `units_sold` | int32 | No | date + store + SKU | Generator simulation | Units sold that day |
| `total_revenue` | float64 | No | date + store + SKU | Generator | `units_sold × avg_unit_price` |
| `avg_unit_price` | float64 | No | date + store + SKU | Generator | Simulated daily average price |
| `transaction_count` | int32 | No | date + store + SKU | Generator | Synthetic transaction count |
| `unique_customers` | int32 | No | date + store + SKU | Generator | Synthetic unique customer count |
| `promotion_flag` | int64 | No | date + store + SKU | Generator | 1 = promotion active that day |

---

### 1.3 `data/raw/inventory_snapshots.parquet` — Synthetic inventory

| Field | Type | Nullable | Grain | Source | Business Meaning |
|---|---|---|---|---|---|
| `date` | string | No | date + store + SKU | Generator | Observation date |
| `store_id` | string | No | date + store + SKU | Generator | Store identifier |
| `sku_id` | string | No | date + store + SKU | Generator | SKU identifier |
| `beginning_inventory` | int32 | No | date + store + SKU | Generator | Opening stock that day |
| `receipts` | int32 | No | date + store + SKU | Generator | Units received from replenishment |
| `units_sold` | int32 | No | date + store + SKU | Generator | Units sold (matches sales_daily) |
| `ending_inventory` | int32 | No | date + store + SKU | Generator | Closing stock (shelf position) |
| `stockout_flag` | int32 | No | date + store + SKU | Generator | 1 = stockout occurred |
| `on_order_qty` | int32 | No | date + store + SKU | Generator | Units in-transit / on order |

---

### 1.4 `data/raw/sku_master.csv`

| Field | Type | Nullable | Grain | Source | Business Meaning |
|---|---|---|---|---|---|
| `sku_id` | string | No | SKU | Generator | SKU identifier (5,000 rows; 100 used in transactions) |
| `sku_name` | string | No | SKU | Generator | Synthetic product name |
| `category` | string | No | SKU | Generator | Product category (6 categories) |
| `sub_category` | string | No | SKU | Generator | Product sub-category |
| `brand` | string | No | SKU | Generator | Synthetic brand |
| `cost_price` | float64 | No | SKU | Generator | Cost price (USD) |
| `base_price` | float64 | No | SKU | Generator | Base selling price (USD) |
| `weight_kg` | float64 | No | SKU | Generator | Weight |
| `supplier_id` | string | No | SKU | Generator | Synthetic supplier ID |
| `lead_time_days` | int64 | No | SKU | Generator | Supplier lead time in days |
| `reorder_point` | int64 | No | SKU | Generator | Reorder point (units) — computed as `lead_time_demand + safety_stock` |
| `safety_stock` | int64 | No | SKU | Generator | Safety stock (units) — computed as `1.65 × √lead_time × demand_sd` |

---

## 2. Cleaned Schema (`data/processed/`)

After Phase 3 cleaning:

| Table | Key additions / transformations |
|---|---|
| `online_retail_clean` | Cancellations separated; negative quantities flagged; missing `Customer ID` → guest flag |
| `sku_master_clean` | Whitespace trimmed; nulls imputed where documented |
| `sales_daily_clean` | Date parsed; type conversions |
| `inventory_snapshots_clean` | Date parsed; type conversions |
| `calendar_clean` | Date parsed |

UCI `promotion_flag`, `ending_inventory`, `on_order_qty` remain **null** after cleaning — not fabricated.

---

## 3. Integrated (CAM) Schema (`data/processed/integrated/`)

| Table | Grain | Key fields | Notes |
|---|---|---|---|
| `dim_calendar` | one per date | `date`, `year`, `month`, `quarter`, `week_of_year`, `is_holiday`, `season` | UCI dates get calendar-math attributes; `is_holiday=0`, `holiday_name=NULL` for UCI — not fabricated |
| `dim_product` | one per `product_key` | `product_key`, `product_name`, `category`, `sub_category`, `brand`, `lead_time_days`, `reorder_point`, `safety_stock` | SYN: `SYN_<sku_id>` — full master; UCI: `UCI_<StockCode>` — all master attributes NULL |
| `dim_entity` | one per `(source_dataset, entity_id)` | `entity_id`, `entity_type`, `city`, `state`, `region` | SYN: physical stores; UCI: `entity_id=ONLINE`, store attrs NULL |
| `dim_customer` | one per customer key | `customer_key`, `customer_segment`, `loyalty_member` | Guest UCI transactions have no row; keyed by NULL |
| `fact_sales` | date + entity + product | `units_sold`, `revenue`, `transaction_count`, `unique_customers`, `promotion_flag`, `source_dataset` | `promotion_flag` NULL for UCI |
| `fact_inventory` | date + entity + product | `beginning_inventory`, `ending_inventory`, `on_order_qty`, `stockout_flag`, `source_dataset` | Only populated for SYNTHETIC |
| `forecast_base` | date + entity + product | all fact_sales fields + dim calendar/product attrs | Input to feature engineering |

---

## 4. Forecasting Schema (`data/processed/features/forecast_features.parquet`)

| Field | Type | Nullable | Source | Grain | Notes |
|---|---|---|---|---|---|
| `date` | datetime | No | integration | date + entity + product | Observation date |
| `source_dataset` | string | No | integration | — | UCI or SYNTHETIC — never mixed in training |
| `entity_id` | string | No | dim_entity | — | ONLINE (UCI) or STORE_XXX (SYN) |
| `product_key` | string | No | dim_product | — | UCI_StockCode or SYN_sku_id |
| `units_sold` | float | No | fact_sales | — | Target variable |
| `revenue` | float | No | fact_sales | — | — |
| `promotion_flag` | float | Yes | fact_sales | — | NULL for all UCI rows |
| `ending_inventory` | float | Yes | fact_inventory | — | NULL for all UCI rows |
| `on_order_qty` | float | Yes | fact_inventory | — | NULL for all UCI rows |
| `stockout_flag` | float | Yes | fact_inventory | — | NULL for all UCI rows |
| `units_sold_lag_N` | float | Yes | computed | — | Lag features (1,2,3,7,14,21,28,30); computed strictly on past observations |
| `rolling_mean_N` | float | Yes | computed | — | Rolling statistics; no future leakage |
| `month_sin`, `month_cos` | float | No | calendar | — | Cyclical encoding |
| `dow_sin`, `dow_cos` | float | No | calendar | — | Cyclical encoding |
| `is_holiday` | int | No | calendar | — | 0 for UCI dates |
| `split` | string | No | pipeline | — | TRAIN / TEST — enforced temporally |

---

## 5. Inventory Risk Schema (`outputs/risk_scores/inventory_risk_matrix.parquet`)

| Field | Type | Nullable | Source | Business Meaning |
|---|---|---|---|---|
| `store_id` | string | No | inventory snapshot | Store identifier |
| `sku_id` | string | No | inventory snapshot | SKU identifier |
| `ending_inventory` | int | No | inventory snapshot | Current shelf stock |
| `on_order_qty` | int | No | inventory snapshot | In-transit quantity |
| `avg_daily_demand` | float | No | historical average | Rolling historical average — NOT forecast-derived |
| `std_daily_demand` | float | No | historical average | Demand variability |
| `effective_daily_demand` | float | No | derived | max(avg_daily_demand, 0.05) |
| `lead_time_days` | int | No | sku_master | Supplier lead time |
| `safety_stock` | int | No | sku_master | Safety stock |
| `reorder_point` | int | No | sku_master | Catalog reorder point |
| `dynamic_rop` | float | No | derived | `effective_daily_demand × lead_time + safety_stock` |
| `days_of_supply` | float | No | derived | `ending_inventory / effective_daily_demand` |
| `stockout_risk_score` | float | No | derived | Composite score |
| `stockout_risk_level` | string | No | derived | LOW/SAFE, CRITICAL/HIGH |
| `overstock_risk_score` | float | No | derived | Composite score |
| `overstock_risk_level` | string | No | derived | OPTIMAL, MODERATE OVERSTOCK, SEVERE OVERSTOCK |
| `reorder_triggered` | bool | No | derived | `ending_inventory <= reorder_point` (shelf stock only — Phase 16 correction verified) |
| `rop_position_triggered` | bool | No | derived | `inventory_position (ending + on_order) <= reorder_point` |
| `recommended_reorder_qty` | int | No | derived | Based on ending_inventory deficit to target max |
| `replenishment_note` | string | No | derived | Decision-support note; not a PO |
| `extract_note` | string | No | BI layer | "1000-row reference extract" |

**Note:** Risk engine uses historical average demand, NOT frozen forecast predictions. This is a documented implementation difference from the Zidio specification.

---

## 6. BI Export Schema (`outputs/bi/`)

See `outputs/bi/schema.json` for the machine-readable version.

| File | Required Columns |
|---|---|
| `executive_kpis.parquet` | `layer`, `decision_support_only`, `forecast_mae`, `forecast_rmse`, `forecast_wape`, `forecast_bias`, `inventory_n_rows`, `inventory_stockout_critical_high`, `inventory_extract_note` |
| `product_demand.parquet` | `sku_id`, `store_id`, `total_recent_units`, `demand_share`, `growth_class`, `stockout_risk_level`, `demand_rank_label`, `extract_note` |
| `forecast_performance.parquet` | `forecast_date`, `source_dataset`, `horizon`, `actual`, `forecast`, `error`, `absolute_error`, `p10`, `p90`, `grain` |
| `inventory_risk.parquet` | `sku_id`, `store_id`, `stockout_risk_level`, `overstock_risk_level`, `reorder_triggered`, `risk_matrix_cell`, `extract_note` |
| `recommendations.parquet` | `sku_id`, `recommended_review`, `evidence`, `reason`, `confidence_limitation`, `autonomous_decision` |
| `system_health.parquet` | `uci_h1_hash`, `synthetic_h1_hash`, `uci_hash_matches_phase12`, `synthetic_hash_matches_phase12`, `live_data`, `monitoring_snapshot` |

All BI exports are file snapshots — not live data. `live_data = False` is recorded in `system_health.parquet`.

---

## 7. Dashboard Schema

| Dashboard | Input tables | Key columns |
|---|---|---|
| `dashboard/executive_intelligence.py` | All `outputs/bi/*.parquet` | See BI export schema above |
| `dashboard/app.py` | `outputs/risk_scores/inventory_risk_matrix.parquet`, `data/processed/forecasts/final/final_predictions.parquet` | forecast_date, actual, prediction, stockout_risk_level, reorder_triggered |
