# Phase 16 — Dataset Audit

**Project:** Demand & Inventory Intelligence · **Phase:** 16
**Date:** 2026-08-19

---

## 1. UCI Online Retail II — Full Audit

### File-Level

| Property | Value |
|----------|-------|
| Filename | online_retail_II.csv |
| Extension | .csv |
| Size | 94,850,204 bytes |
| SHA-256 | `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9` |
| Compression | None (plain CSV) |
| Row count | 1,067,371 |
| Column count | 8 |

### Schema-Level

| Column | Type | Nullable | Nulls |
|--------|------|----------|-------|
| Invoice | string | No | 0 |
| StockCode | string | No | 0 |
| Description | string | Yes | 4,382 |
| Quantity | int64 | No | 0 |
| InvoiceDate | string | No | 0 |
| Price | float64 | No | 0 |
| Customer ID | float64 | Yes | 243,007 |
| Country | string | No | 0 |

- Unique keys: Invoice + line number (implicit, no explicit line key)
- Duplicate rows: Present (same invoice can have multiple identical lines)
- Missing values: Description (4,382), Customer ID (243,007)

### Business-Level

| Concept | Value |
|---------|-------|
| Grain | Invoice line (transaction) |
| Date range | 2009-12-01 to 2011-12-09 |
| Product identifier | StockCode (5,305 unique) |
| Store identifier | None (online-only retailer) |
| Customer identifier | Customer ID (5,942 identified; 243,007 guest rows) |
| Sales quantity | Quantity (positive = sale, negative = return/cancellation) |
| Revenue | Derived: Quantity × Price |
| Prices | Price (unit price) |
| Promotions | NOT AVAILABLE FROM SOURCE |
| Inventory | NOT AVAILABLE FROM SOURCE |
| Lead time | NOT AVAILABLE FROM SOURCE |
| Reorder point | NOT AVAILABLE FROM SOURCE |

---

## 2. Current Synthetic Dataset — Full Audit

### sales_daily.parquet

| Property | Value |
|----------|-------|
| Rows | 1,461,000 |
| Columns | 9 |
| SHA-256 | `4c521a50f15e1a3c4634bc372613af369c4f0310e583f48df7502999c3c472ab` |
| Size | 6,259,981 bytes |
| Date range | 2022-01-01 to 2025-12-31 |
| Stores used | 10 |
| SKUs used | 100 |
| Grain | date + store_id + sku_id (daily aggregation) |

Schema: date, store_id, sku_id, units_sold (int32), total_revenue (float64), avg_unit_price (float64), transaction_count (int32), unique_customers (int32), promotion_flag (int64)

### inventory_snapshots.parquet

| Property | Value |
|----------|-------|
| Rows | 1,461,000 |
| Columns | 9 |
| SHA-256 | `674e65ba321fc55265dc51bb851f377df94a0177fe5a50eb55c0175140ea5972` |
| Size | 4,979,408 bytes |
| Date range | 2022-01-01 to 2025-12-31 |
| Stores used | 10 |
| SKUs used | 100 |
| Grain | date + store_id + sku_id (daily snapshot) |

Schema: date, store_id, sku_id, beginning_inventory (int32), receipts (int32), units_sold (int32), ending_inventory (int32), stockout_flag (int32), on_order_qty (int32)

### sku_master.csv

| Property | Value |
|----------|-------|
| Rows | 5,000 |
| Columns | 12 |
| Key columns | sku_id, sku_name, category, sub_category, brand, cost_price, base_price, weight_kg, supplier_id, lead_time_days, reorder_point, safety_stock |

### store_master.csv

| Property | Value |
|----------|-------|
| Rows | 30 |
| Columns | 8 |
| Key columns | store_id, store_name, city, state, region, store_type, store_size_sqft, opening_date |

### calendar.csv

| Property | Value |
|----------|-------|
| Rows | 1,461 |
| Columns | 12 |
| Key columns | date, year, month, quarter, day, day_of_week, day_name, is_weekend, is_holiday, holiday_name, season, week_of_year |

### customer_master.csv

| Property | Value |
|----------|-------|
| Rows | 10,000 |
| Columns | 5 |
| Key columns | customer_id, customer_name, customer_segment, loyalty_member, signup_date |

---

## 3. Provenance Determination (Section 12)

### Current synthetic data origin

**Classification: C — Independently generated synthetic data**

Evidence:
1. `src/generate_synthetic_retail.py` uses `numpy.random.seed(42)` with hardcoded parameters
2. Store IDs follow `STORE_001` to `STORE_030` format (not Kaggle format)
3. SKU IDs follow `SKU_00001` to `SKU_05000` format (not Kaggle format)
4. Customer IDs follow `CUST_000001` to `CUST_010000` format (not Kaggle format)
5. The Kaggle Synthetic Retail 10M describes "10 million transactions" — this repository has 1,461,000 sales rows
6. No Kaggle download artifacts (ZIP files, metadata) exist in the repository

**CURRENT SYNTHETIC DATA = REPOSITORY-GENERATED SYNTHETIC DATA**
**KAGGLE EQUIVALENCE = NOT PROVEN**
