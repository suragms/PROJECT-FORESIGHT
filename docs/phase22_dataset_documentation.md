# Phase 22 — Dataset Documentation

## Overview

PROJECT FORESIGHT uses two datasets with distinct roles. **Do not misrepresent either dataset.**

---

## UCI Online Retail II

### Official Reference Source

https://www.kaggle.com/datasets/cgrymn/online-retail-ii-uci-dataset

| Aspect | Status |
|--------|--------|
| Pipeline input | `data/raw/online_retail_II.csv` — **1,067,371 rows** |
| SHA-256 | `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9` |
| Original Kaggle ZIP in repo | **NO** — archive not preserved; pipeline CSV verified |

### Purpose

Historical UK online retail transaction data for benchmarking and research.

### Role in Project

**RESEARCH CANDIDATE** — evaluated through Phases 17–18; **not promoted to production**.

### Data Grain

Invoice-day product level (ONLINE source)

### Usage

- Baseline comparison and candidate evaluation
- Phase 17/18 promotion gate analysis
- Demonstrates cross-dataset generalization challenges

### Limitations

- Different grain and structure from Synthetic production pipeline
- WAPE significantly higher than Synthetic (64.19% candidate vs 91.63% baseline in Phase 17)
- Not compatible with weekly SKU production contract
- **Not used for Phase 20 production forecasting**

### Key Fields

Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country

---

## Synthetic Retail Dataset

### Official Reference Source (not used directly by current pipeline)

**Kaggle Synthetic Retail Dataset — 10 Million Transactions**  
https://www.kaggle.com/datasets/mrayyanshehzad/synthetic-retail-dataset-10-million-transactions

| Aspect | Status |
|--------|--------|
| Official Kaggle archive in repository | **NOT INCLUDED** — manual download required |
| Used by validated production pipeline | **NO** — local generator used instead |

### Current Pipeline Implementation

**Local source:** `src/generate_synthetic_retail.py` (NumPy seed=42)

**Pipeline inputs:** `data/raw/sales_daily.parquet`, `inventory_snapshots.parquet`, master tables, `calendar.csv`

### Purpose

Multi-store relational retail dataset with stores, SKUs, customers, calendar, daily sales, and inventory.

### Role in Project

**PRODUCTION PROMOTED FORECASTING SOURCE** — Phase 20 model trained and validated on this data at weekly SKU grain.

### Data Grain

- Raw: daily store × SKU
- Production: **weekly SKU-level** aggregation

### Usage

- Feature engineering (45-feature contract)
- Rolling-origin backtest validation
- Production forecast generation
- Risk engine scoring (100 SKUs in production matrix)

### Core Tables

| Table | Description |
|-------|-------------|
| `store_master.csv` | Store dimensions |
| `sku_master.csv` | Product catalog, pricing, lead times |
| `customer_master.csv` | Customer profiles |
| `calendar.csv` | Holidays, seasons |
| `sales_daily.parquet` | Daily sales |
| `inventory_snapshots.parquet` | Daily inventory |

### Limitations

- Synthetic data — patterns may not fully represent real retail complexity
- 100 SKUs in production risk matrix (not full catalog)
- Holiday bias partially unresolved in Nov–Dec

---

## Comparison Summary

| Aspect | UCI | Synthetic |
|--------|-----|-----------|
| Status | Research Candidate | Production Source |
| Grain | Invoice-day | Weekly SKU |
| Production Model | No | Yes (phase20_synthetic_lightgbm) |
| Phase 20 API | Rejected | Accepted |
