# Project Foresight — Final Dataset Verification Report

**Audit date:** 2026-08-29  
**Script:** `src/dataset_inventory_and_validation.py` (executed this audit)

---

## Official sources

| Dataset | Kaggle URL |
|---------|------------|
| UCI Online Retail II | https://www.kaggle.com/datasets/cgrymn/online-retail-ii-uci-dataset |
| Synthetic Retail 10M | https://www.kaggle.com/datasets/mrayyanshehzad/synthetic-retail-dataset-10-million-transactions |

---

## UCI Online Retail II

| Check | Result |
|-------|--------|
| Kaggle credentials | **NOT FOUND** |
| Original Kaggle ZIP in `data/raw_downloads/uci_online_retail_ii/original_archive/` | **Absent** |
| Pipeline CSV `data/raw/online_retail_II.csv` | **Present** |
| Extracted inventory copy | `data/raw/uci_online_retail_ii/extracted_files/online_retail_II.csv` |
| Rows (full line scan) | **1,067,371** |
| SHA-256 | `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9` |
| Matches Phase 17 manifest | **YES** (verified) |
| Columns (observed) | `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country` |
| Date range | 2009-12-01 → 2011-12-09 |
| Unique invoices | 53,628 |
| Unique stock codes | 5,305 |
| Unique customers | 5,942 |
| Countries | 43 |

**Status: PARTIAL** — pipeline-ready UCI CSV verified; official Kaggle archive not present.

---

## Synthetic Retail Dataset

| Check | Result |
|-------|--------|
| Kaggle 10M archive | **NOT DOWNLOADED** (manual download required) |
| Repository synthetic data | **Present** under `data/raw/` |
| Provenance | **`src/generate_synthetic_retail.py`** (local generator, seed=42) |
| **NOT** the Kaggle 10M dataset | Confirmed by Phase 16 equivalence report + row counts |

### Local synthetic files (verified)

| File | Rows | SHA-256 (prefix) |
|------|------|------------------|
| `sales_daily.parquet` | 1,461,000 | `4c521a50…` |
| `inventory_snapshots.parquet` | 1,461,000 | `674e65ba…` |
| `sku_master.csv` | 5,000 | `e94ed3a2…` |
| `store_master.csv` | 30 | `9d729d2d…` |
| `customer_master.csv` | 10,000 | `0d0a02b9…` |
| `calendar.csv` | 1,461 | `a7cc1a7e…` |

Sales date range: 2022-01-01 → 2025-12-31  
Stores in sales: 10 | SKUs in sales: 100  
Key relationships: sales→sku_master match rate **1.0**; sales→store_master match rate **1.0**

**Status: PARTIAL** — complete local synthetic relational dataset for pipeline; official Kaggle 10M **not extracted**.

---

## Dataset separation

| Rule | Status |
|------|--------|
| UCI and SYNTHETIC never merged in training | **PASS** — separate `source_dataset` columns, separate phase paths |
| Phase 17 ingestion manifest | UCI + Synthetic both **PASS** |
| Phase 20 production model | **SYNTHETIC only** |
| UCI routed to production Phase 20 | **Rejected** — `phase20_api_adapter.py` raises for UCI source |

---

## Phase usage comparison

| Phase | Source used | Notes |
|-------|-------------|-------|
| 17 | UCI CSV + local synthetic parquet | Manifest SHA matches |
| 19 | Phase 19 synthetic features (56 cols) | Holiday hardening |
| 20 | `phase19_synthetic_lightgbm` lineage → promoted copy | Hash lineage verified in e2e |
| 21 | Phase 20 outputs + feature contract | Observability only |
| 22 | Phase 20 + 21 adapters | Executive bundle |

Unused official files: Kaggle ZIP archives for both datasets.

---

## Integrity artifacts

- `docs/complete_dataset_inventory.md`
- `docs/dataset_source_integrity.json`
- `docs/dataset_inventory_detail.json`

**Frozen models / Phase 17–22 validated outputs:** unchanged (verified during inventory run).
