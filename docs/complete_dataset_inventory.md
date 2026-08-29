# Project Foresight Dataset Inventory

**Generated:** 2026-08-29T15:23:25.937015+00:00

## Dataset Sources

### UCI Online Retail II

Source: https://www.kaggle.com/datasets/cgrymn/online-retail-ii-uci-dataset

**Download status:** ALREADY AVAILABLE

**Kaggle original archive present:** False

### Synthetic Retail Dataset

Source (official Kaggle target): https://www.kaggle.com/datasets/mrayyanshehzad/synthetic-retail-dataset-10-million-transactions

**Download status:** MANUAL DOWNLOAD REQUIRED

**Kaggle original archive present:** False

> **OBSERVED FACT:** Repository synthetic files under `data/raw/` are from `src/generate_synthetic_retail.py`, not the Kaggle 10M download.

---

## UCI — Files discovered

- `data/raw/uci_online_retail_ii/extracted_files/online_retail_II.csv` — 94,850,204 bytes — SHA-256 `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9` — rows=1067371

### UCI schema (observed)

- Rows: **1067371**
- Date range: **2009-12-01 07:45:00 → 2011-12-09 12:50:00**
- Unique invoices: **53628**
- Unique stock codes: **5305**
- Unique customers (non-null): **5942**
- Countries: **43**
- Negative quantities: **22950**
- Cancellation invoices (prefix C): **19494**

#### Field presence (OBSERVED FACT)

- transaction_invoice_id: column `Invoice` — PRESENT
- product_sku: column `StockCode` — PRESENT
- product_description: column `Description` — PRESENT
- quantity: column `Quantity` — PRESENT
- transaction_date: column `InvoiceDate` — PRESENT
- unit_price: column `Price` — PRESENT
- customer_identifier: column `Customer ID` — PRESENT
- country: column `Country` — PRESENT

---

## Synthetic (repository local) — Files discovered

- `data/raw/synthetic_retail/extracted_files/calendar.csv` — 87,019 bytes — SHA-256 `a7cc1a7e3fb3acbc75854c5cc38f453912cc8380d4fe33bd291008dc6384d933` — rows=1461
- `data/raw/synthetic_retail/extracted_files/customer_master.csv` — 510,945 bytes — SHA-256 `0d0a02b9dd665c1dd2e310a351cb0ba33dcdc1d8accec7a7b5b6036de1c071e5` — rows=10000
- `data/raw/synthetic_retail/extracted_files/inventory_snapshots.csv` — 67,830,275 bytes — SHA-256 `fd9e7f9ef13be3aad928638e4cde5a9627bcb42adb7eb1bd2772bf16f5bfcf07` — rows=1461000
- `data/raw/synthetic_retail/extracted_files/inventory_snapshots.parquet` — 4,979,408 bytes — SHA-256 `674e65ba321fc55265dc51bb851f377df94a0177fe5a50eb55c0175140ea5972` — rows=1461000
- `data/raw/synthetic_retail/extracted_files/sales_daily.csv` — 76,131,903 bytes — SHA-256 `4485b2317a3752136e66c5679c1128dc8a942781f9d31350d6a2c5be36555a3e` — rows=1461000
- `data/raw/synthetic_retail/extracted_files/sales_daily.parquet` — 6,259,981 bytes — SHA-256 `4c521a50f15e1a3c4634bc372613af369c4f0310e583f48df7502999c3c472ab` — rows=1461000
- `data/raw/synthetic_retail/extracted_files/sku_master.csv` — 560,108 bytes — SHA-256 `e94ed3a23542ff4c59232a36883a0fafb63a1c0c2f6d0518c27e585eb30c1c9a` — rows=5000
- `data/raw/synthetic_retail/extracted_files/store_master.csv` — 2,500 bytes — SHA-256 `9d729d2d727f5dafaff53b37dbfefb8fc4bb7a9d1ddcfa477e742188b2bb1d5e` — rows=30

### Synthetic schema summary (OBSERVED FACT — local generator)

- Sales rows: **1461000**
- Sales columns: `['date', 'store_id', 'sku_id', 'units_sold', 'total_revenue', 'avg_unit_price', 'transaction_count', 'unique_customers', 'promotion_flag']`
- Date range: **2022-01-01 00:00:00 → 2025-12-31 00:00:00**
- Stores in sales: **10**
- SKUs in sales: **100**
- SKU master rows: **5000**
- Customer master IDs: **10000**
- Categories: **6**
- Inventory rows: **1461000**

### Relationships (supported by keys)

- `sales_daily.sku_id` → `sku_master.sku_id` — unmatched=0 (match_rate=1.0)
- `sales_daily.store_id` → `store_master.store_id` — unmatched=0 (match_rate=1.0)

### Kaggle 10M status

**MANUAL DOWNLOAD REQUIRED** — place archive in `data/raw_downloads/synthetic_retail/original_archive/`.

---

## Phase usage comparison

```json
{
  "phase17": {
    "uci_source": "data/raw/online_retail_II.csv",
    "synthetic_source": "data/raw/sales_daily.parquet (+ inventory/sku/store)",
    "manifest": "data/phase17/ingestion_manifest.json"
  },
  "phase19_20_21_22": {
    "note": "Use Phase 17/19 processed weekly features and Phase 20 production artifacts derived from repository synthetic + UCI pipelines \u2014 not Kaggle 10M archive."
  },
  "kaggle_10m_used_in_phases": false,
  "uci_kaggle_zip_present": false
}
```

## Integrity

- Dataset separation: **PASS** (UCI and synthetic paths remain separate)
- Source files preserved: **PASS** (copies only; originals in `data/raw/` untouched as masters)
- Frozen models: **PASS** / Phase20 **PASS**

## Extraction layout

```
data/raw_downloads/
  uci_online_retail_ii/original_archive/   # place Kaggle ZIP here
  synthetic_retail/original_archive/       # place Kaggle ZIP here
data/raw/
  online_retail_II.csv                     # pipeline UCI source
  sales_daily.* / inventory_* / masters    # pipeline synthetic (local)
  uci_online_retail_ii/extracted_files/    # inventory copies
  synthetic_retail/extracted_files/        # inventory copies + PROVENANCE
```

