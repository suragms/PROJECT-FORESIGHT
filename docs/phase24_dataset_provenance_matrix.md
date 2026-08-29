# Phase 24 — Dataset Provenance Matrix

**Generated:** 2026-08-29  
**Rule:** Distinguish official reference source, pipeline input, and archive status.

---

## Matrix

| Dataset | Official Source | Local Pipeline Source | Actually Used by Validated Pipeline | Archive Present in Repo | Status |
|---------|-----------------|----------------------|-------------------------------------|-------------------------|--------|
| **UCI Online Retail II** | [Kaggle — cgrymn/online-retail-ii-uci-dataset](https://www.kaggle.com/datasets/cgrymn/online-retail-ii-uci-dataset) | `data/raw/online_retail_II.csv` | **YES** (research / Phase 17–18) | Kaggle ZIP: **NO** | **VERIFIED PIPELINE INPUT** |
| **Synthetic Retail (local)** | [Kaggle — 10M transactions](https://www.kaggle.com/datasets/mrayyanshehzad/synthetic-retail-dataset-10-million-transactions) | `src/generate_synthetic_retail.py` → `data/raw/sales_daily.parquet` (+ masters, inventory, calendar) | **YES** (Phase 20 production) | Kaggle ZIP: **NO** | **VERIFIED PIPELINE INPUT** (local generator) |
| **Kaggle Synthetic 10M (official)** | Same Kaggle URL | Not downloaded | **NO** | **NO** | **REFERENCE SOURCE ONLY / MANUAL DOWNLOAD REQUIRED** |

---

## UCI — Verified pipeline input

| Field | Value |
|-------|-------|
| File | `data/raw/online_retail_II.csv` |
| Rows | 1,067,371 |
| SHA-256 | `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9` |
| Phase 17 manifest match | **YES** |
| Columns | `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country` |
| Archive | Original Kaggle ZIP **not stored** — this is **SOURCE ARCHIVE NOT PRESERVED**, not pipeline data missing |

---

## Synthetic — Verified pipeline input (local generator)

| Field | Value |
|-------|-------|
| Generator | `src/generate_synthetic_retail.py` (seed=42) |
| Primary sales file | `data/raw/sales_daily.parquet` — 1,461,000 rows |
| Inventory | `data/raw/inventory_snapshots.parquet` — 1,461,000 rows |
| Masters | sku (5,000), store (30), customer (10,000), calendar (1,461) |
| Official Kaggle 10M | **NOT used** — ~10M rows not present |
| Integration policy | **Do not auto-replace** — would require retraining and new validation |

---

## Kaggle credentials (Phase 24 check)

| Check | Result |
|-------|--------|
| `~/.kaggle/kaggle.json` | **NOT FOUND** |
| Automatic official download | **Not performed** |
| Future integration | **OFFICIAL DATASET AVAILABLE FOR FUTURE CONTROLLED INTEGRATION** |

Validated production pipeline remains **unchanged**.

---

## Terminology guide

| Term | Meaning in this project |
|------|-------------------------|
| **Official reference dataset** | Kaggle URL listed in project docs |
| **Pipeline input data** | Files actually read by Phase 17–20 ingestion |
| **Local generated data** | Synthetic data from `generate_synthetic_retail.py` |
| **Archive present** | Original Kaggle ZIP stored under `data/raw_downloads/` |

Do **not** conflate "reference source" with "pipeline input" or claim Kaggle 10M was used without archive evidence.
