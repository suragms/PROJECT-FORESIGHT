# Phase 17 — Data Quality Report

## UCI Online Retail II

| Check | Result |
|-------|--------|
| Raw rows | 1,067,371 |
| Invalid dates | Checked, dropped if NaT |
| Null Invoice | 0 |
| Null StockCode | 0 |
| Null Description | 4,382 |
| Null Customer ID | 243,007 (22.8%) |
| Negative quantity rows | 22,950 |
| Zero price rows | Present, excluded from sales |
| Negative price rows | Present, excluded from sales |
| Cancellation rows (C-prefix) | 19,494 — excluded from demand |
| Duplicates removed | From valid sales after filtering |
| Clean weekly demand rows | 198,395 |
| Weekly SKUs | 4,917 |

### Cleaning Decisions

| Problem | Decision | Reason | Impact |
|---------|----------|--------|--------|
| C-prefix invoices | Excluded from demand | Cancellations are not demand | Removed ~19K rows |
| Negative quantities (non-cancel) | Excluded | Returns/adjustments are not demand | Removed ~3.5K rows |
| Zero/negative prices | Excluded | Invalid for revenue calculation | Minor |
| Missing Customer ID | Kept for demand; excluded from customer analysis | Guest transactions are valid demand | Retains 22.8% of transactions |
| Missing Description | Kept | Description not needed for demand forecasting | No impact |

## Synthetic Dataset

| Check | Result |
|-------|--------|
| Sales rows | 1,461,000 |
| Stores | 10 |
| SKUs | 100 |
| Negative quantities | 0 |
| Null store_id | 0 |
| Null sku_id | 0 |
| Duplicate grain (date+store+sku) | 0 |
| Inventory rows | 1,461,000 |
| Weekly demand rows | 21,000 |
| Date range | 2022-01-01 to 2025-12-31 |

### Cleaning Decisions

| Problem | Decision | Reason | Impact |
|---------|----------|--------|--------|
| No issues found | No cleaning required | Data generated with seed=42, clean by construction | None |
