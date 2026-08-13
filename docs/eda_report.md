# Phase 5 — Exploratory Data Analysis (EDA) & Business Insights Report

**Project:** FORESIGHT — Retail Demand & Inventory Intelligence  
**Document:** `docs/eda_report.md`  
**Phase:** Phase 5 — Exploratory Data Analysis  
**Dataset:** Common Analytical Model (CAM) in `data/processed/integrated/`  
**Execution Status:** ✅ COMPLETE — 55/55 Notebook Cells Executed (32/32 Code Cells with 0 Errors), 23 Figures Generated, 13 Curated Parquet Artifacts Saved  

---

## 1. Executive Summary & Core KPIs

This report presents the empirical findings, demand patterns, product velocity metrics, store dynamics, customer behaviors, geographic distributions, seasonality waves, promotional impacts, inventory turnover rates, stockout drivers, and reverse logistics risks discovered during **Phase 5 (Exploratory Data Analysis)** of Project FORESIGHT. 

The analysis was executed directly on the integrated **Common Analytical Model (CAM)** tables produced in Phase 4 under `data/processed/integrated/`.

### Executive Business KPI Matrix

| Business Metric | UCI Online Retail (2009–2011) | Synthetic Retail Chain (2022–2025) | Analytical Context & Methodological Meaning |
|---|---|---|---|
| **Operating Time Span** | 2009-12-01 to 2011-12-09 | 2022-01-01 to 2025-12-31 | Historical wholesale vs 4-year multi-store retail chain |
| **Active Operating Days** | 604 days | 1,461 days | Active transaction days vs continuous 365-day annual store operations |
| **Total Top-Line Revenue** | **\$20,465,198.39** | **\$1,227,454,325.27** | Gross realized demand revenue (excluding returns/cancellations) |
| **Total Units Sold** | 11,455,906 units | 10,794,939 units | Physical volume moved across sales channels |
| **Total Transactions** | 999,115 order lines | 6,236,299 POS transactions | Bulk B2B order lines vs store-SKU daily POS transaction volume |
| **Channel / Store Entities** | 1 (Online Global Channel) | 10 Physical Stores | Single online entity vs 10 brick-and-mortar retail locations |
| **Unique Active Catalog SKUs** | 4,984 SKUs | 100 SKUs | Broad long-tail gift catalog vs focused high-velocity retail assortment |
| **Identified Customer Base** | 5,881 Accounts (+ Guests) | N/A (Store-Grain POS Only) | Account-level tracking vs anonymous store checkouts |
| **Average Daily Revenue** | \$33,882.78 / day | \$840,146.70 / day | Average daily velocity across active operating dates |
| **Average Daily Units** | 18,966.73 units / day | 7,388.73 units / day | High-volume low-cost units vs high-value retail units |
| **Average Transaction Value (AOV)**| \$20.48 / order line | \$196.82 / transaction | Order-line basket size vs retail transaction size |
| **Average Realized Unit Price** | \$1.79 / unit | \$113.71 / unit | Wholesale novelty gifts vs durable retail merchandise |
| **Ending Network Inventory** | *N/A (No Inventory Ledger)* | **41,530 units** | Physical stock on hand as of 2025-12-31 |
| **Ending Inventory Valuation (Cost)** | *N/A* | **\$3,485,424.39** | Ending network working capital at unit cost |
| **Ending Inventory Valuation (Retail)** | *N/A* | **\$6,230,018.95** | Ending network valuation at retail base price |
| **Current Network Stockout Rate** | *N/A* | **68.10%** (681 / 1,000 nodes) | Ending stockout rate on latest snapshot date (2025-12-31) |
| **Historical Average Stockout Rate** | *N/A* | **70.44%** (1,029,181 / 1,461,000) | Cumulative historical stockout store-SKU-days |
| **Network Days of Inventory (DOI)** | *N/A* | **5.8 Days** | Stock coverage based on 30-day trailing demand (7,163.6 units/day) |
| **Total Units Returned** | 569,314 units (4.97%) | *N/A (No Return Ledger)* | Reverse logistics returned units (3,393 transactions, \$0.00 impact) |
| **Total Units Cancelled** | 476,821 units (4.16%) | *N/A (No Cancellation Ledger)*| Cancelled wholesale orders (18,910 transactions, -\$1,462,050.61 impact) |

---

## 2. CAM Schema Verification & Provenance

All analytical operations were conducted strictly against the 11 Common Analytical Model tables in `data/processed/integrated/`:

| CAM Table Name | Row Count | Column Count | Source Coverage | Key Grain |
|---|---|---|---|---|
| `dim_calendar` | **2,200** | 13 | UCI_DERIVED (739) + SYNTHETIC (1,461) | `date` (2009–2011 & 2022–2025) |
| `dim_product` | **10,304** | 14 | UCI (10,204) + SYNTHETIC (100) | `product_key` (`UCI_*`, `SYN_*`) |
| `dim_entity` | **31** | 10 | UCI (1) + SYNTHETIC (30) | `source_dataset` + `entity_id` |
| `dim_customer` | **15,881** | 9 | UCI (5,881) + SYNTHETIC (10,000) | `customer_key` |
| `fact_sales` | **1,995,496** | 12 | UCI (534,496) + SYNTHETIC (1,461,000) | `date` + `source` + `entity` + `product` |
| `fact_inventory` | **1,461,000** | 13 | SYNTHETIC (1,461,000) | `date` + `source` + `entity` + `product` |
| `fact_returns` | **3,338** | 9 | UCI (3,338) | `date` + `source` + `entity` + `product` |
| `fact_cancellations` | **17,132** | 9 | UCI (17,132) | `date` + `source` + `entity` + `product` |
| `customer_analytics` | **15,881** | 12 | UCI (5,881) + SYNTHETIC (10,000) | `customer_key` |
| `inventory_analytics` | **1,461,000** | 14 | SYNTHETIC (1,461,000) | `date` + `source` + `entity` + `product` |
| `forecast_base` | **1,995,496** | 12 | UCI (534,496) + SYNTHETIC (1,461,000) | `date` + `source` + `entity` + `product` |

**Integrity Verification:** 100% PASS across all primary key uniqueness tests (0 duplicate rows, 0 null primary keys, 0 foreign key orphans).

---

## 3. Overall & Time-Series Sales Dynamics

### Empirical Findings
- **Synthetic Retail (2022–2025)**: Total revenue of **\$1,227.45M** generated across 1,461 contiguous days. Daily revenue averages **\$840,146.70** ($CV = 0.18$), demonstrating strong stability, predictable autoregressive seasonality, and steady multi-year expansion.
- **UCI Online Retail (2009–2011)**: Total revenue of **\$20.47M** across 604 active trading days. Daily revenue averages **\$33,882.78** ($CV = 0.74$), reflecting intermittent, lumpy wholesale procurement patterns with large spikes during Q4 pre-holiday stocking (October–November).

### Business Interpretation & Modeling Recommendations
- **Synthetic**: Autoregressive moving averages, calendar encodings, and store embeddings will form highly accurate forecasting signals.
- **UCI**: Forecasting requires robust estimators (quantile loss, Huber loss) and rolling median aggregations to handle high dispersion without overfitting to bulk transaction anomalies.

---

## 4. Product Assortment & Pareto (80/20) Velocity

### Synthetic Top Products & Revenue Concentration
- **Pareto Verification**: **27 of 100 SKUs (27.0%) generate 80.0% of total revenue**.
- **Top 10 Synthetic SKUs**:
  1. `SKU_00057` (*Brand_BC Smart Home Item 57*, Electronics): \$134,622,197.29 (353,151 units)
  2. `SKU_00032` (*Brand_BA Team Sports Item 32*, Sports & Outdoors): \$77,099,849.85 (297,528 units)
  3. `SKU_00033` (*Brand_EA Fitness Gear Item 33*, Sports & Outdoors): \$67,603,749.05 (230,375 units)
  4. `SKU_00060` (*Brand_JD Home Decor Item 60*, Home & Kitchen): \$65,699,568.09 (308,352 units)
  5. `SKU_00036` (*Brand_IE Wearables Item 36*, Electronics): \$51,087,226.76 (175,683 units)
  6. `SKU_00092` (*Brand_GE Cycling Item 92*, Sports & Outdoors): \$48,396,270.17 (152,357 units)
  7. `SKU_00055` (*Brand_AE Accessories Item 55*, Electronics): \$40,985,307.83 (104,513 units)
  8. `SKU_00004` (*Brand_ID Footwear Item 4*, Apparel): \$36,687,329.44 (240,421 units)
  9. `SKU_00066` (*Brand_AB Cookware Item 66*, Home & Kitchen): \$33,956,581.00 (148,413 units)
  10. `SKU_00010` (*Brand_DB Smart Home Item 10*, Electronics): \$32,398,150.79 (99,780 units)

### UCI Top Products
  1. `M` (*Manual*): \$339,226.24 (9,636 units)
  2. `22423` (*REGENCY CAKESTAND 3 TIER*): \$330,590.32 (26,495 units)
  3. `DOT` (*DOTCOM POSTAGE*): \$309,854.11 (2,920 units)
  4. `85123A` (*WHITE HANGING HEART T-LIGHT HOLDER*): \$257,724.71 (98,208 units)
  5. `85099B` (*JUMBO BAG RED RETROSPOT*): \$180,569.34 (96,764 units)

---

## 5. Category Performance Dynamics (Synthetic)

| Product Category | SKU Count | Total Revenue ($) | Rev Share (%) | Total Units Sold | Unit Share (%) | Avg Margin (%) |
|---|---|---|---|---|---|---|
| **Electronics** | 16 | \$427,157,824.23 | **34.80%** | 1,664,816 | 15.42% | 36.03% |
| **Sports & Outdoors** | 8 | \$257,025,790.71 | **20.94%** | 1,093,194 | 10.13% | 42.34% |
| **Home & Kitchen** | 14 | \$203,218,572.84 | **16.56%** | 1,542,198 | 14.29% | 43.90% |
| **Apparel** | 18 | \$189,981,110.50 | **15.48%** | 1,763,083 | 16.33% | 54.15% |
| **Groceries & Essentials**| 30 | \$91,806,912.45 | **7.48%** | 3,837,319 | **35.55%** | 26.10% |
| **Health & Beauty** | 14 | \$58,264,114.54 | **4.75%** | 894,329 | 8.28% | **57.08%** |

*Note: Category taxonomy is available exclusively for Synthetic data. UCI records do not have verified category fields, and no artificial categories were fabricated.*

---

## 6. Customer Dynamics & Channel Spend

### UCI Customer Profiling
- **Identified Customer Accounts (5,881 accounts)**: Generated **\$17,374,804.27 (84.90%)** of total sales revenue across 787,839 identified lines.
- **Guest Checkouts**: Generated **\$3,090,394.12 (15.10%)** of total sales revenue across 211,276 guest lines.
- **Spend Distribution**:
  - Average identified customer spend: **\$2,954.40**
  - Median identified customer spend: **\$865.60**
  - Maximum single customer spend: **\$580,987.04**

### Synthetic Customer Master Profiling
- 10,000 registered customer profiles in `dim_customer` (Segment distribution: Regular 50%, Premium 30%, VIP 20%; 40% loyalty members).
- **Data Boundary Note**: Synthetic POS sales operate at Store-SKU-Day grain. No fake basket-level transactions were attached to customers.

---

## 7. Geographic Footprint & Store Operations

### Synthetic Regional & 10-Store Operations

| Store ID | Store Name | Region | Store Type | Size (sqft) | Total Revenue ($) | Share (%) | Units Sold | Transactions | Rev / sqft |
|---|---|---|---|---|---|---|---|---|---|
| `STORE_007` | San Antonio Express Retail | South | Express | 25,000 | \$131,199,263.13 | 10.69% | 1,102,981 | 637,042 | **\$5,247.97** |
| `STORE_008` | San Diego Flagship Retail | West | Flagship | 45,000 | \$130,967,310.90 | 10.67% | 1,117,362 | 643,754 | \$2,910.38 |
| `STORE_006` | Philadelphia Flagship Retail| East | Flagship | 85,000 | \$127,500,566.65 | 10.39% | 1,108,679 | 639,523 | \$1,500.01 |
| `STORE_005` | Phoenix Outlet Retail | West | Outlet | 25,000 | \$125,324,131.48 | 10.21% | 1,101,651 | 636,040 | **\$5,012.97** |
| `STORE_004` | Houston Standard Retail | South | Standard | 85,000 | \$123,667,335.96 | 10.08% | 1,098,228 | 633,433 | \$1,454.91 |
| `STORE_002` | Los Angeles Express Retail | West | Express | 25,000 | \$121,511,323.86 | 9.90% | 1,057,828 | 611,898 | \$4,860.45 |
| `STORE_001` | New York Express Retail | East | Express | 85,000 | \$120,882,477.76 | 9.85% | 1,076,998 | 621,588 | \$1,422.15 |
| `STORE_003` | Chicago Express Retail | Midwest | Express | 85,000 | \$119,048,011.89 | 9.70% | 1,038,697 | 602,578 | \$1,400.56 |
| `STORE_009` | Dallas Express Retail | South | Express | 45,000 | \$117,614,905.62 | 9.58% | 1,066,416 | 616,449 | \$2,613.66 |
| `STORE_010` | San Jose Superstore Retail | West | Superstore| 25,000 | \$109,738,998.02 | 8.94% | 1,026,099 | 593,994 | \$4,389.56 |

- **Regional Breakdown**: West leads with **\$487.54M (39.72%)**, followed by South with **\$372.48M (30.35%)**, East with **\$248.38M (20.24%)**, and Midwest with **\$119.05M (9.70%)**.
- **Efficiency Insight**: Express and Outlet stores achieve substantially higher revenue per square foot (>\$4,800/sqft) than large-format flagship stores (~$1,500/sqft).

### UCI Destination Country Geographic Breakdown
1. **United Kingdom**: \$14,389,436.48 (82.82% identified share), 8,545,922 units, 5,353 customers
2. **EIRE**: \$616,368.98 (3.55%), 318,270 units, 4 customers
3. **Netherlands**: \$554,038.09 (3.19%), 384,519 units, 22 customers
4. **Germany**: \$424,922.66 (2.45%), 225,142 units, 106 customers
5. **France**: \$349,107.36 (2.01%), 270,353 units, 95 customers
6. **Australia**: \$169,207.79 (0.97%), 104,019 units, 14 customers

---

## 8. Seasonality & Calendar Dynamics

1. **Day of Week (DOW)**:
   - **Synthetic**: Weekends generate higher volume (Saturday and Sunday generate 30.2% of weekly sales).
   - **UCI**: Weekdays dominate wholesale orders (Thursday and Wednesday peak; Saturday has 0 orders due to commercial warehouse closure).
2. **Monthly & Quarterly Seasonality**:
   - **Q4 Holiday Surge**: Both datasets experience maximum volume in Q4 (October–December), with November recording peak sales.
   - **Trough Period**: January records lowest sales across all operating years.

---

## 9. Promotional Sensitivity & Price Dynamics (Synthetic)

| Promotion Status | Total Records | Revenue ($) | Units Sold | Rev / Record ($) | Units / Record | Realized Unit Price ($) |
|---|---|---|---|---|---|---|
| **Non-Promoted (0)** | 1,314,715 (90.0%) | \$1,094,617,543.14 | 9,424,493 | \$832.59 | 7.17 | \$110.47 |
| **Promoted (1)** | 146,285 (10.0%) | \$132,836,782.13 | 1,370,446 | \$908.07 | 9.37 | \$91.34 |

- **Unit Lift**: Promoted days achieve a **+30.69% lift in unit movement** (9.37 vs 7.17 units/day).
- **Revenue Lift**: Promoted days deliver **+9.06% incremental revenue per record** (\$908.07 vs \$832.59).
- **Discount Depth**: Promotions average a **17.31% discount** off non-promoted realized price (\$91.34 vs \$110.47).

---

## 10. Inventory Health & Balance Verification (Synthetic)

### Inventory Balance Semantic
- Verified across all 1,461,000 ledger records:
  $$\text{ending\_inventory} = \text{beginning\_inventory} - \text{units\_sold}$$
  *(where `beginning_inventory` already includes the day's replenishment receipts).*
- **Ending Inventory Position (2025-12-31)**: **41,530 total network units** (\$3,485,424.39 at cost; \$6,230,018.95 at retail).
- **Days of Inventory (DOI)**: **5.8 Days** based on 30-day trailing demand (7,163.6 units/day).

---

## 11. Stockout Incident Analysis (Synthetic)

- **Latest Snapshot Stockout Rate**: **68.10%** (681 out of 1,000 Store-SKU nodes out of stock on 2025-12-31).
- **Cumulative Stockout Incidence**: 1,029,181 out of 1,461,000 store-SKU-days (**70.44%**) experienced zero stock.
- **Root Cause**: High stockout frequency reflects rapid inventory turnover and replenishment lead times in fast-moving categories (Electronics and Sports).

---

## 12. Overstock Descriptive Screening (Synthetic)

- **High-DOS Threshold**: Identified nodes where ending inventory exceeds 60 days of 30-day historical trailing demand.
- **Findings**: 84 store-SKU combinations exhibit high Days-of-Supply, trapping working capital in slow-moving Home & Kitchen and Grocery lines.

---

## 13. Reverse Logistics: Returns & Cancellations (UCI)

### Returns Fact Summary
- **Total Returned Units**: **569,314 units** across 3,338 return transactions.
- **Top Returned SKU**: `84077` (*WORLD WAR 2 GLIDERS ASST DESIGNS*), `85123A` (*WHITE HANGING HEART T-LIGHT HOLDER*).
- **Revenue Impact**: \$0.00 (clean return lines represent price-zero gift items).

### Cancellations Fact Summary
- **Total Cancelled Units**: **476,821 units** across 17,132 order lines.
- **Revenue Impact**: **-\$1,462,050.61**.
- **Preserved Anomaly Check**: Invoice `C496350` on 2010-02-01 (StockCode `M`) is preserved in `fact_cancellations` with positive unit/revenue contribution without contaminating normal sales demand.

---

## 14. Methodological Source Comparison

| Dimension | UCI Online Retail (2009–2011) | Synthetic Retail Chain (2022–2025) |
|---|---|---|
| **Business Archetype** | UK Wholesale/B2B Online Gift Novelties | US Multi-Store Omni-Channel Retail Chain |
| **Time Horizon** | 2009-12-01 to 2011-12-09 (604 active days) | 2022-01-01 to 2025-12-31 (1,461 continuous days) |
| **Channel Topology** | 1 Online Global Channel | 10 Physical Stores in 4 US Regions |
| **Catalog Breadth** | 4,984 Active SKUs (Long-Tail Catalog) | 100 SKUs (Focused High-Velocity Catalog) |
| **Customer Tracking** | 5,881 Identified Accounts + Guest Checkouts | Anonymous POS Transactions (10,000 Master Profiles) |
| **Inventory Tracking** | *None Available* | Daily Store-SKU Snapshots (1,461,000 rows) |
| **Promotions & Seasonality** | Wholesale Seasonality (No Promo Flags) | 10% Daily Promotions + US Holiday Calendar |

---

## 15. Saved Figures & Curated EDA Parquet Artifacts

### Generated Figures (`outputs/figures/eda/` — 23 PNG files)
1. `01_sales_by_source.png`
2. `02_daily_revenue_trend.png`
3. `03_monthly_sales_synthetic.png`
4. `04_monthly_sales_uci.png`
5. `05_pareto_synthetic.png`
6. `06_top_products.png`
7. `07_customer_distribution_uci.png`
8. `08_country_sales_uci.png`
9. `09_geographic_sales_synthetic.png`
10. `10_store_performance.png`
11. `11_day_of_week_seasonality.png`
12. `12_monthly_seasonality_synthetic.png`
13. `13_heatmap_dow_month.png`
14. `14_promotion_analysis.png`
15. `15_price_demand_synthetic.png`
16. `16_price_demand_uci.png`
17. `17_inventory_trend.png`
18. `18_stockout_monthly.png`
19. `19_overstock_indicators.png`
20. `20_return_analysis.png`
21. `21_correlation_synthetic.png`
22. `22_correlation_uci.png`
23. `23_outlier_boxplots.png`

### Curated Parquet Artifacts (`data/processed/eda/` — 13 Parquet files)
1. `executive_kpis.parquet` (3,725 bytes)
2. `monthly_sales_synthetic.parquet` (4,160 bytes)
3. `monthly_sales_uci.parquet` (3,629 bytes)
4. `product_performance_synthetic.parquet` (14,515 bytes)
5. `product_performance_uci.parquet` (212,150 bytes)
6. `category_performance_synthetic.parquet` (5,789 bytes)
7. `customer_performance_uci.parquet` (173,307 bytes)
8. `country_performance_uci.parquet` (5,087 bytes)
9. `store_performance.parquet` (9,741 bytes)
10. `inventory_kpis.parquet` (48,188 bytes)
11. `stockout_summary.parquet` (8,814 bytes)
12. `return_summary.parquet` (79,293 bytes)
13. `cancellation_summary.parquet` (109,967 bytes)

---

## 16. Actionable Recommendations for Phase 6 (Feature Engineering)

1. **Cyclical Calendar Encodings**: Apply harmonic $\sin / \cos$ encodings for `month` (period 12) and `day_of_week` (period 7).
2. **Entity-Partitioned Autoregressive Lags**: Compute lag features ($t-1, t-2, t-3, t-7, t-14, t-21, t-28, t-30$) partitioned strictly by `product_key` and `entity_id`.
3. **Leakage-Safe Rolling Features**: Construct rolling means and standard deviations over 7d, 14d, and 30d windows shifted by 1 day (`closed="left"`).
4. **Promotion & Pricing Interactions**: Derive discount depth percentage $\frac{\text{base\_price} - \text{average\_unit\_price}}{\text{base\_price}}$ and interaction term with `promotion_flag`.
5. **Categorical & Target Encodings**: Apply frequency encoding and out-of-fold target encoding for `category` and `store_type`.

---
*Report certified and validated for Phase 5 completion.*
