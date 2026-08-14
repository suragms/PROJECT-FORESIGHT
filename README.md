# Demand & Inventory Intelligence (Project FORESIGHT)
## Retail Demand Forecasting and Inventory Risk Prediction System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📌 1. Project Overview
**Demand & Inventory Intelligence** is an enterprise-grade retail intelligence platform developed under **Project FORESIGHT**. The system analyzes multi-year retail transactions, detects seasonal buying patterns and demand drivers, forecasts future product demand using machine learning, identifies inventory risks (stockouts & overstocking), and generates actionable business recommendations.

---

## 🎯 2. Business Objectives & The 10 Core Questions
The system is built to provide answers to 10 fundamental supply chain and retail questions:
1. **Top Products:** Which products are generating the highest sales volume and revenue?
2. **Bottom Products:** Which products are slow-moving or at risk of becoming deadstock?
3. **Demand Dynamics:** How does customer demand evolve across time and retail channels?
4. **Seasonality:** Which SKUs exhibit weekly, quarterly, or holiday-driven seasonal surges?
5. **Demand Growth:** Which products show positive growth trajectories?
6. **Future Demand:** What is the expected multi-step future demand across forecasting horizons?
7. **Stockout Risk:** Which products are at imminent risk of stocking out before replenishment?
8. **Overstock Risk:** Which products carry excess inventory that inflates carrying costs?
9. **Replenishment Trigger:** Which SKUs have breached their Reorder Point (ROP)?
10. **Actionable Recommendations:** What specific actions should store and inventory managers take?

---

## 📊 3. Datasets Analyzed
The platform leverages two complementary datasets:

### Dataset 1: Online Retail II (UCI Machine Learning Repository)
- **Scale:** 1,067,371 raw transaction line items across 2009–2011.
- **Scope:** UK-based online non-store gift retailer with international wholesale customers.
- **Key Fields:** `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`.

### Dataset 2: Multi-Store Relational Retail Intelligence Dataset
- **Scale:** Relational model covering 30 Stores, 5,000 SKUs, 10,000 Customers, 4-Year Calendar (2022–2025), and 1,461,000 daily sales and inventory tracking records.
- **Core Tables:**
  - `store_master.csv`: Store dimensions (format, region, square footage, city).
  - `sku_master.csv`: Product catalog with pricing, margins, supplier lead times, reorder points, and safety stock.
  - `customer_master.csv`: Customer profiles and segmentation.
  - `calendar.csv`: Full temporal calendar with holiday markers and seasonal quarters.
  - `sales_daily.parquet` / `sales_daily.csv`: Daily store-SKU sales aggregations.
  - `inventory_snapshots.parquet` / `inventory_snapshots.csv`: Daily inventory balance ($End = Beg + Receipts - Sold$).

---

## 🏗️ 4. Project Architecture & Directory Structure

```
Demand-Inventory-Intelligence/
├── data/
│   ├── raw/                      # Ingested datasets (Parquet, CSV)
│   ├── processed/                # Cleaned, standardized analytical datasets
│   └── sample/                   # Samples for rapid testing
├── notebooks/
│   ├── 01_project_and_data_understanding.ipynb   # [COMPLETED & EXECUTED]
│   ├── 02_data_cleaning.ipynb                    # [COMPLETED & EXECUTED]
│   ├── 03_data_integration.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_baseline_forecasting.ipynb
│   ├── 07_ml_forecasting.ipynb
│   ├── 08_model_evaluation.ipynb
│   └── 09_inventory_risk.ipynb
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── data_integration.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── evaluation.py
│   ├── risk_scoring.py
│   ├── generate_synthetic_retail.py
│   └── inspect_datasets.py
├── models/                       # Trained ML models (.joblib)
├── outputs/
│   ├── figures/                  # High-resolution visual artifacts
│   ├── forecasts/                # Demand forecast outputs
│   └── risk_scores/              # Inventory risk scores
├── dashboard/
│   └── app.py                   # Streamlit Intelligence Web Application
├── powerbi/                      # Power BI dashboard templates & data
├── docs/                         # Documentation and profiling summaries
├── requirements.txt              # Environment dependencies
├── README.md
└── .gitignore
```

---

## 🛠️ 5. Technology Stack
- **Languages:** Python 3.12
- **Data Engineering & Manipulation:** Pandas, NumPy, PyArrow (Parquet)
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning & Time Series:** Scikit-learn, XGBoost, LightGBM, Statsmodels
- **Dashboard & Delivery:** Streamlit, Power BI
- **Serialization:** Joblib

---

## 🚀 6. Installation & Quickstart

### Step 1: Clone the Repository
```bash
git clone https://github.com/suragms/Demand-Inventory-Intelligence.git
cd Demand-Inventory-Intelligence
```

### Step 2: Set Up Virtual Environment & Dependencies
```bash
# Using uv (Recommended)
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt

# Or using standard venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run Data Profiling & Verification
```bash
python src/inspect_datasets.py
```

### Step 4: Run the Data Cleaning Pipeline
```bash
python src/data_cleaning.py        # end-to-end cleaning -> data/processed/ + docs/data_quality_report.json
jupyter nbconvert --to notebook --execute notebooks/02_data_cleaning.ipynb --output-dir notebooks
```

### Step 5: Explore the Notebooks
```bash
jupyter notebook notebooks/01_project_and_data_understanding.ipynb
```

### Forecast serving (Phase 12)

Final models: UCI h=1 frozen Phase 8 LightGBM; SYNTHETIC h=1 hurdle; both sources h=3/7/14/30 direct LightGBM. See `docs/final_forecasting_report.md`.

```bash
python src/validate_phase12.py
python -m pytest tests -q
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
streamlit run dashboard/forecast_analytics.py
python -m src.forecasting.batch_forecast --help
python -m src.monitoring.run_monitoring
```

Docs: `docs/api_documentation.md`, `docs/deployment_guide.md`, `docs/monitoring_guide.md`.

Authentication is not included in this academic/reference implementation.

---

## 📈 7. Development Roadmap & Status

- [x] **Phase 1: Business Understanding** — Defined problem statement, KPI framework, and target metrics.
- [x] **Phase 2: Data Collection & Profiling** — Ingested datasets, executed comprehensive data inspection, and completed `01_project_and_data_understanding.ipynb`.
- [x] **Phase 3: Data Cleaning** — Full data-quality engineering pipeline in `src/data_cleaning.py` + `notebooks/02_data_cleaning.ipynb`. See [Section 7.1](#71-phase-3-data-cleaning--quality-engineering-report).
- [x] **Phase 4: Data Integration** — Building the Common Analytical Model (CAM) in `src/data_integration.py`.
- [ ] **Phase 5: Exploratory Data Analysis (EDA)** — Sales, product, customer, and temporal patterns.
- [x] **Phase 6: Feature Engineering** — Lags, rolling statistics, and calendar features in `src/feature_engineering.py`.
- [x] **Phase 7: Baseline Forecasting** — Naive, moving average, and seasonal lag baselines in `src/forecasting.py`.
- [x] **Phase 8: Machine Learning Forecasting** — LightGBM selected; 57/57 validation. Frozen benchmark.
- [x] **Phase 9: Stability & residual analysis** — 146/146 PASS.
- [x] **Phase 10: Experimental improvements** — hurdle, direct horizon, intervals, HPO.
- [x] **Phase 11: Final model selection** — 140/140 PASS; READY WITH MONITORING.
- [x] **Phase 12: Production packaging** — inference package, FastAPI, monitoring, forecast dashboard (academic/reference).
- [x] **Streamlit executive application** — `dashboard/app.py` (inventory + scenario UI).
- [ ] **Power BI Dashboard / cloud deployment** — not executed in this repository.

---

## 📋 7.1 Phase 3: Data Cleaning & Quality Engineering Report

**Status: COMPLETED & EXECUTED** — full pipeline in [src/data_cleaning.py](src/data_cleaning.py), executed notebook in [notebooks/02_data_cleaning.ipynb](notebooks/02_data_cleaning.ipynb). Machine-readable report: `docs/data_quality_report.json` / `.csv`.

### Online Retail II (UCI) — status **REVIEW**
| Step | Result |
|---|---|
| Original rows | 1,067,371 |
| Exact duplicates removed | 34,335 (3.22%) — raw file untouched |
| Final rows | 1,033,036 |
| Missing `Customer ID` | 243,007 (22.77%) — treated as **guest transactions**, kept for sales analysis, excluded only from customer-level analysis |
| Missing `Description` | 4,382 — 4,019 recovered via StockCode; 363 labelled `"Unknown Product"` |
| Cancellations (`C`-prefix) | 19,104 (1.79%) → `online_retail_cancellations.csv` |
| Returns (negative qty) | 3,393 (0.32%) → `online_retail_returns.csv` |
| Invalid accounting lines | 6 (Adjust bad debt) → `online_retail_invalid.csv` |
| Special transactions (zero/neg price) | 6,019 — flagged, **not deleted** |
| Derived columns | `transaction_type`, `is_guest_transaction`, `price_category`, `is_special_transaction`, `description_recovered` |

### Synthetic relational data — status **PASS** (all tables), Inventory **REVIEW**
- Store, SKU, Customer, Calendar, Sales Daily all pass schema/numeric/categorical validation with zero issues.
- `sales_daily` grain `(date, store_id, sku_id)` unique; `total_revenue == units_sold × avg_unit_price` holds 100%.
- **Inventory equation:** raw `beginning_inventory` already *includes* the day's receipts. Canonical form `End = Beg_pre_receipts + Receipts − Sold` fails on 122,208 rows (8.36%) as written in the generator; `End = Beg − Sold` holds 100%, proving the semantic. Fix: added documented derived column `beginning_inventory_pre_receipts` + `inventory_balance_ok` flag — **values were reported, never silently overwritten**.

### Referential integrity & outliers
- All orphan checks (sales→store/sku, inventory→store/sku) pass with 0 orphans; no master records invented.
- Outliers investigated via IQR & Z-score (e.g. `Quantity`, `ending_inventory`, `on_order_qty`) — **reported, not removed** (extreme but legitimate high-volume/high-value retail).

### Outputs
`data/processed/`: `online_retail_clean.csv`, `online_retail_sales.csv`, `online_retail_returns.csv`, `online_retail_cancellations.csv`, `online_retail_invalid.csv`, `store_master_clean.csv`, `sku_master_clean.csv`, `customer_master_clean.csv`, `calendar_clean.csv`, `sales_daily_clean.parquet`, `inventory_snapshots_clean.parquet`.
`outputs/figures/`: 6 cleaning figures (transaction types, quantity distribution, price categories, inventory balance, outlier boxplots, before/after rows).

### Data-science rules upheld
No raw data modified · no fabricated values · no silent deletion (every removal logged with reason) · returns/cancellations preserved · missing Customer ID ≠ invalid · legitimate outliers kept · no leakage · reproducible `run_cleaning_pipeline()` · reusable functions, no repeated code.

---

## 👤 Author
**Surag M S**  
*Data Science & Analytics Intern — Project FORESIGHT*
