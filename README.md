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
│   ├── 02_data_cleaning.ipynb
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

### Step 4: Explore the Notebooks
```bash
jupyter notebook notebooks/01_project_and_data_understanding.ipynb
```

---

## 📈 7. Development Roadmap & Status

- [x] **Phase 1: Business Understanding** — Defined problem statement, KPI framework, and target metrics.
- [x] **Phase 2: Data Collection & Profiling** — Ingested datasets, executed comprehensive data inspection, and completed `01_project_and_data_understanding.ipynb`.
- [ ] **Phase 3: Data Cleaning** — Handling negative quantities/prices, cancellations, and schema normalization.
- [ ] **Phase 4: Data Integration** — Building the Common Analytical Model (CAM).
- [ ] **Phase 5: Exploratory Data Analysis (EDA)** — Sales, product, customer, and temporal patterns.
- [ ] **Phase 6: Feature Engineering** — Lags, rolling statistics, and calendar features.
- [ ] **Phase 7: Baseline Forecasting** — Naive, moving average, and seasonal lag baselines.
- [ ] **Phase 8: Machine Learning Forecasting** — Random Forest, XGBoost, LightGBM.
- [ ] **Phase 9: Model Evaluation & Benchmarking** — MAE, RMSE, MAPE, and model selection.
- [ ] **Phase 10: Inventory Risk Scoring Engine** — Stockout risk, overstock scoring, and days of inventory.
- [ ] **Phase 11: Power BI Dashboard** — 12-page executive BI dashboard.
- [ ] **Phase 12: Streamlit Interactive Application** — Web portal with SKU selector and scenario forecasting.
- [ ] **Phase 13–15: Deployment, Documentation & Final Presentation**.

---

## 👤 Author
**Surag M S**  
*Data Science & Analytics Intern — Project FORESIGHT*
