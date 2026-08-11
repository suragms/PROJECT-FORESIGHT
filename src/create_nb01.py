"""
Script to create notebooks/01_project_and_data_understanding.ipynb
"""

import json
import os

def create_notebook_01():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Demand & Inventory Intelligence System\n",
                    "## Retail Demand Forecasting and Inventory Risk Prediction\n",
                    "---\n",
                    "### Phase 1 & 2: Business Understanding & Comprehensive Data Profiling\n",
                    "\n",
                    "**Author:** Data Science & Analytics Intern  \n",
                    "**Project:** Project FORESIGHT — Retail Demand & Inventory Intelligence  \n",
                    "**Date:** August 2026  \n",
                    "\n",
                    "---\n",
                    "### 1. Executive Summary & Business Problem Statement\n",
                    "\n",
                    "Modern retail supply chains operate under fine margins where inventory imbalances directly impact profitability:\n",
                    "- **Stockouts (Understocking)** lead to immediate lost revenue, damaged customer loyalty, and loss of market share to competitors.\n",
                    "- **Overstocking** ties up valuable working capital, inflates holding/warehousing costs, increases markdown frequency, and causes inventory obsolescence/spoilage.\n",
                    "\n",
                    "**Core Project Objective:**\n",
                    "Build an end-to-end retail intelligence system that ingests historical transactional and inventory data, models product velocity, extracts time-series and demand drivers, produces accurate future demand forecasts, and quantifies stockout & overstock risks into interpretable business recommendations."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. The 10 Core Business Questions to Answer\n",
                    "1. **What products are selling the most?** (Top revenue & volume drivers / Pareto 80-20 analysis)\n",
                    "2. **What products are selling the least?** (Slow-moving & deadstock candidates)\n",
                    "3. **How does demand change over time?** (Trend lines, growth trajectories, and macroeconomic cycles)\n",
                    "4. **Which products show seasonal patterns?** (Weekly spikes, quarterly peaks, holiday surges like Q4 Black Friday/Christmas)\n",
                    "5. **Which products are likely to have increasing demand?** (Upward momentum SKUs)\n",
                    "6. **What is the expected future demand?** (Accurate multi-step point and range forecasts via ML models)\n",
                    "7. **Which products are at risk of stockout?** (Forecast Demand > Available Inventory + Lead Time Pipeline)\n",
                    "8. **Which products may be overstocked?** (Current Inventory >> Forecast Demand + Safety Stock)\n",
                    "9. **Which products require immediate replenishment?** (Inventory Level $\\le$ Reorder Point)\n",
                    "10. **What actionable recommendations should management take?** (Automated prescriptive action engine)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Setup Environment & Core Libraries\n",
                    "import os\n",
                    "import sys\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import plotly.express as px\n",
                    "import plotly.graph_objects as go\n",
                    "import warnings\n",
                    "\n",
                    "warnings.filterwarnings('ignore')\n",
                    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                    "plt.rcParams['figure.figsize'] = (12, 6)\n",
                    "plt.rcParams['font.size'] = 11\n",
                    "\n",
                    "print(\"Environment initialized successfully.\")\n",
                    "print(f\"Pandas Version: {pd.__version__}\")\n",
                    "print(f\"NumPy Version: {np.__version__}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Data Sources & Schema Inspection\n",
                    "We evaluate two primary analytical data sources:\n",
                    "1. **Dataset 1: Online Retail II (UCI)** — 1,067,371 raw transactional line items across 2009–2011 from a UK-based online non-store gift retailer.\n",
                    "2. **Dataset 2: Multi-Store Relational Retail Intelligence Dataset** — Relational schema comprising Store Master, SKU Master, Customer Master, Calendar (2022–2025), Daily Aggregated Sales, and Daily Inventory Snapshots."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Inspecting Dataset 1: UCI Online Retail II\n",
                    "raw_dir = \"../data/raw\" if os.path.exists(\"../data/raw\") else \"data/raw\"\n",
                    "uci_path = os.path.join(raw_dir, \"online_retail_II.csv\")\n",
                    "\n",
                    "df_uci = pd.read_csv(uci_path, low_memory=False)\n",
                    "print(\"=== ONLINE RETAIL II (UCI) OVERVIEW ===\")\n",
                    "print(f\"Shape: {df_uci.shape[0]:,} rows x {df_uci.shape[1]} columns\")\n",
                    "print(f\"Memory Usage: {df_uci.memory_usage(deep=True).sum() / (1024**2):.2f} MB\\n\")\n",
                    "display(df_uci.head(5))\n",
                    "display(df_uci.tail(5))"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Data Types, Nulls, and Duplicates in Dataset 1\n",
                    "uci_info_df = pd.DataFrame({\n",
                    "    'Column': df_uci.columns,\n",
                    "    'Data Type': df_uci.dtypes.values,\n",
                    "    'Non-Null Count': df_uci.notnull().sum().values,\n",
                    "    'Null Count': df_uci.isnull().sum().values,\n",
                    "    'Null %': np.round(df_uci.isnull().sum().values / len(df_uci) * 100, 2),\n",
                    "    'Unique Values': [df_uci[col].nunique() for col in df_uci.columns]\n",
                    "})\n",
                    "display(uci_info_df)\n",
                    "\n",
                    "duplicate_count = df_uci.duplicated().sum()\n",
                    "print(f\"Total Duplicate Rows in UCI Dataset: {duplicate_count:,} ({duplicate_count/len(df_uci)*100:.2f}%)\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Statistical Profile of Dataset 1\n",
                    "display(df_uci.describe().T)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "#### Key Findings from Dataset 1 (UCI Online Retail II):\n",
                    "- **Negative Quantities & Prices:** `Quantity` has minimum values of -80,995 (indicative of cancellations/returns prefixed with 'C' in Invoice) and negative/zero prices (accounting/administrative adjustments).\n",
                    "- **Missing Customer IDs:** 243,007 rows (22.77%) have null `Customer ID` (guest checkout transactions).\n",
                    "- **Missing Descriptions:** 4,382 rows have null `Description`.\n",
                    "- **Lack of Native Inventory & Lead Time:** No inventory snapshots, supplier lead times, or stock levels exist in this raw transactional log. Derived inventory simulation is required if modeling replenishment for this dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Inspecting Dataset 2: Multi-Store Relational Schema\n",
                    "store_df = pd.read_csv(os.path.join(raw_dir, \"store_master.csv\"))\n",
                    "sku_df = pd.read_csv(os.path.join(raw_dir, \"sku_master.csv\"))\n",
                    "cust_df = pd.read_csv(os.path.join(raw_dir, \"customer_master.csv\"))\n",
                    "cal_df = pd.read_csv(os.path.join(raw_dir, \"calendar.csv\"))\n",
                    "sales_df = pd.read_parquet(os.path.join(raw_dir, \"sales_daily.parquet\"))\n",
                    "inv_df = pd.read_parquet(os.path.join(raw_dir, \"inventory_snapshots.parquet\"))\n",
                    "\n",
                    "relational_summary = pd.DataFrame([\n",
                    "    {'Table': 'store_master', 'Rows': len(store_df), 'Cols': store_df.shape[1], 'Key Columns': 'store_id, city, store_type, store_size_sqft'},\n",
                    "    {'Table': 'sku_master', 'Rows': len(sku_df), 'Cols': sku_df.shape[1], 'Key Columns': 'sku_id, category, sub_category, base_price, lead_time_days, reorder_point, safety_stock'},\n",
                    "    {'Table': 'customer_master', 'Rows': len(cust_df), 'Cols': cust_df.shape[1], 'Key Columns': 'customer_id, customer_segment, loyalty_member'},\n",
                    "    {'Table': 'calendar', 'Rows': len(cal_df), 'Cols': cal_df.shape[1], 'Key Columns': 'date, year, month, quarter, day_of_week, is_weekend, is_holiday, season'},\n",
                    "    {'Table': 'sales_daily', 'Rows': len(sales_df), 'Cols': sales_df.shape[1], 'Key Columns': 'date, store_id, sku_id, units_sold, total_revenue, avg_unit_price, promotion_flag'},\n",
                    "    {'Table': 'inventory_snapshots', 'Rows': len(inv_df), 'Cols': inv_df.shape[1], 'Key Columns': 'date, store_id, sku_id, beginning_inventory, receipts, units_sold, ending_inventory, stockout_flag, on_order_qty'}\n",
                    "])\n",
                    "display(relational_summary)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Visualizing the Relational Schema Distributions\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                    "\n",
                    "# 1. Category Distribution in SKU Master\n",
                    "sku_df['category'].value_counts().plot(kind='barh', ax=axes[0, 0], color='#1f77b4', edgecolor='black')\n",
                    "axes[0, 0].set_title('SKU Master: Product Distribution by Category', fontsize=12, fontweight='bold')\n",
                    "axes[0, 0].set_xlabel('Number of SKUs')\n",
                    "\n",
                    "# 2. Store Types in Store Master\n",
                    "store_df['store_type'].value_counts().plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%', colors=['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd'])\n",
                    "axes[0, 1].set_title('Store Master: Store Format Breakdown', fontsize=12, fontweight='bold')\n",
                    "axes[0, 1].set_ylabel('')\n",
                    "\n",
                    "# 3. Lead Time Distribution\n",
                    "sns.histplot(sku_df['lead_time_days'], bins=15, kde=True, ax=axes[1, 0], color='#ff7f0e', edgecolor='black')\n",
                    "axes[1, 0].set_title('SKU Master: Supplier Lead Time (Days)', fontsize=12, fontweight='bold')\n",
                    "axes[1, 0].set_xlabel('Lead Time (Days)')\n",
                    "\n",
                    "# 4. Daily Sales Units Distribution\n",
                    "sns.histplot(sales_df['units_sold'].sample(50000, random_state=42), bins=30, kde=True, ax=axes[1, 1], color='#2ca02c', edgecolor='black')\n",
                    "axes[1, 1].set_title('Sales Daily: Daily Units Sold Distribution (Sample 50k)', fontsize=12, fontweight='bold')\n",
                    "axes[1, 1].set_xlabel('Units Sold per Day')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Comparative Schema & Analytical Gap Analysis\n",
                    "\n",
                    "| Feature / Analytical Dimension | Dataset 1: Online Retail II (UCI) | Dataset 2: Multi-Store Relational Retail Intelligence |\n",
                    "| :--- | :--- | :--- |\n",
                    "| **Granularity** | Transaction Line Item (Invoice-level) | Daily Aggregated & Multi-Store SKU Relational Tables |\n",
                    "| **Total Rows** | 1,067,371 transactions | 1,461,000 daily sales + 1,461,000 inventory snapshots |\n",
                    "| **Temporal Span** | 2009-12-01 to 2011-12-09 (2 Years) | 2022-01-01 to 2025-12-31 (4 Years) |\n",
                    "| **SKU Count** | 5,305 unique StockCodes | 5,000 catalog SKUs (100 tracked core time series) |\n",
                    "| **Customer Entities** | 5,942 unique customer IDs | 10,000 profiled customers with segment attributes |\n",
                    "| **Store / Location Dimension** | Country-level (predominantly UK) | 30 explicit retail store branches (City, State, Region, Size) |\n",
                    "| **Product Categorization** | Free text `Description` only | Structured `category`, `sub_category`, `brand` |\n",
                    "| **Pricing Structure** | `Price` (Unit selling price) | `cost_price`, `base_price`, `avg_unit_price` |\n",
                    "| **Promotions & Discounts** | Implicit in price variation | Explicit `promotion_flag`, promotional multipliers |\n",
                    "| **Inventory & Supply Chain Fields** | **None** (Requires derived simulation layer) | Explicit `beginning_inv`, `receipts`, `ending_inv`, `lead_time`, `ROP`, `safety_stock` |\n",
                    "| **Primary Target Variable** | Daily Units Demanded (`Quantity`) | Daily Units Sold (`units_sold`) / Future Demand |\n",
                    "| **Inventory Risk Target** | Simulated Days of Supply / Stockout Risk | Explicit `stockout_flag`, `ending_inventory`, `on_order_qty` |"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Architectural Strategy: Common Analytical Model (CAM)\n",
                    "Because the schemas are structurally distinct, we **do not blindly concatenate** them into a broken single file. Instead, we establish a **Common Analytical Model (CAM)**:\n",
                    "1. In **Phase 3 (Data Cleaning)**, both datasets undergo rigorous cleaning:\n",
                    "   - In UCI data: Filter invalid prices/quantities, separate cancellations (`C` prefix), clean descriptions, and aggregate to daily SKU-level sales.\n",
                    "   - In Relational data: Verify relational key consistency, remove zero-demand anomalies, and validate inventory balance equations ($EndInv = BegInv + Receipts - UnitsSold$).\n",
                    "2. In **Phase 4 (Data Integration)**, we build unified analytical tables (`daily_sales`, `sku_master`, `calendar_master`, `inventory_risk_master`).\n",
                    "3. In **Phase 5–10**, all feature engineering, machine learning forecasting (RF, XGBoost, LightGBM), evaluation, and inventory risk scoring operate consistently across this unified model."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 6. Summary & Sign-off for Phase 1 & 2\n",
                    "The business understanding is documented and both datasets are profiled.\n",
                    "\n",
                    "- **Data Understanding:** Completed.\n",
                    "- **Profiling Reports & Schema Dictionaries:** Generated.\n",
                    "- **Next Step:** Phase 3 — Data Cleaning (`notebooks/02_data_cleaning.ipynb` & `src/data_cleaning.py`)."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 4
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "01_project_and_data_understanding.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {nb_path} successfully.")

if __name__ == "__main__":
    create_notebook_01()
