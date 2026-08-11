"""
Script to create notebooks/02_data_cleaning.ipynb (Phase 3).
The notebook is built programmatically with nbformat so it stays reproducible.
Execute it afterwards with:  python -m jupyter nbconvert --to notebook --execute
"""

import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------
def md(src):
    return new_markdown_cell(src)


def code(src):
    return new_code_cell(src)


cells = []

# ===========================================================================
# 1. INTRODUCTION
# ===========================================================================
cells.append(md(
    "# Demand & Inventory Intelligence System\n"
    "## Retail Demand Forecasting and Inventory Risk Prediction\n"
    "---\n"
    "### Phase 3 — Data Cleaning & Data Quality Engineering\n"
    "\n"
    "**Project:** Project FORESIGHT — Retail Demand & Inventory Intelligence  \n"
    "**Author:** Data Science & Analytics Intern  \n"
    "**Date:** August 2026  \n"
    "**Prerequisite:** Phase 1 & 2 (Business Understanding & Data Profiling) — completed\n"
    "\n"
    "---\n"
    "### 1.1 Objective\n"
    "\n"
    "This phase converts the two raw dataset families into **reliable, clean, "
    "validated datasets** that can be used safely for EDA, feature engineering, "
    "forecasting and inventory-risk analysis in later phases. The cleaning "
    "pipeline detects data-quality problems, handles missing values / duplicates / "
    "invalid values / cancellations / returns, validates dates, numbers, "
    "categories and relationships, preserves useful information, and produces a "
    "**data-quality report**.\n"
    "\n"
    "### 1.2 Dataset families\n"
    "\n"
    "1. **UCI Online Retail II** — 1,067,371 transaction line-items (2009-12-01 "
    "to 2011-12-09) from a UK-based online gift retailer.\n"
    "2. **Synthetic multi-store relational dataset** (2022-01-01 to 2025-12-31): "
    "`store_master`, `sku_master`, `customer_master`, `calendar`, `sales_daily`, "
    "`inventory_snapshots`.\n"
    "\n"
    "### 1.3 Cleaning principles (the rules we follow)\n"
    "\n"
    "1. **Never modify raw data.** All outputs live under `data/processed/`.\n"
    "2. **Never fabricate missing information.** Missing descriptions are recovered "
    "only from the same StockCode; otherwise a documented `Unknown Product` "
    "sentinel is used.\n"
    "3. **Never silently delete records.** Every removal has a documented reason.\n"
    "4. **Returns & cancellations are preserved separately**, not deleted.\n"
    "5. **Missing Customer ID is not treated as invalid** — guest transactions "
    "are kept for sales analysis and only excluded from customer-level analysis.\n"
    "6. **Legitimate retail outliers are kept**; outliers are *investigated*, "
    "not removed.\n"
    "7. The pipeline is **reproducible** and driven by **reusable functions** in "
    "`src/data_cleaning.py`.\n"
    "\n"
    "### 1.4 Deliverables of Phase 3\n"
    "\n"
    "- Executed notebook: `notebooks/02_data_cleaning.ipynb`\n"
    "- Reusable module: `src/data_cleaning.py`\n"
    "- Clean datasets: `data/processed/*`\n"
    "- Data-quality report: `docs/data_quality_report.json` + `.csv`\n"
    "- Cleaning visualisations: `outputs/figures/*`\n"
))

# ===========================================================================
# 2. LOAD RAW DATA
# ===========================================================================
cells.append(md(
    "### 2. Load Raw Data\n"
    "\n"
    "We locate the project root, import the reusable cleaning module "
    "(`src/data_cleaning.py`) and load every raw dataset. Parquet versions are "
    "used for the two large fact tables (`sales_daily`, `inventory_snapshots`)."
))

cells.append(code(
    "# ---- Environment & project root -------------------------------------\n"
    "import os, sys, json, warnings\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import plotly.express as px\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "# Locate project root robustly regardless of the kernel working directory.\n"
    "def _find_project_root():\n"
    "    d = os.getcwd()\n"
    "    for _ in range(4):\n"
    "        if os.path.exists(os.path.join(d, 'data', 'raw', 'online_retail_II.csv')):\n"
    "            return d\n"
    "        d = os.path.dirname(d)\n"
    "    return os.path.abspath(os.path.join(os.getcwd(), '..'))\n"
    "\n"
    "PROJECT_ROOT = _find_project_root()\n"
    "sys.path.insert(0, PROJECT_ROOT)\n"
    "sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))\n"
    "print('Project root:', PROJECT_ROOT)\n"
    "\n"
    "from data_cleaning import *\n"
    "from data_cleaning import UNKNOWN_PRODUCT, TX_SALE, TX_RETURN, TX_CANCELLATION, TX_INVALID\n"
    "\n"
    "dirs = ensure_directories()\n"
    "print('Directories ready:', dirs)\n"
    "\n"
    "print(f\"pandas {pd.__version__} | numpy {np.__version__} | matplotlib {plt.matplotlib.__version__} | seaborn {sns.__version__}\")\n"
    "\n"
    "# ---- Consistent, colour-safe plotting style ---------------------------\n"
    "# Validated default palette (dataviz reference palette):\n"
    "# categorical slots in fixed order, light surface, recessive grids.\n"
    "PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']\n"
    "GOOD, WARN, SERIOUS, CRITICAL = '#0ca30c', '#fab219', '#ec835a', '#d03b3b'\n"
    "INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'\n"
    "sns.set_theme(style='whitegrid', rc={\n"
    "    'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,\n"
    "    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,\n"
    "    'axes.spines.top': False, 'axes.spines.right': False,\n"
    "    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,\n"
    "    'text.color': INK, 'axes.titleweight': 'bold', 'font.size': 11,\n"
    "})\n"
    "plt.rcParams['figure.dpi'] = 100\n"
    "plt.rcParams['savefig.dpi'] = 150\n"
    "plt.rcParams['savefig.bbox'] = 'tight'\n"
))

cells.append(code(
    "# ---- Load raw data ---------------------------------------------------\n"
    "raw_uci = load_online_retail_data()\n"
    "syn = load_synthetic_retail_data()\n"
    "\n"
    "summary = pd.DataFrame([\n"
    "    {'Dataset': 'online_retail_II', 'Rows': len(raw_uci), 'Columns': raw_uci.shape[1], 'Source': 'CSV (95 MB)'},\n"
    "    {'Dataset': 'store_master', 'Rows': len(syn['store_master']), 'Columns': syn['store_master'].shape[1], 'Source': 'CSV'},\n"
    "    {'Dataset': 'sku_master', 'Rows': len(syn['sku_master']), 'Columns': syn['sku_master'].shape[1], 'Source': 'CSV'},\n"
    "    {'Dataset': 'customer_master', 'Rows': len(syn['customer_master']), 'Columns': syn['customer_master'].shape[1], 'Source': 'CSV'},\n"
    "    {'Dataset': 'calendar', 'Rows': len(syn['calendar']), 'Columns': syn['calendar'].shape[1], 'Source': 'CSV'},\n"
    "    {'Dataset': 'sales_daily', 'Rows': len(syn['sales_daily']), 'Columns': syn['sales_daily'].shape[1], 'Source': 'Parquet (6 MB)'},\n"
    "    {'Dataset': 'inventory_snapshots', 'Rows': len(syn['inventory_snapshots']), 'Columns': syn['inventory_snapshots'].shape[1], 'Source': 'Parquet (5 MB)'},\n"
    "])\n"
    "display(summary)\n"
    "\n"
    "# Working containers shared by all sections.\n"
    "clean = {}\n"
    "reports = {}\n"
    "print('\\nAll raw datasets loaded.')\n"
))

# ===========================================================================
# 3. DATA QUALITY ASSESSMENT
# ===========================================================================
cells.append(md(
    "### 3. Data Quality Assessment (Baseline)\n"
    "\n"
    "Before cleaning we profile every dataset: schema, data types, missing "
    "values, exact duplicates, and date ranges. This is the *baseline* that the "
    "cleaning decisions in the following sections are measured against. For the "
    "Online Retail II dataset we also run the authoritative cleaning pipeline "
    "(`clean_online_retail`) so that every later section reports the same numbers."
))

cells.append(code(
    "# ---- Baseline profile of all raw datasets ---------------------------\n"
    "def _baseline(name, df):\n"
    "    return {\n"
    "        'Dataset': name,\n"
    "        'Rows': len(df),\n"
    "        'Columns': df.shape[1],\n"
    "        'Null Cells': int(df.isna().sum().sum()),\n"
    "        'Exact Duplicate Rows': int(df.duplicated().sum()),\n"
    "    }\n"
    "\n"
    "baseline = pd.DataFrame([\n"
    "    _baseline('online_retail_II', raw_uci),\n"
    "    _baseline('store_master', syn['store_master']),\n"
    "    _baseline('sku_master', syn['sku_master']),\n"
    "    _baseline('customer_master', syn['customer_master']),\n"
    "    _baseline('calendar', syn['calendar']),\n"
    "    _baseline('sales_daily', syn['sales_daily']),\n"
    "    _baseline('inventory_snapshots', syn['inventory_snapshots']),\n"
    "])\n"
    "display(baseline)\n"
))

cells.append(code(
    "# ---- UCI: schema validation + missing/duplicate detail ---------------\n"
    "schema_uci = validate_schema(raw_uci, ONLINE_RETAIL_REQUIRED_COLUMNS, 'online_retail_ii')\n"
    "print('UCI schema OK:', schema_uci['schema_ok'])\n"
    "if not schema_uci['schema_ok']:\n"
    "    print('Missing columns:', schema_uci['missing_columns'])\n"
    "\n"
    "uci_missing = missing_value_summary(raw_uci, 'online_retail_ii')\n"
    "display(pd.DataFrame({\n"
    "    'Column': list(uci_missing['missing_by_column'].keys()),\n"
    "    'Missing': [v['count'] for v in uci_missing['missing_by_column'].values()],\n"
    "    'Missing %': [v['pct'] for v in uci_missing['missing_by_column'].values()],\n"
    "}))\n"
    "\n"
    "print(f\"Exact duplicate rows in raw UCI data: {raw_uci.duplicated().sum():,} \"\n"
    "      f\"({raw_uci.duplicated().sum()/len(raw_uci)*100:.2f}%)\")\n"
    "print(f\"Missing Customer ID: {raw_uci['Customer ID'].isna().sum():,} \"\n"
    "      f\"({raw_uci['Customer ID'].isna().mean()*100:.2f}%)\")\n"
    "print(f\"Missing Description: {raw_uci['Description'].isna().sum():,} \"\n"
    "      f\"({raw_uci['Description'].isna().mean()*100:.2f}%)\")\n"
))

cells.append(code(
    "# ---- UCI: run the authoritative cleaning to establish the baseline ---\n"
    "uci_clean, uci_report = clean_online_retail(raw_uci)\n"
    "clean['online_retail'] = uci_clean\n"
    "reports['online_retail_ii'] = uci_report\n"
    "\n"
    "print(f\"Raw rows        : {uci_report['original_rows']:,}\")\n"
    "print(f\"Clean rows      : {uci_report['final_rows']:,}\")\n"
    "print(f\"Rows removed    : {uci_report['original_rows'] - uci_report['final_rows']:,} (exact duplicates)\")\n"
    "print(f\"Cancellations   : {uci_report['cancellation_count']:,} ({uci_report['cancellation_pct']:.2f}%)\")\n"
    "print(f\"Returns         : {uci_report['return_count']:,}\")\n"
    "print(f\"Invalid lines   : {uci_report['invalid_count']:,}\")\n"
    "print(f\"Guest txns      : {uci_report['guest_transaction_count']:,} ({uci_report['guest_transaction_pct']:.2f}%)\")\n"
))

# ===========================================================================
# 4. DUPLICATE HANDLING
# ===========================================================================
cells.append(md(
    "### 4. Duplicate Handling\n"
    "\n"
    "**Decision rule.** Exact full-row duplicates are treated as data-entry "
    "errors (a line recorded twice) and are removed from the *processed* dataset. "
    "The raw files are never touched. We first verify that the duplicates are "
    "indeed exact (all columns identical) and quantify them per transaction "
    "class before removal. For the relational tables the relevant uniqueness "
    "constraint is the business key: `store_id`, `sku_id`, `customer_id`, "
    "`calendar.date`, and the `(date, store_id, sku_id)` grain of the fact tables."
))

cells.append(code(
    "# ---- UCI: exact-duplicate analysis ----------------------------------\n"
    "dups_uci = raw_uci.duplicated(keep=False)   # mark all occurrences of duplicated rows\n"
    "dup_rows = raw_uci[dups_uci]\n"
    "print(f\"Rows participating in a duplicate set : {len(dup_rows):,} \"\n"
    "      f\"({len(dup_rows)/len(raw_uci)*100:.2f}%)\")\n"
    "print(f\"Duplicates removed (kept first)       : {uci_report['duplicate_rows']:,} \"\n"
    "      f\"({uci_report['duplicate_pct']:.2f}%)\")\n"
    "\n"
    "# Confirm duplicates are exact across every column\n"
    "exact = dup_rows.groupby(list(raw_uci.columns)).size()\n"
    "print(f\"Unique duplicated row signatures       : {len(exact)}\")\n"
    "print(\"Sample duplicated signature:\")\n"
    "display(dup_rows.head(6))\n"
))

cells.append(code(
    "# ---- Synthetic masters: ID uniqueness ---------------------------------\n"
    "store_master, store_rep = clean_store_master(syn['store_master'])\n"
    "clean['store_master'] = store_master\n"
    "reports['store_master'] = store_rep\n"
    "\n"
    "sku_master, sku_rep = clean_sku_master(syn['sku_master'])\n"
    "clean['sku_master'] = sku_master\n"
    "reports['sku_master'] = sku_rep\n"
    "\n"
    "customer_master, cust_rep = clean_customer_master(syn['customer_master'])\n"
    "clean['customer_master'] = customer_master\n"
    "reports['customer_master'] = cust_rep\n"
    "\n"
    "id_checks = pd.DataFrame([\n"
    "    {'Table': 'store_master', 'Rows': len(store_master), 'Duplicate IDs': store_rep['duplicate_store_ids'], 'Missing IDs': store_rep['missing_ids']},\n"
    "    {'Table': 'sku_master', 'Rows': len(sku_master), 'Duplicate IDs': sku_rep['duplicate_sku_ids'], 'Missing IDs': sku_rep['missing_ids']},\n"
    "    {'Table': 'customer_master', 'Rows': len(customer_master), 'Duplicate IDs': cust_rep['duplicate_customer_ids'], 'Missing IDs': cust_rep['missing_ids']},\n"
    "])\n"
    "display(id_checks)\n"
    "print('Store master status:', store_rep['quality_status'], '| SKU master status:', sku_rep['quality_status'],\n"
    "      '| Customer master status:', cust_rep['quality_status'])\n"
))

# ===========================================================================
# 5. MISSING VALUE HANDLING
# ===========================================================================
cells.append(md(
    "### 5. Missing Value Handling\n"
    "\n"
    "Two genuine missing-value problems exist in the UCI data:\n"
    "\n"
    "1. **Description (4,382 rows / 0.41%).** Missing descriptions are recovered "
    "from other line-items sharing the same `StockCode` (modal description). "
    "Lines whose StockCode never carries a known description anywhere are "
    "labelled `Unknown Product` — we never invent product names.\n"
    "2. **Customer ID (243,007 rows / 22.77%).** This is **not** invalid — these "
    "are guest checkout transactions. They are **kept** for sales/demand "
    "analysis and flagged with `is_guest_transaction=True` so that customer "
    "*segmentation* analyses can exclude them without losing sales volume.\n"
    "\n"
    "The synthetic relational tables contain no missing values."
))

cells.append(code(
    "# ---- UCI: Description recovery ---------------------------------------\n"
    "print('Recovered from same StockCode :', int((uci_clean['description_source']=='recovered').sum()))\n"
    "print('Unknown Product (unrecoverable):', int((uci_clean['description_source']=='unknown').sum()))\n"
    "print('Originally present            :', int((uci_clean['description_source']=='original').sum()))\n"
    "\n"
    "# Example of a recovered description\n"
    "rec = uci_clean[uci_clean['description_source']=='recovered'].head(3)\n"
    "display(rec[['Invoice', 'StockCode', 'Description', 'description_source']])\n"
    "\n"
    "print('Remaining missing Description after cleaning: ', int(uci_clean['Description'].isna().sum()))\n"
))

cells.append(code(
    "# ---- UCI: Guest-customer treatment -----------------------------------\n"
    "guest = uci_clean[uci_clean['is_guest_transaction']]\n"
    "ident = uci_clean[~uci_clean['is_guest_transaction']]\n"
    "print(f\"Guest transactions (kept for sales analysis): {len(guest):,} ({len(guest)/len(uci_clean)*100:.2f}%)\")\n"
    "print(f\"Identified-customer transactions              : {len(ident):,} ({len(ident)/len(uci_clean)*100:.2f}%)\")\n"
    "print('\\n-> Use full dataset for demand/sales analysis; use only identified customers for segmentation.')\n"
))

cells.append(code(
    "# ---- Synthetic: missing-value scan -----------------------------------\n"
    "for key in ['store_master', 'sku_master', 'customer_master', 'calendar', 'sales_daily', 'inventory_snapshots']:\n"
    "    df = syn[key]\n"
    "    total = int(df.isna().sum().sum())\n"
    "    if total:\n"
    "        cols = [c for c, n in df.isna().sum().items() if n > 0]\n"
    "        print(f\"{key:24s} missing cells = {total:,}  (columns: {cols})\")\n"
    "    else:\n"
    "        print(f\"{key:24s} missing cells = 0\")\n"
))

# ===========================================================================
# 6. INVALID VALUE HANDLING
# ===========================================================================
cells.append(md(
    "### 6. Invalid Value Handling\n"
    "\n"
    "For the UCI data we analyse `Quantity` and `Price` in detail. Retail "
    "transaction logs legitimately contain negative quantities (cancellations "
    "and returns), zero prices (postage, samples, manual adjustments) and "
    "negative prices (accounting journal lines). Instead of deleting them we "
    "classify every line with a documented `transaction_type`, and annotate "
    "`price_category` + `is_special_transaction` so downstream analysis can "
    "include or exclude them deliberately."
))

cells.append(code(
    "# ---- UCI: Quantity profile (raw) -------------------------------------\n"
    "qty = raw_uci['Quantity']\n"
    "qty_prof = pd.DataFrame([\n"
    "    {'Band': 'Quantity < 0', 'Rows': int((qty<0).sum()), 'Share %': round((qty<0).mean()*100, 3)},\n"
    "    {'Band': 'Quantity == 0', 'Rows': int((qty==0).sum()), 'Share %': round((qty==0).mean()*100, 3)},\n"
    "    {'Band': 'Quantity > 0', 'Rows': int((qty>0).sum()), 'Share %': round((qty>0).mean()*100, 3)},\n"
    "])\n"
    "display(qty_prof)\n"
    "print('Quantity min / max:', qty.min(), '/', qty.max())\n"
))

cells.append(code(
    "# ---- UCI: Price profile (raw) ----------------------------------------\n"
    "price = raw_uci['Price']\n"
    "price_prof = pd.DataFrame([\n"
    "    {'Band': 'Price < 0', 'Rows': int((price<0).sum()), 'Share %': round((price<0).mean()*100, 4)},\n"
    "    {'Band': 'Price == 0', 'Rows': int((price==0).sum()), 'Share %': round((price==0).mean()*100, 3)},\n"
    "    {'Band': 'Price > 0', 'Rows': int((price>0).sum()), 'Share %': round((price>0).mean()*100, 3)},\n"
    "])\n"
    "display(price_prof)\n"
    "print('Price min / max:', price.min(), '/', price.max())\n"
    "\n"
    "# Investigate zero / negative price lines before any decision.\n"
    "special = raw_uci[price <= 0]\n"
    "print('\\nDescription of zero/negative-price lines (top 12):')\n"
    "print(special['Description'].value_counts().head(12).to_string())\n"
    "print('\\nAll negative-price lines are accounting journal lines:')\n"
    "print(special[special['Price']<0][['Invoice','StockCode','Description','Quantity','Price']].to_string())\n"
))

cells.append(code(
    "# ---- UCI: transaction-type & price-category annotation ---------------\n"
    "print('transaction_type distribution (clean):')\n"
    "display(pd.DataFrame.from_dict(\n"
    "    uci_report['transaction_type_counts'], orient='index', columns=['Rows']).rename_axis('transaction_type'))\n"
    "\n"
    "print('price_category distribution (clean):')\n"
    "display(uci_clean['price_category'].value_counts().rename('Rows').to_frame())\n"
    "\n"
    "print(f\"Special transactions (price <= 0) kept but flagged: {uci_clean['is_special_transaction'].sum():,}\")\n"
    "print(f\"Invalid (accounting adjustment) lines removed from sales: {uci_report['invalid_values']}\")\n"
))

cells.append(code(
    "# ---- Synthetic: invalid-value & flag checks --------------------------\n"
    "flag_checks = pd.DataFrame([\n"
    "    {'Table': 'customer_master', 'Check': 'loyalty_member in {0,1}', 'Invalid': cust_rep['invalid_loyalty_member']},\n"
    "    {'Table': 'customer_master', 'Check': 'valid customer_segment', 'Invalid': cust_rep['invalid_segment']},\n"
    "    {'Table': 'store_master', 'Check': 'store_size_sqft > 0', 'Invalid': store_rep['invalid_store_sizes']},\n"
    "    {'Table': 'sku_master', 'Check': 'cost_price > 0', 'Invalid': sku_rep['invalid_cost_price']},\n"
    "    {'Table': 'sku_master', 'Check': 'base_price > 0', 'Invalid': sku_rep['invalid_base_price']},\n"
    "    {'Table': 'sku_master', 'Check': 'lead_time_days > 0', 'Invalid': sku_rep['invalid_lead_time_days']},\n"
    "    {'Table': 'sku_master', 'Check': 'reorder_point >= 0', 'Invalid': sku_rep['negative_reorder_point']},\n"
    "    {'Table': 'sku_master', 'Check': 'safety_stock >= 0', 'Invalid': sku_rep['negative_safety_stock']},\n"
    "])\n"
    "display(flag_checks)\n"
))

# ===========================================================================
# 7. TRANSACTION CLEANING
# ===========================================================================
cells.append(md(
    "### 7. Transaction Cleaning\n"
    "\n"
    "Every UCI line is classified with the following **documented rule set** "
    "(discovered by inspecting the actual data):\n"
    "\n"
    "| Rule | Condition | transaction_type | Treatment |\n"
    "|---|---|---|---|\n"
    "| 1 | `Invoice` starts with `C` | CANCELLATION | kept in a separate dataset |\n"
    "| 2 | `Invoice` starts with `A` (all are `Adjust bad debt` journal lines) | INVALID | excluded from sales |\n"
    "| 3 | `Price < 0` (accounting adjustment) | INVALID | excluded from sales |\n"
    "| 4 | `Quantity < 0` (non-C, non-A) | RETURN | kept in a separate dataset |\n"
    "| 5 | otherwise | SALE | valid sales transaction |\n"
    "\n"
    "Zero-price `SALE` lines (postage, samples, manual adjustments) are **kept** "
    "and flagged via `price_category` / `is_special_transaction`; they are not "
    "silently removed."
))

cells.append(code(
    "# ---- UCI: final cleaned transaction frame ----------------------------\n"
    "print('Cleaned Online Retail II schema:')\n"
    "print(list(uci_clean.columns))\n"
    "print('\\nSample of cleaned sales lines:')\n"
    "display(uci_clean[uci_clean['transaction_type']==TX_SALE][\n"
    "    ['Invoice','StockCode','Description','Quantity','Price','InvoiceDate','Country',\n"
    "     'transaction_type','price_category','is_guest_transaction','description_source']\n"
    "].head(8))\n"
))

cells.append(code(
    "# ---- UCI: description source & date-field sanity ---------------------\n"
    "print('description_source:')\n"
    "print(uci_clean['description_source'].value_counts().to_string())\n"
    "print('\\nDate columns added during cleaning:')\n"
    "print(uci_clean[['invoice_year','invoice_month','invoice_weekday']].head(3).to_string())\n"
    "print('\\nNo missing values remain in Description:', int(uci_clean['Description'].isna().sum()) == 0)\n"
))

# ===========================================================================
# 8. RETURN & CANCELLATION HANDLING
# ===========================================================================
cells.append(md(
    "### 8. Return & Cancellation Handling\n"
    "\n"
    "Cancellations and returns are **not deleted** — they are material business "
    "events (lost revenue, refund exposure, return logistics). They are split "
    "into dedicated analytical datasets so they can be modelled independently, "
    "and their monetary impact is quantified."
))

cells.append(code(
    "# ---- UCI: split into analytical components ---------------------------\n"
    "uci_sales, uci_returns, uci_cancellations, uci_invalid = split_transactions(uci_clean)\n"
    "\n"
    "split_table = pd.DataFrame([\n"
    "    {'Component': 'Valid sales (SALE)', 'Rows': len(uci_sales), 'Share %': round(len(uci_sales)/len(uci_clean)*100, 2)},\n"
    "    {'Component': 'Returns (RETURN)', 'Rows': len(uci_returns), 'Share %': round(len(uci_returns)/len(uci_clean)*100, 2)},\n"
    "    {'Component': 'Cancellations (CANCELLATION)', 'Rows': len(uci_cancellations), 'Share %': round(len(uci_cancellations)/len(uci_clean)*100, 2)},\n"
    "    {'Component': 'Invalid / adjustments (INVALID)', 'Rows': len(uci_invalid), 'Share %': round(len(uci_invalid)/len(uci_clean)*100, 2)},\n"
    "])\n"
    "display(split_table)\n"
    "assert len(uci_sales)+len(uci_returns)+len(uci_cancellations)+len(uci_invalid) == len(uci_clean)\n"
    "print('Split is exhaustive (components sum to the cleaned frame).')\n"
))

cells.append(code(
    "# ---- UCI: monetary impact of returns & cancellations ------------------\n"
    "def _value(df):\n"
    "    return float((df['Quantity'] * df['Price']).sum())\n"
    "\n"
    "print(f\"Gross value of valid sales lines : {_value(uci_sales):+,.2f}\")\n"
    "print(f\"Value of returns                 : {_value(uci_returns):+,.2f}\")\n"
    "print(f\"Value of cancellations          : {_value(uci_cancellations):+,.2f}\")\n"
    "print(f\"Value of invalid/adjustment lines: {_value(uci_invalid):+,.2f}\")\n"
    "\n"
    "# Cancellation vs return samples\n"
    "print('\\nSample cancellation lines:')\n"
    "display(uci_cancellations[['Invoice','StockCode','Description','Quantity','Price']].head(5))\n"
    "print('\\nSample return lines:')\n"
    "display(uci_returns[['Invoice','StockCode','Description','Quantity','Price']].head(5))\n"
))

cells.append(code(
    "# ---- Synthetic: returns/cancellations modelling note -----------------\n"
    "print('The relational fact tables are DAILY AGGREGATES at (date, store_id, sku_id) grain.')\n"
    "print('Returns and cancellations are not recorded as separate line items; they are netted')\n"
    "print('into units_sold / total_revenue. Returns/cancellation analytics are therefore modelled')\n"
    "print('only on the Online Retail II transaction log, while the relational tables model')\n"
    "print('net demand and inventory movement.')\n"
))

# ===========================================================================
# 9. DATE/TIME CLEANING
# ===========================================================================
cells.append(md(
    "### 9. Date/Time Cleaning\n"
    "\n"
    "All date columns are converted to `datetime64` and validated for missing, "
    "invalid and future values. The calendar dimension is additionally checked "
    "for **internal consistency** (its `year/month/day/quarter/week_of_year/"
    "day_of_week/is_weekend` attributes must match the actual date)."
))

cells.append(code(
    "# ---- UCI: InvoiceDate validation -------------------------------------\n"
    "dt = uci_clean['InvoiceDate']\n"
    "print('Date range            :', dt.min(), '->', dt.max())\n"
    "print('Total distinct days   :', dt.dt.normalize().nunique())\n"
    "print('Invalid / missing     :', int(dt.isna().sum()))\n"
    "print('Future dates (> 2011) :', int((dt > '2011-12-31').sum()))\n"
    "print('Weekend transactions  :', int(dt.dt.dayofweek.isin([5,6]).sum()))\n"
))

cells.append(code(
    "# ---- Synthetic: calendar validation -----------------------------------\n"
    "calendar_clean, cal_rep = clean_calendar(syn['calendar'])\n"
    "clean['calendar'] = calendar_clean\n"
    "reports['calendar'] = cal_rep\n"
    "\n"
    "print('Calendar status     :', cal_rep['quality_status'])\n"
    "print('Date range          :', cal_rep['date_validation']['min_date'], '->', cal_rep['date_validation']['max_date'])\n"
    "print('Duplicate dates     :', cal_rep['duplicate_dates'])\n"
    "print('Attribute mismatches:', cal_rep['attribute_mismatch_total'])\n"
    "display(pd.DataFrame.from_dict(cal_rep['attribute_mismatches'], orient='index', columns=['Mismatch Rows']).rename_axis('attribute'))\n"
))

cells.append(code(
    "# ---- Synthetic: opening / signup date checks --------------------------\n"
    "master_date_checks = pd.DataFrame([\n"
    "    {'Table': 'store_master', 'Column': 'opening_date', 'Min': store_rep['date_validation']['min_date'],\n"
    "     'Max': store_rep['date_validation']['max_date'], 'Invalid': store_rep['date_validation']['invalid_dates'],\n"
    "     'Future': store_rep['date_validation']['future_dates']},\n"
    "    {'Table': 'customer_master', 'Column': 'signup_date', 'Min': cust_rep['date_validation']['min_date'],\n"
    "     'Max': cust_rep['date_validation']['max_date'], 'Invalid': cust_rep['date_validation']['invalid_dates'],\n"
    "     'Future': cust_rep['date_validation']['future_dates']},\n"
    "])\n"
    "display(master_date_checks)\n"
))

# ===========================================================================
# 10. NUMERICAL VALIDATION
# ===========================================================================
cells.append(md(
    "### 10. Numerical Validation\n"
    "\n"
    "We validate every numeric field for negatives, zeros and plausible bounds, "
    "and validate the **grain** of the fact tables. For `sales_daily` we also "
    "check the internal identity `total_revenue == units_sold * avg_unit_price`; "
    "for `inventory_snapshots` we check the canonical balance equation (see the "
    "documented semantic below)."
))

cells.append(code(
    "# ---- UCI: numeric profiles -------------------------------------------\n"
    "uci_num = uci_report['numeric_validation']\n"
    "num_rows = []\n"
    "for col, prof in uci_num.items():\n"
    "    num_rows.append({'Column': col, 'Non-null': prof['non_null'], 'Negative': prof['negative'],\n"
    "                     'Zero': prof['zero'], 'Positive': prof['positive'],\n"
    "                     'Min': prof['min'], 'Max': prof['max']})\n"
    "display(pd.DataFrame(num_rows))\n"
))

cells.append(code(
    "# ---- Synthetic: sales_daily validation -------------------------------\n"
    "sales_daily, sales_rep = clean_sales_daily(\n"
    "    syn['sales_daily'],\n"
    "    store_ids=store_master['store_id'],\n"
    "    sku_ids=sku_master['sku_id'],\n"
    "    calendar_dates=calendar_clean['date'].astype(str),\n"
    ")\n"
    "clean['sales_daily'] = sales_daily\n"
    "reports['sales_daily'] = sales_rep\n"
    "\n"
    "print('Sales status            :', sales_rep['quality_status'])\n"
    "print('Grain (date,store,sku) dups:', sales_rep['duplicate_grain'])\n"
    "print('Invalid promotion flags :', sales_rep['invalid_promotion_flags'])\n"
    "print('Revenue != units*price  :', sales_rep['revenue_mismatch_rows'])\n"
    "print('Zero-unit rows          :', f\"{sales_rep['zero_unit_rows']:,}\")\n"
    "num = sales_rep['numeric_validation']\n"
    "display(pd.DataFrame([\n"
    "    {'Column': c, 'Negative': p['negative'], 'Zero': p['zero'], 'Min': p['min'], 'Max': p['max']}\n"
    "    for c, p in num.items()\n"
    "]))\n"
))

cells.append(code(
    "# ---- Synthetic: inventory_snapshots validation ------------------------\n"
    "inventory, inv_rep = clean_inventory_data(syn['inventory_snapshots'], sales_df=sales_daily)\n"
    "clean['inventory_snapshots'] = inventory\n"
    "reports['inventory_snapshots'] = inv_rep\n"
    "\n"
    "print('Inventory status            :', inv_rep['quality_status'])\n"
    "print('Grain (date,store,sku) dups :', inv_rep['duplicate_grain'])\n"
    "print('Invalid stockout flags      :', inv_rep['invalid_stockout_flags'])\n"
    "print()\n"
    "print('--- Inventory balance equation (canonical: ending = beginning + receipts - units_sold) ---')\n"
    "print('Rows satisfying canonical equation :', f\"{inv_rep['canonical_equation_ok_rows']:,}\")\n"
    "print('Rows violating canonical equation  :', f\"{inv_rep['canonical_equation_mismatch_rows']:,} \"\n"
    "      f\"({inv_rep['canonical_equation_mismatch_pct']:.2f}%)\")\n"
    "print('Rows satisfying ending = beginning - units_sold (raw semantics):',\n"
    "      f\"{inv_rep['ending_equals_beginning_minus_sold_rows']:,}\")\n"
    "print('Cross-check units_sold == sales_daily.units_sold:', inv_rep['cross_sales_units_match'])\n"
))

cells.append(md(
    "**Inventory equation — documented semantic.** Investigation shows the raw "
    "`beginning_inventory` **already includes the day's receipts** (the "
    "simulator snapshots opening stock after deliveries). Therefore:\n"
    "\n"
    "- `ending == beginning + receipts - units_sold` fails on exactly the "
    "`receipts > 0` rows (8.36% of the dataset), by exactly the receipts amount;\n"
    "- `ending == beginning - units_sold` holds on **100%** of rows.\n"
    "\n"
    "Rather than overwriting any value, we add a documented derived column "
    "`beginning_inventory_pre_receipts` (opening stock *before* the day's "
    "deliveries) so the canonical balance equation holds everywhere, and flag "
    "each row with `inventory_balance_ok`."
))

cells.append(code(
    "# ---- Inventory: derived balance column -------------------------------\n"
    "print('Derived column added:', inv_rep['derived_column_added'])\n"
    "ok_rows = int(inventory['inventory_balance_ok'].sum())\n"
    "print('Rows with canonical balance on derived basis:', f\"{ok_rows:,}\", '/', len(inventory))\n"
    "assert ok_rows == len(inventory)\n"
    "print('Inventory columns now:', len(inventory.columns))\n"
    "display(inventory.head(5))\n"
))

# ===========================================================================
# 11. CATEGORICAL VALIDATION
# ===========================================================================
cells.append(md(
    "### 11. Categorical Validation\n"
    "\n"
    "Categorical fields are checked for their expected value sets and dominance: "
    "`Country` (UCI), product `category`/`sub_category`/`brand` (SKU master), "
    "`customer_segment`, `store_type`, `region`, and the boolean flag fields."
))

cells.append(code(
    "# ---- UCI: Country -----------------------------------------------------\n"
    "country = uci_clean['Country'].value_counts()\n"
    "print('Distinct countries:', country.shape[0])\n"
    "top_countries = country.head(10).to_frame('Rows')\n"
    "top_countries['Share %'] = (country.head(10)/len(uci_clean)*100).round(2)\n"
    "display(top_countries)\n"
))

cells.append(code(
    "# ---- UCI: StockCode / product sanity ----------------------------------\n"
    "print('Distinct StockCodes:', uci_clean['StockCode'].nunique())\n"
    "print('Distinct product Descriptions:', uci_clean['Description'].nunique())\n"
    "print('\\nTop 10 StockCodes by lines:')\n"
    "print(uci_clean['StockCode'].value_counts().head(10).to_string())\n"
))

cells.append(code(
    "# ---- Synthetic: categorical profiles ---------------------------------\n"
    "cat_table = []\n"
    "cat_table.append(('sku.category', sku_master['category'].value_counts()))\n"
    "cat_table.append(('sku.sub_category', sku_master['sub_category'].value_counts()))\n"
    "cat_table.append(('sku.brand (top 5)', sku_master['brand'].value_counts().head(5)))\n"
    "cat_table.append(('store.store_type', store_master['store_type'].value_counts()))\n"
    "cat_table.append(('store.region', store_master['region'].value_counts()))\n"
    "cat_table.append(('customer.segment', customer_master['customer_segment'].value_counts()))\n"
    "cat_table.append(('calendar.season', calendar_clean['season'].value_counts()))\n"
    "\n"
    "for name, vc in cat_table:\n"
    "    print(f'\\n-- {name} --')\n"
    "    print(vc.to_string())\n"
    "\n"
    "# Boolean flags\n"
    "print('\\n-- Boolean flag validation --')\n"
    "print('calendar.is_weekend values:', calendar_clean['is_weekend'].unique())\n"
    "print('calendar.is_holiday values:', calendar_clean['is_holiday'].unique())\n"
    "print('sales.promotion_flag values:', sales_daily['promotion_flag'].unique())\n"
    "print('inventory.stockout_flag values:', inventory['stockout_flag'].unique())\n"
    "print('customer.loyalty_member values:', customer_master['loyalty_member'].unique())\n"
))

# ===========================================================================
# 12. REFERENTIAL INTEGRITY
# ===========================================================================
cells.append(md(
    "### 12. Referential Integrity\n"
    "\n"
    "We validate every declared relationship between fact tables and their "
    "dimension tables. **Master records are never invented** to fix orphans — "
    "orphans are counted and reported."
))

cells.append(code(
    "# ---- Relational integrity checks -------------------------------------\n"
    "relationships = [\n"
    "    ('sales_daily.store_id', 'store_master.store_id', sales_daily, store_master, 'store_id', 'store_id'),\n"
    "    ('sales_daily.sku_id',   'sku_master.sku_id',   sales_daily, sku_master,   'sku_id', 'sku_id'),\n"
    "    ('sales_daily.date',     'calendar.date',       sales_daily, calendar_clean, 'date', 'date'),\n"
    "    ('inventory.store_id',   'store_master.store_id', inventory, store_master, 'store_id', 'store_id'),\n"
    "    ('inventory.sku_id',     'sku_master.sku_id',     inventory, sku_master,   'sku_id', 'sku_id'),\n"
    "    ('inventory.date',       'calendar.date',         inventory, calendar_clean, 'date', 'date'),\n"
    "]\n"
    "\n"
    "ref_rows = []\n"
    "for label, _par, child, parent, ccol, pcol in relationships:\n"
    "    r = validate_referential_integrity(child, ccol, parent, pcol, 'child', 'parent')\n"
    "    ref_rows.append({'Relationship': label, 'Child Rows': r['total_child_rows'],\n"
    "                     'Valid Links': r['valid_relationships'], 'Orphans': r['orphan_records']})\n"
    "ref_df = pd.DataFrame(ref_rows)\n"
    "display(ref_df)\n"
    "print('Orphan records are reported, not fixed by fabricating master rows.')\n"
))

# ===========================================================================
# 13. OUTLIER INVESTIGATION
# ===========================================================================
cells.append(md(
    "### 13. Outlier Investigation\n"
    "\n"
    "Retail datasets naturally contain extreme but **legitimate** transactions "
    "(bulk orders, holiday spikes, clearance lots). We therefore *investigate* "
    "outliers using IQR and Z-score but **do not remove them** — removing them "
    "would bias the demand distribution and corrupt the forecasts in later "
    "phases. The counts below are reported for review, not excised."
))

cells.append(code(
    "# ---- UCI: Quantity & Price outliers -----------------------------------\n"
    "uci_out = outlier_report_for_dataset(uci_clean, ['Quantity', 'Price'], 'online_retail_ii')\n"
    "out_rows = []\n"
    "for col, r in uci_out.items():\n"
    "    out_rows.append({'Column': col, 'N': r['n'], 'IQR Outliers': r['iqr_outlier_count'],\n"
    "                     'IQR %': r['iqr_outlier_pct'], 'IQR Min': r['iqr_outlier_min'], 'IQR Max': r['iqr_outlier_max'],\n"
    "                     'Z>3 Outliers': r['z_outlier_count']})\n"
    "display(pd.DataFrame(out_rows))\n"
    "reports['online_retail_ii']['outliers'] = uci_out\n"
))

cells.append(code(
    "# ---- Synthetic: units_sold / revenue / inventory / lead-time outliers -\n"
    "syn_out_cols = [\n"
    "    (sales_daily['units_sold'], 'sales_daily.units_sold'),\n"
    "    (sales_daily['total_revenue'], 'sales_daily.total_revenue'),\n"
    "    (inventory['ending_inventory'], 'inventory.ending_inventory'),\n"
    "    (inventory['on_order_qty'], 'inventory.on_order_qty'),\n"
    "    (sku_master['lead_time_days'], 'sku_master.lead_time_days'),\n"
    "    (sku_master['base_price'], 'sku_master.base_price'),\n"
    "]\n"
    "syn_out_rows = []\n"
    "for series, colname in syn_out_cols:\n"
    "    r = investigate_outliers(series, colname, colname.split('.')[0])\n"
    "    syn_out_rows.append({'Column': colname, 'N': r['n'], 'IQR Outliers': r['iqr_outlier_count'],\n"
    "                         'IQR %': r['iqr_outlier_pct'], 'IQR Min': r['iqr_outlier_min'], 'IQR Max': r['iqr_outlier_max'],\n"
    "                         'Z>3 Outliers': r['z_outlier_count']})\n"
    "display(pd.DataFrame(syn_out_rows))\n"
    "\n"
    "# Attach outlier reports to the per-dataset reports (used in Section 16)\n"
    "reports['sales_daily']['outliers'] = {\n"
    "    'units_sold': investigate_outliers(sales_daily['units_sold'], 'units_sold', 'sales_daily'),\n"
    "    'total_revenue': investigate_outliers(sales_daily['total_revenue'], 'total_revenue', 'sales_daily'),\n"
    "}\n"
    "reports['inventory_snapshots']['outliers'] = {\n"
    "    'ending_inventory': investigate_outliers(inventory['ending_inventory'], 'ending_inventory', 'inventory_snapshots'),\n"
    "    'on_order_qty': investigate_outliers(inventory['on_order_qty'], 'on_order_qty', 'inventory_snapshots'),\n"
    "}\n"
    "reports['sku_master']['outliers'] = {\n"
    "    'lead_time_days': investigate_outliers(sku_master['lead_time_days'], 'lead_time_days', 'sku_master'),\n"
    "    'base_price': investigate_outliers(sku_master['base_price'], 'base_price', 'sku_master'),\n"
    "}\n"
    "print('\\nAll outliers are reported for business review — none were removed.')\n"
))

cells.append(code(
    "# ---- Outlier / quality visualisations --------------------------------\n"
    "fig_dir = FIGURES_DIR\n"
    "\n"
    "# 1. UCI transaction-type split\n"
    "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
    "tx_counts = uci_clean['transaction_type'].value_counts().reindex([TX_SALE, TX_RETURN, TX_CANCELLATION, TX_INVALID])\n"
    "colors = {'SALE': PALETTE[0], 'RETURN': PALETTE[2], 'CANCELLATION': PALETTE[1], 'INVALID': CRITICAL}\n"
    "bars = ax.bar(tx_counts.index, tx_counts.values, color=[colors[k] for k in tx_counts.index],\n"
    "              edgecolor='white', linewidth=0.8)\n"
    "ax.bar_label(bars, fmt=lambda v: f'{v/1e6:.2f}M' if v >= 1e6 else f'{v:,.0f}')\n"
    "ax.set_title('Online Retail II — Transaction Types after Cleaning')\n"
    "ax.set_ylabel('Lines')\n"
    "ax.tick_params(axis='x', rotation=0)\n"
    "plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'fig_uci_transaction_types.png')); plt.show()\n"
    "\n"
    "# 2. UCI Quantity distribution (log scale) — SALE only\n"
    "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
    "q = uci_clean.loc[uci_clean['transaction_type']==TX_SALE, 'Quantity']\n"
    "sns.histplot(q, bins=50, color=PALETTE[0], log_scale=(True, False), ax=ax)\n"
    "ax.set_title('Online Retail II — Quantity Distribution (SALE lines, log-x)')\n"
    "ax.set_xlabel('Quantity (units)'); ax.set_ylabel('Count')\n"
    "plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'fig_uci_quantity_distribution.png')); plt.show()\n"
    "\n"
    "# 3. UCI price special transactions\n"
    "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
    "pc = uci_clean['price_category'].value_counts().reindex(['PRICE_POSITIVE', 'PRICE_ZERO', 'PRICE_NEGATIVE'])\n"
    "pc_colors = [PALETTE[0], WARN, CRITICAL]\n"
    "bars = ax.bar(['Price > 0', 'Price = 0', 'Price < 0'], pc.values, color=pc_colors,\n"
    "              edgecolor='white', linewidth=0.8)\n"
    "ax.bar_label(bars, fmt=lambda v: f'{v:,.0f}')\n"
    "ax.set_title('Online Retail II — Price Category Distribution')\n"
    "ax.set_ylabel('Lines')\n"
    "plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'fig_uci_price_categories.png')); plt.show()\n"
))

cells.append(code(
    "# 4. Inventory balance: before (raw semantics) vs after (derived)\n"
    "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
    "vals = [inv_rep['canonical_equation_mismatch_rows'], len(inventory) - inv_rep['canonical_equation_mismatch_rows']]\n"
    "ax.bar(['Violates canonical equation', 'Satisfies canonical equation'], vals,\n"
    "       color=[CRITICAL, PALETTE[2]], edgecolor='white', linewidth=0.8)\n"
    "ax.set_title('Inventory — Canonical Balance Equation (raw columns)')\n"
    "ax.set_ylabel('Rows')\n"
    "for i, v in enumerate(vals):\n"
    "    ax.text(i, v, f'{v:,} ({v/len(inventory)*100:.1f}%)', ha='center', va='bottom')\n"
    "plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'fig_inventory_balance.png')); plt.show()\n"
    "\n"
    "# 5. Outlier boxplots (selected columns)\n"
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))\n"
    "sns.boxplot(x=uci_clean.loc[uci_clean['transaction_type']==TX_SALE, 'Quantity'].clip(upper=200), color=PALETTE[0], ax=axes[0])\n"
    "axes[0].set_title('UCI Quantity (SALE, clipped @ 200)')\n"
    "sns.boxplot(x=sales_daily['units_sold'].clip(upper=100), color=PALETTE[1], ax=axes[1])\n"
    "axes[1].set_title('Sales Daily units_sold (clipped @ 100)')\n"
    "sns.boxplot(x=inventory['ending_inventory'].clip(upper=500), color=PALETTE[2], ax=axes[2])\n"
    "axes[2].set_title('Inventory ending_inventory (clipped @ 500)')\n"
    "plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'fig_outlier_boxplots.png')); plt.show()\n"
))

# ===========================================================================
# 14. FINAL VALIDATION
# ===========================================================================
cells.append(md(
    "### 14. Final Validation\n"
    "\n"
    "Automated assertions verify the cleaned datasets satisfy every contract "
    "required by later phases: complete schemas, no duplicate keys, valid dates, "
    "valid numerics, valid foreign keys, the inventory balance identity, and "
    "that the Online Retail components recombine to the full cleaned frame."
))

cells.append(code(
    "# ===== SCHEMA =====\n"
    "assert set(ONLINE_RETAIL_REQUIRED_COLUMNS).issubset(uci_clean.columns), 'UCI schema incomplete'\n"
    "for key, req in [('store_master', STORE_MASTER_COLUMNS), ('sku_master', SKU_MASTER_COLUMNS),\n"
    "                 ('customer_master', CUSTOMER_MASTER_COLUMNS), ('calendar', CALENDAR_COLUMNS),\n"
    "                 ('sales_daily', SALES_DAILY_COLUMNS), ('inventory_snapshots', INVENTORY_COLUMNS)]:\n"
    "    assert set(req).issubset(clean[key].columns), f'{key} schema incomplete'\n"
    "print('PASS: required columns exist in all cleaned datasets')\n"
    "\n"
    "# ===== DUPLICATES / PRIMARY KEYS =====\n"
    "assert not uci_clean.duplicated().any(), 'UCI clean still contains duplicates'\n"
    "assert not store_master['store_id'].duplicated().any(), 'Duplicate store_id'\n"
    "assert not sku_master['sku_id'].duplicated().any(), 'Duplicate sku_id'\n"
    "assert not customer_master['customer_id'].duplicated().any(), 'Duplicate customer_id'\n"
    "assert calendar_clean['date'].duplicated().sum() == 0, 'Duplicate calendar date'\n"
    "assert not sales_daily.duplicated(['date','store_id','sku_id']).any(), 'Sales grain not unique'\n"
    "assert not inventory.duplicated(['date','store_id','sku_id']).any(), 'Inventory grain not unique'\n"
    "print('PASS: no duplicate primary keys / no duplicate fact grain')\n"
    "\n"
    "# ===== DATES =====\n"
    "assert int(uci_clean['InvoiceDate'].isna().sum()) == 0, 'UCI InvoiceDate contains invalid/missing dates'\n"
    "assert uci_clean['InvoiceDate'].between(pd.Timestamp('2009-01-01'), pd.Timestamp('2013-01-01')).all(), \\\n"
    "    'UCI InvoiceDate outside expected 2009-2011 range'\n"
    "assert int(calendar_clean['parsed_date'].isna().sum()) == 0, 'Calendar contains invalid dates'\n"
    "print('PASS: dates are valid and in expected ranges')\n"
    "\n"
    "# ===== NUMERICS =====\n"
    "assert (sales_daily['units_sold'] >= 0).all(), 'Negative units_sold'\n"
    "assert (sales_daily['total_revenue'] >= 0).all(), 'Negative total_revenue'\n"
    "assert (sales_daily['avg_unit_price'] >= 0).all(), 'Negative avg_unit_price'\n"
    "assert (sales_daily['transaction_count'] >= 0).all(), 'Negative transaction_count'\n"
    "assert (sales_daily['unique_customers'] >= 0).all(), 'Negative unique_customers'\n"
    "assert (inventory['beginning_inventory'] >= 0).all(), 'Negative beginning_inventory'\n"
    "assert (inventory['receipts'] >= 0).all(), 'Negative receipts'\n"
    "assert (inventory['ending_inventory'] >= 0).all(), 'Negative ending_inventory'\n"
    "assert (inventory['on_order_qty'] >= 0).all(), 'Negative on_order_qty'\n"
    "assert (sku_master['cost_price'] > 0).all() and (sku_master['base_price'] > 0).all(), 'Non-positive SKU prices'\n"
    "assert (store_master['store_size_sqft'] > 0).all(), 'Non-positive store size'\n"
    "print('PASS: numeric fields are non-negative / positive as required')\n"
    "\n"
    "# ===== FOREIGN KEYS =====\n"
    "assert sales_daily['store_id'].isin(store_master['store_id']).all(), 'sales.store_id orphans'\n"
    "assert sales_daily['sku_id'].isin(sku_master['sku_id']).all(), 'sales.sku_id orphans'\n"
    "assert sales_daily['date'].isin(calendar_clean['date']).all(), 'sales.date orphans'\n"
    "assert inventory['store_id'].isin(store_master['store_id']).all(), 'inv.store_id orphans'\n"
    "assert inventory['sku_id'].isin(sku_master['sku_id']).all(), 'inv.sku_id orphans'\n"
    "assert inventory['date'].isin(calendar_clean['date']).all(), 'inv.date orphans'\n"
    "print('PASS: foreign keys are valid (no orphans)')\n"
    "\n"
    "# ===== INVENTORY BALANCE =====\n"
    "assert (inventory['beginning_inventory_pre_receipts'] + inventory['receipts']\n"
    "        - inventory['units_sold'] == inventory['ending_inventory']).all(), 'Inventory balance violated'\n"
    "print('PASS: canonical inventory balance holds on derived basis')\n"
    "\n"
    "# ===== UCI COMPONENT RECOMBINATION =====\n"
    "assert len(uci_sales) + len(uci_returns) + len(uci_cancellations) + len(uci_invalid) == len(uci_clean), \\\n"
    "    'UCI splits do not recombine to the cleaned frame'\n"
    "print('PASS: Online Retail components recombine exactly to the cleaned frame')\n"
    "\n"
    "print('\\nALL FINAL VALIDATION ASSERTIONS PASSED.')\n"
))

# ===========================================================================
# 15. SAVE PROCESSED DATA
# ===========================================================================
cells.append(md(
    "### 15. Save Processed Data\n"
    "\n"
    "All cleaned datasets are written to `data/processed/`. The UCI dataset is "
    "persisted both as one full cleaned frame and as the analytical components "
    "(sales / returns / cancellations / invalid). Large fact tables use Parquet. "
    "**`data/raw/` is never modified.**"
))

cells.append(code(
    "# ---- Write processed datasets -----------------------------------------\n"
    "paths = save_processed_data(clean, uci_splits=(uci_sales, uci_returns, uci_cancellations, uci_invalid))\n"
))

cells.append(code(
    "# ---- Verify written outputs are readable and row counts match ----------\n"
    "expected_rows = {\n"
    "    'online_retail_clean': len(uci_clean),\n"
    "    'online_retail_sales': len(uci_sales),\n"
    "    'online_retail_returns': len(uci_returns),\n"
    "    'online_retail_cancellations': len(uci_cancellations),\n"
    "    'online_retail_invalid': len(uci_invalid),\n"
    "    'store_master_clean': len(store_master),\n"
    "    'sku_master_clean': len(sku_master),\n"
    "    'customer_master_clean': len(customer_master),\n"
    "    'calendar_clean': len(calendar_clean),\n"
    "    'sales_daily_clean': len(sales_daily),\n"
    "    'inventory_snapshots_clean': len(inventory),\n"
    "}\n"
    "verify = []\n"
    "for key, p in paths.items():\n"
    "    n = len(pd.read_parquet(p)) if p.endswith('.parquet') else len(pd.read_csv(p))\n"
    "    exp = expected_rows.get(key)\n"
    "    verify.append({'File': os.path.basename(p), 'Rows': n, 'Expected': exp, 'OK': n == exp})\n"
    "verify_df = pd.DataFrame(verify)\n"
    "display(verify_df)\n"
    "assert verify_df['OK'].all(), 'Processed file row-count mismatch'\n"
    "print('\\nAll processed files verified (row counts match in-memory cleaned frames).')\n"
))

# ===========================================================================
# 16. DATA QUALITY SUMMARY
# ===========================================================================
cells.append(md(
    "### 16. Data Quality Summary\n"
    "\n"
    "The data-quality report is persisted to `docs/data_quality_report.json` "
    "(full detail) and `docs/data_quality_report.csv` (flat per-dataset "
    "summary). The table below summarises the before/after state of every "
    "dataset."
))

cells.append(code(
    "# ---- Generate the data-quality report ---------------------------------\n"
    "summary_df = generate_quality_report(reports)\n"
    "display(summary_df)\n"
))

cells.append(code(
    "# ---- Before / After summary table -------------------------------------\n"
    "before_after = pd.DataFrame([\n"
    "    {'Dataset': 'Online Retail II (UCI)', 'Original': uci_report['original_rows'],\n"
    "     'Duplicates Found': uci_report['duplicate_rows'], 'Duplicates Removed': uci_report['duplicates_removed'],\n"
    "     'Missing Values': uci_report['missing_values'], 'Invalid Values': uci_report['invalid_values'],\n"
    "     'Final': uci_report['final_rows'], 'Status': uci_report['quality_status']},\n"
    "    {'Dataset': 'Store Master', 'Original': store_rep['original_rows'],\n"
    "     'Duplicates Found': store_rep['duplicate_rows'], 'Duplicates Removed': 0,\n"
    "     'Missing Values': 0, 'Invalid Values': store_rep['invalid_values'],\n"
    "     'Final': store_rep['final_rows'], 'Status': store_rep['quality_status']},\n"
    "    {'Dataset': 'SKU Master', 'Original': sku_rep['original_rows'],\n"
    "     'Duplicates Found': sku_rep['duplicate_rows'], 'Duplicates Removed': 0,\n"
    "     'Missing Values': 0, 'Invalid Values': sku_rep['invalid_values'],\n"
    "     'Final': sku_rep['final_rows'], 'Status': sku_rep['quality_status']},\n"
    "    {'Dataset': 'Customer Master', 'Original': cust_rep['original_rows'],\n"
    "     'Duplicates Found': cust_rep['duplicate_rows'], 'Duplicates Removed': 0,\n"
    "     'Missing Values': 0, 'Invalid Values': cust_rep['invalid_values'],\n"
    "     'Final': cust_rep['final_rows'], 'Status': cust_rep['quality_status']},\n"
    "    {'Dataset': 'Calendar', 'Original': cal_rep['original_rows'],\n"
    "     'Duplicates Found': cal_rep['duplicate_rows'], 'Duplicates Removed': 0,\n"
    "     'Missing Values': 0, 'Invalid Values': cal_rep['invalid_values'],\n"
    "     'Final': cal_rep['final_rows'], 'Status': cal_rep['quality_status']},\n"
    "    {'Dataset': 'Sales Daily', 'Original': sales_rep['original_rows'],\n"
    "     'Duplicates Found': sales_rep['duplicate_rows'], 'Duplicates Removed': 0,\n"
    "     'Missing Values': 0, 'Invalid Values': sales_rep['invalid_values'],\n"
    "     'Final': sales_rep['final_rows'], 'Status': sales_rep['quality_status']},\n"
    "    {'Dataset': 'Inventory Snapshots', 'Original': inv_rep['original_rows'],\n"
    "     'Duplicates Found': inv_rep['duplicate_rows'], 'Duplicates Removed': 0,\n"
    "     'Missing Values': 0, 'Invalid Values': inv_rep['invalid_values'],\n"
    "     'Final': inv_rep['final_rows'], 'Status': inv_rep['quality_status']},\n"
    "])\n"
    "display(before_after)\n"
))

cells.append(code(
    "# ---- Before / after visualisation -------------------------------------\n"
    "fig, ax = plt.subplots(figsize=(11, 5))\n"
    "datasets = before_after['Dataset']\n"
    "x = np.arange(len(datasets))\n"
    "w = 0.38\n"
    "ax.bar(x - w/2, before_after['Original'], w, label='Original', color=PALETTE[0], edgecolor='white', linewidth=0.8)\n"
    "ax.bar(x + w/2, before_after['Final'], w, label='Final (cleaned)', color=PALETTE[2], edgecolor='white', linewidth=0.8)\n"
    "ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=25, ha='right')\n"
    "ax.set_yscale('log')\n"
    "ax.set_ylabel('Rows (log scale)')\n"
    "ax.set_title('Before / After — Rows per Dataset')\n"
    "ax.legend(frameon=False)\n"
    "plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, 'fig_before_after_rows.png')); plt.show()\n"
))

cells.append(md(
    "### 16.1 Cleaning Decisions — Summary\n"
    "\n"
    "**Online Retail II (UCI)**\n"
    "\n"
    "- **Exact duplicates (34,335 rows, 3.22%)** were confirmed as exact "
    "full-row repeats and removed; the raw file was not touched.\n"
    "- **Cancellations (19,104 lines, 1.79% of raw)** were identified by the "
    "`C` invoice prefix and kept in `online_retail_cancellations.csv` — not "
    "deleted.\n"
    "- **Returns (3,393 non-C negative-quantity lines)** were kept in "
    "`online_retail_returns.csv`. Many carry zero price and act as inventory "
    "adjustment lines; they remain available for return analytics.\n"
    "- **Invalid / accounting lines (6)** — the `A`-prefixed `Adjust bad debt` "
    "journal lines — were separated and excluded from sales.\n"
    "- **Zero-price special lines (6,014)** (postage, samples, manual "
    "adjustments) were kept in the SALE stream and flagged with "
    "`is_special_transaction` for downstream filtering.\n"
    "- **Missing Descriptions (4,382)** were recovered from the same StockCode "
    "wherever possible (4,275 after dedup; 3,912 recovered, 363 unrecoverable "
    "labelled `Unknown Product`).\n"
    "- **Missing Customer ID (22.76% after dedup)** was treated as guest "
    "checkout, kept for demand analysis and flagged with `is_guest_transaction`.\n"
    "\n"
    "**Synthetic multi-store dataset**\n"
    "\n"
    "- All master tables passed: unique IDs, valid prices/lead-times/sizes, "
    "valid segments and loyalty flags, internally consistent calendar.\n"
    "- `sales_daily` passed: unique `(date, store, sku)` grain, non-negative "
    "numbers, `revenue == units * price`, promotion flags `{0,1}`, and zero "
    "orphan references.\n"
    "- `inventory_snapshots`: the canonical balance equation fails on 8.36% of "
    "rows because `beginning_inventory` **already includes** the day's "
    "receipts. We documented this semantic, added the derived column "
    "`beginning_inventory_pre_receipts`, and verified the canonical equation "
    "holds on 100% of rows on that basis.\n"
    "\n"
    "**Outliers** were investigated (IQR + Z-score) but **not removed** — they "
    "represent legitimate extreme retail transactions.\n"
    "\n"
    "**Phase 3 is complete.** No EDA, feature engineering, forecasting or "
    "visualisation app has been started, per the phase boundary."
))

cells.append(md(
    "---\n"
    "**End of Phase 3.** Next step (after approval): **Phase 4 — Data "
    "Integration & Common Analytical Model**."
))

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "nbformat": 4,
        "nbformat_minor": 4,
    },
}


def main():
    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "02_data_cleaning.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Created {nb_path} successfully ({len(cells)} cells).")


if __name__ == "__main__":
    main()
