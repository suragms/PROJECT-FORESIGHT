# Phase 16 — Compatibility Matrix

**Project:** Project FORESIGHT — Demand & Inventory Intelligence  
**Phase:** 16

This matrix assesses each pipeline component against the three data scenarios:
- **Current data** — what is actually in `data/raw/`
- **UCI** — `online_retail_II.csv`
- **Kaggle Synthetic 10M** — not loaded; compatibility is projected based on documented schema

---

## Compatibility Matrix

| Component | Current Data | UCI | Kaggle Synthetic 10M | Compatible? | Action Required |
|---|---|---|---|---|---|
| **Data pipeline** (`src/data_cleaning.py`, `src/data_integration.py`) | WORKS — both UCI + synthetic processed | WORKS — UCI path tested | UNKNOWN — not loaded; would require schema mapping | PARTIAL | If Kaggle 10M loaded: verify column names match expected schema before cleaning |
| **EDA** (Phase 2) | COMPLETE | COMPLETE | NOT DONE | PARTIAL | Kaggle 10M EDA would be needed in Phase 17 |
| **Seasonal-naive baseline** (`src/baseline_forecasting.py`) | WORKS — daily grain | WORKS | UNKNOWN | PARTIAL | Baseline is data-agnostic; would work if grain is daily |
| **Feature engineering** (`src/feature_engineering.py`) | WORKS — lag/rolling/calendar computed correctly | WORKS — null inventory/promo accepted | UNKNOWN — depends on column names | PARTIAL | If Kaggle 10M has different column names, feature engineering needs mapping |
| **LightGBM** (frozen `uci_h1_phase8_lightgbm`) | VALIDATED on current UCI data | VALIDATED | NOT VALIDATED — not trained on Kaggle 10M | PARTIAL for new data | Do not present as Kaggle-trained; retrain required for Kaggle compatibility |
| **Hurdle model** (frozen `synthetic_h1_hurdle_th050`) | VALIDATED on current synthetic (10 stores / 100 SKUs) | N/A | NOT VALIDATED | PARTIAL for new data | Do not present as Kaggle-trained; retrain required for Kaggle compatibility |
| **Risk scoring** (`src/risk_scoring.py`) | WORKS — uses historical demand average | PARTIAL — no inventory available | UNKNOWN | PARTIAL | Risk engine uses historical avg demand, not forecast; Zidio spec requires forecast-driven; gap documented |
| **Dashboard** (`dashboard/app.py`, `dashboard/executive_intelligence.py`) | WORKS | PARTIAL — UCI rows displayed | UNKNOWN | PARTIAL | Dashboards are schema-agnostic; if BI exports are regenerated correctly, they will work |
| **API** (`src/api/`) | WORKS — serves frozen predictions | WORKS | UNKNOWN | PARTIAL | API is model-agnostic; works with any registered model |
| **BI exports** (`src/bi/exports.py`) | WORKS — 1,000-row extract | PARTIAL — UCI forecasts included | UNKNOWN | PARTIAL | BI exports depend on forecast outputs and inventory risk; both would need regeneration for Kaggle data |
| **Monitoring** (`src/monitoring/`) | WORKS | WORKS | UNKNOWN | PARTIAL | Monitoring reads monitoring JSONs; data-agnostic |
| **Rupee impact** | MISSING | MISSING | UNKNOWN | FAIL | Zidio brief specifies Rupee; no currency conversion implemented |
| **Weekly aggregation layer** | MISSING | MISSING | UNKNOWN | FAIL | Zidio requires weekly SKU forecast; model is daily-grain only |

---

## Risk Engine Compatibility Detail

| Risk Input | Current Implementation | Zidio Spec | Gap |
|---|---|---|---|
| Historical average demand | YES — primary demand signal | Supplementary | Used as primary; should be supplementary |
| Forecast demand | NO | Primary signal | **MISSING** — documented gap |
| Lead time | YES — from sku_master | Required | PASS |
| On-hand inventory | YES — ending_inventory | Required | PASS |
| On-order quantity | YES — on_order_qty | Required | PASS |
| Safety stock | YES — from sku_master | Required | PASS |
| Reorder point | YES — from sku_master | Required | PASS |
| Forward demand window (overstock) | YES — `target_coverage_days` parameter | Required | PASS |

---

## Model Compatibility Summary

| Model | Trained on | Valid for current data | Valid for Kaggle 10M |
|---|---|---|---|
| `uci_h1_phase8_lightgbm` | UCI Online Retail II (this repository) | YES | NOT YET VALIDATED |
| `synthetic_h1_hurdle_th050` | Repository synthetic (10 stores / 100 SKUs, seed 42) | YES | NOT YET VALIDATED |

Neither frozen model can be legitimately presented as trained on the Kaggle Synthetic Retail 10M dataset because that dataset has not been loaded, cleaned, or used to train any model in this repository.

---

## Overall Classification

| Area | Classification |
|---|---|
| Current data vs frozen stack | GREEN — stack is consistent with current data |
| Current data vs Kaggle 10M | YELLOW/RED — not loaded; not validated |
| Business scope vs Zidio brief | YELLOW — multi-store extension; scope deviation documented |
| Risk engine vs Zidio spec | YELLOW — functional but not fully forecast-driven |
| Missing Rupee impact | RED (for NorthBay deliverable) |
| Missing weekly aggregation | YELLOW |
