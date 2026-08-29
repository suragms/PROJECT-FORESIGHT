# Phase 23 — Unified Application Navigation

## Overview

Phase 23 adds a **unified Streamlit application** with left-sidebar navigation, grouping all Phase 20–22 capabilities into one professional analytics platform.

**Entry point:** `streamlit run app.py`

## Architecture

```
app.py                          # Unified entry point
dashboard/
  navigation.py                 # Nav groups and page keys
  components/
    sidebar.py                  # Brand + navigation
    theme.py                    # Design system CSS
    status_cards.py             # Status badges
    data_loader.py              # Cached Phase 20–22 data
    charts.py                   # Shared charts
  pages/
    home.py                     # Home
    executive.py                # Phase 22 executive view
    forecasting.py              # Forecasting group
    inventory.py                # Inventory intelligence
    analytics.py                # Analytics group
    ml.py                       # Machine learning group
    monitoring.py               # Phase 21 monitoring
    system.py                   # Model info, docs, about
```

## Navigation Groups

| Group | Pages |
|-------|-------|
| Overview | Home, Executive Dashboard |
| Forecasting | Demand Forecasting, Forecast Explorer, Horizon Analysis |
| Inventory Intelligence | Overview, Stockout, Overstock, Recommendations |
| Analytics | Business Analytics, Demand Trends, SKU Analysis, Seasonality, Performance |
| Machine Learning | Model Overview, Feature Contract, Performance, Explainability |
| Monitoring | System Health, Data Quality, Drift, Alerts, Integrity |
| System | Model Information, Documentation, Validation, About |

## Data Sources (Reuse Strategy)

| Page Group | Source |
|------------|--------|
| Forecasting / Inventory | `src.phase20_dashboard_adapter`, `src.phase22_executive_adapter` |
| Monitoring | `data/phase21/monitoring/*.json` via Phase 21 artifacts |
| ML / Features | `docs/phase20_feature_contract.json` |
| Analytics | `data/phase19/features/synthetic_weekly_features.parquet` |

No production logic is duplicated or modified.

## Backward Compatibility

These standalone dashboards remain unchanged:

- `dashboard/phase20_production.py`
- `dashboard/phase21_monitoring.py`
- `dashboard/phase22_executive_dashboard.py`
- `dashboard/app.py` (legacy Phase 1–15 app)

API routes `/phase20/*` and `/phase21/*` are unchanged.

## How to Run

```bash
streamlit run app.py
```

## Design Principles

- Validation metrics labeled **VALIDATION / BACKTEST**
- Live performance: **PENDING ACTUALS**
- Unsupported metrics display **NOT AVAILABLE**
- Semantic status badges (PASS / WARNING / FAIL / PENDING)
- No model retraining or artifact modification
