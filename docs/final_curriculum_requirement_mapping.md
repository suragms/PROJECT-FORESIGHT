# Project Foresight — Curriculum Requirement Mapping

**Audit date:** 2026-08-29  
**Rule:** PASS only when implementation evidence exists in code/tests/artifacts.

| # | Requirement | Implementation evidence | File(s) | Status |
|---|-------------|-------------------------|---------|--------|
| 1 | Data Science | End-to-end DS pipeline from raw retail data to production forecasts | `src/run_phase17.py` … `run_phase20.py`, `data/phase17/` | **PASS** |
| 2 | Data Analytics | Executive KPIs, BI exports, business impact metrics | `src/bi/`, `outputs/bi/`, `phase22_executive_adapter.py` | **PASS** |
| 3 | Data Cleaning | UCI + synthetic cleaning, validation reports | `src/data_cleaning.py`, `data/phase17/ingestion_manifest.json` | **PASS** |
| 4 | Data Preprocessing | Integration, weekly aggregation, dtype handling | `src/data_integration.py`, `phase17_dataset_ingestion.py` | **PASS** |
| 5 | Exploratory Data Analysis | EDA notebooks + exact table computation | `src/compute_exact_eda_tables.py`, `notebooks/` | **PASS** |
| 6 | Data Visualization | Streamlit dashboards, Chart.js SPA, monitoring charts | `dashboard/`, `public/js/app.js` | **PASS** |
| 7 | Feature Engineering | 46-col weekly features, lags, rolling, calendar | `src/phase17_features.py`, `phase19_features.py` | **PASS** |
| 8 | Machine Learning | LightGBM, hurdle, direct multi-horizon, quantile | `src/ml_forecasting.py`, `models/final/` | **PASS** |
| 9 | Performance Evaluation Metrics | WAPE, MAE, RMSE, bias, coverage | `src/evaluation.py`, phase backtest JSON | **PASS** |
| 10 | Predictive Analytics | Demand forecasting + risk scoring | `phase17_forecasting.py`, `risk_scoring.py` | **PASS** |
| 11 | Time Series Analysis | Weekly grain, seasonality, horizon analysis | `phase9_horizon_analysis.py`, seasonality pages | **PASS** |
| 12 | Model Validation | Rolling-origin backtests, leakage audit | `phase17_leakage_audit.py`, backtest parquets | **PASS** |
| 13 | Cross Validation / Backtesting | 5-fold rolling origin (Phase 17 manifest) | `data/phase17/backtests/` | **PASS** |
| 14 | Deployment | Docker, Render docs, Vercel static deploy | `Dockerfile`, `vercel.json`, deployment guide | **PARTIAL** — live backend health not verified (timeout) |
| 15 | API Integration | FastAPI scoring + Phase 20/21 routes | `src/api/` | **PASS** (local tests); **PARTIAL** (live) |
| 16 | Dashboard | Streamlit unified + Vercel SPA + legacy phase dashboards | `app.py`, `public/` | **PASS** |
| 17 | Cloud Deployment | Vercel + Render URLs configured | README, `vercel.json` | **PARTIAL** — frontend HTTP 200; backend not verified |
| 18 | Industry-Level Project Structure | Phased docs, registries, tests, CI-ready layout | `docs/`, `tests/`, registries | **PASS** |

## Not implemented (correctly absent)

| Topic | Status | Evidence |
|-------|--------|----------|
| NLP | **NOT IMPLEMENTED** | No NLP modules in `src/` |
| CNN / deep learning vision | **NOT IMPLEMENTED** | No CNN/TensorFlow/PyTorch training |
| Hadoop / Spark | **NOT IMPLEMENTED** | No Spark/Hadoop dependencies or jobs |
| Customer segmentation / churn ML apps | **NOT IMPLEMENTED** | No standalone routes; not in navigation |
| Automated model retraining in production | **NOT IMPLEMENTED** | Phase 21 observability-only policy |

## Internship requirement match summary

**Overall: PARTIAL** — core DS/ML/analytics/deployment stack is implemented and tested locally; official Kaggle synthetic 10M archive not extracted; live Render API not verified in this audit run.
