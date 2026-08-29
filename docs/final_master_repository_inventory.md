# Project Foresight — Final Master Repository Inventory

**Audit date:** 2026-08-29  
**Method:** Independent file-system scan + import checks + pytest (286 collected)

---

## 1. Top-level structure

| Path | Role | File count (approx.) |
|------|------|----------------------|
| `src/` | Pipelines, API, auth, phase runners | 145 Python modules |
| `tests/` | Automated validation | 21 test modules, 286 tests |
| `dashboard/` | Streamlit unified app + legacy dashboards | 29 Python modules |
| `docs/` | Phase reports, registries, audits | 97 Markdown + 92 JSON |
| `data/` | Raw, processed, phase artifacts | 115 parquet + 36 CSV |
| `models/` | Frozen + production + candidates | 21 joblib files |
| `public/` | Vercel SPA (HTML/CSS/JS) | Static frontend |
| `outputs/` | BI exports, monitoring copies | Generated artifacts |

---

## 2. Phase discovery (from code + docs)

| Phase | Primary entry / evidence | Tests |
|-------|--------------------------|-------|
| 5 | EDA notebooks, `compute_exact_eda_tables.py` | Indirect via integration |
| 6 | `feature_engineering.py`, `validate_features.py` | `test_validation.py` |
| 7–8 | `ml_forecasting.py`, `final_forecasting.py` | `test_inference.py` |
| 9 | `run_phase9.py`, `validate_phase9.py` | Artifact regression |
| 10 | `run_phase10.py`, `validate_phase10.py` | Artifact regression |
| 11 | `run_phase11.py`, `final_model_registry.json` | `test_regression_artifacts.py` |
| 12 | `validate_phase12.py` | Historical 41/42 (legacy artifact) |
| 13 | `validate_phase13.py` | `test_security.py` (18) |
| 14 | `validate_phase14.py` | `test_phase14_e2e.py` (6) |
| 15 | `validate_phase15.py` | `test_phase15_bi.py` (10) |
| 16 | `validate_phase16_datasets.py` | `test_phase16_dataset_alignment.py` (19) |
| 17 | `run_phase17.py`, `phase17_*.py` | 28 tests across 4 files |
| 18 | `phase18_promotion_gate.py` | `test_phase18_promotion_gate.py` (28) |
| 19 | `run_phase19.py`, `phase19_*.py` | `test_phase19_hardening.py` (25) |
| 20 | `run_phase20.py`, `phase20_*` | `test_phase20_production_integration.py` (21) |
| 21 | `run_phase21.py`, `phase21_*` | `test_phase21_monitoring.py` (24) |
| 22 | `run_phase22.py`, `phase22_final_audit.py` | `test_phase22_final_delivery.py` (27) |
| 23 | Auth, navigation, UI | `test_phase23_*` (45) |

---

## 3. API routes (verified in source)

| Router | Prefix | Key endpoints |
|--------|--------|---------------|
| `src/api/routes.py` | `/` | `/health`, `/ready`, `/forecast`, `/forecast/batch`, `/model` |
| `src/api/phase20_routes.py` | `/phase20` | `/model`, `/forecast`, `/risk/explain`, `/contract` |
| `src/api/phase21_routes.py` | `/phase21` | `/health`, `/monitoring/latest`, `/alerts`, `/integrity` |
| `src/api/auth_routes.py` | `/auth` | `/register`, `/login`, `/me`, `/logout` |

Application factory: `src/api/app.py` → `create_app()`

---

## 4. Dashboard surfaces

| Surface | Entry | Auth |
|---------|-------|------|
| Unified Streamlit | `app.py` + `dashboard/pages/*` | `dashboard/session_auth.py` |
| Legacy Phase 20 | `dashboard/phase20_production.py` | Standalone |
| Legacy Phase 21 | `dashboard/phase21_monitoring.py` | Standalone |
| Legacy Phase 22 | `dashboard/phase22_executive_dashboard.py` | Standalone |
| Vercel SPA | `public/index.html` + `public/js/app.js` | JWT via `/auth/*` |

Navigation config: `dashboard/navigation.py` (28 Streamlit pages)

---

## 5. Models inventory

| Location | Count | Purpose |
|----------|-------|---------|
| `models/final/` | 12 | Frozen Phase 11 registry models |
| `models/final/phase20/` | 1 | Production promoted model |
| `models/phase17/` | 2 | Phase 17 candidates (UCI + Synthetic) |
| `models/phase19/` | 1 | Phase 19 synthetic candidate (parent of Phase 20) |
| `models/` (root) | 4 | Legacy / best-model copies including `lightgbm_forecaster.joblib` |

Registries: `docs/final_model_registry.json`, `docs/phase20_production_registry.json`

---

## 6. Data artifacts (key paths)

| Dataset | Raw source | Phase processed |
|---------|------------|-----------------|
| UCI | `data/raw/online_retail_II.csv` | `data/phase17/processed/uci_weekly_demand.parquet` |
| Synthetic | `data/raw/sales_daily.parquet` (+ masters) | `data/phase17/processed/synthetic_weekly_demand.parquet` |
| Production forecasts | — | `data/phase20/production_forecasts.parquet` |
| Monitoring | — | `data/phase21/monitoring/*.json` |

Inventory scripts: `src/dataset_inventory_and_validation.py`  
Reports: `docs/complete_dataset_inventory.md`, `docs/dataset_source_integrity.json`

---

## 7. Deployment files

| File | Purpose |
|------|---------|
| `vercel.json` | Static frontend + `/api` proxy to Render |
| `Dockerfile` | Container build |
| `requirements.txt` / `pyproject.toml` | Python dependencies |
| `.env.example` | Environment template |
| `docs/phase22_deployment_guide.md` | Deployment instructions |

Live URLs (configured, not all verified this audit):
- Frontend: https://foresight-project-green.vercel.app/
- Backend: https://project-foresight-api-tofn.onrender.com/

---

## 8. Configuration

- `src/config.py` — paths, model registry references
- `src/auth/config.py` — JWT / auth env aliases
- `.streamlit/config.toml` — Streamlit theme

---

## 9. Test inventory (286 total)

| Test file | Tests |
|-----------|-------|
| `test_phase18_promotion_gate.py` | 28 |
| `test_phase22_final_delivery.py` | 27 |
| `test_phase23_navigation.py` | 27 |
| `test_phase19_hardening.py` | 25 |
| `test_phase21_monitoring.py` | 24 |
| `test_phase20_production_integration.py` | 21 |
| `test_phase16_dataset_alignment.py` | 19 |
| `test_phase23_authentication.py` | 18 |
| `test_security.py` | 18 |
| `test_api.py` | 12 |
| Others | 76 |

**Runtime (2026-08-29):** 286 passed, 0 failed, 0 skipped, 2 warnings
