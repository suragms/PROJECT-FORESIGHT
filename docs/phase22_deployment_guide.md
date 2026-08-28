# Phase 22 — Deployment Guide

## 1. Environment Setup

**Python version:** 3.12 (verified in `requirements.txt`)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

Alternative with `uv`:
```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

## 2. Environment Variables

Copy `.env.example` to `.env` (do not commit `.env`):

| Variable | Purpose |
|----------|---------|
| `FORESIGHT_API_AUTH_ENABLED` | Enable API key authentication |
| `FORESIGHT_API_API_KEY` | API key when auth enabled |
| `FORESIGHT_ENV` | Environment label |

## 3. Data Paths

| Path | Purpose |
|------|---------|
| `data/raw/` | Raw source files |
| `data/phase19/features/` | Weekly feature matrix |
| `data/phase20/` | Production forecasts and risk |
| `data/phase21/monitoring/` | Monitoring reports |

## 4. Model Paths

| Path | Purpose |
|------|---------|
| `models/final/` | 12 frozen Phase 11 models |
| `models/final/phase20/` | Promoted production model |
| `models/phase17/`, `models/phase19/` | Candidate lineage (unchanged) |

**Do not retrain or modify frozen artifacts.**

## 5. Running FastAPI

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Verify: `http://127.0.0.1:8000/docs`

## 6. Running Dashboards

```bash
# Phase 20 — Production forecast view
streamlit run dashboard/phase20_production.py

# Phase 21 — Monitoring observability
streamlit run dashboard/phase21_monitoring.py

# Phase 22 — Executive business view
streamlit run dashboard/phase22_executive_dashboard.py
```

## 7. Running Monitoring

```bash
python src/run_phase21.py
```

Outputs: `data/phase21/monitoring/*.json`

## 8. Running Tests

```bash
python -m pytest tests -q
```

Phase-specific:
```bash
python -m pytest tests/test_phase21_monitoring.py -q
python -m pytest tests/test_phase22_final_delivery.py -q
```

## 9. Final Audit

```bash
python src/phase22_final_audit.py
```

Produces `docs/phase22_integrity_snapshot.json` and `docs/phase22_final_audit.json`.

## 10. Production Monitoring

Phase 21 monitoring is file-based observability. It does not automatically retrain or replace models. Review alerts in `data/phase21/monitoring/alerts.json`.

## 11. Rollback Procedure

Documented in `docs/phase20_rollback_plan.md`:

1. Stop serving Phase 20 routes if needed
2. Revert to prior dashboard/API configuration
3. Phase 20 model is a copy — original Phase 19 candidate remains at `models/phase19/synthetic/phase19_synthetic_lightgbm.joblib`
4. Do not modify frozen `models/final/` 12-model registry

## 12. Cloud / Docker Deployment

**NOT IMPLEMENTED IN CURRENT REPOSITORY**

A local Docker configuration may exist from earlier phases (`Dockerfile`), but cloud deployment, TLS, identity provider, and autoscaling are not deployed. Do not claim cloud production exists.

## Known Legacy Issue

`models/lightgbm_forecaster.joblib` — **LEGACY NON-PRODUCTION ARTIFACT ISSUE** (stale hash). Not part of frozen production stack.
