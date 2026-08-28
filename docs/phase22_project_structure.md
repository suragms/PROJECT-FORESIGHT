# Phase 22 — Project Structure

```
Demand-Inventory-Intelligence/
├── data/
│   ├── raw/                    # Raw source files (CSV, parquet)
│   ├── processed/              # Cleaned datasets
│   ├── phase17/                # Phase 17 integration artifacts
│   ├── phase19/                # Features, backtests (production reference)
│   ├── phase20/                # Production forecasts and risk
│   └── phase21/monitoring/     # Monitoring reports and history
│
├── models/
│   ├── final/                  # 12 frozen Phase 11 models
│   │   └── phase20/            # Promoted production model
│   ├── phase17/                # Phase 17 candidates
│   └── phase19/                # Phase 19 hardened candidate
│
├── src/
│   ├── api/                    # FastAPI app and route modules
│   │   ├── app.py              # Main application
│   │   ├── phase20_routes.py   # Production API
│   │   └── phase21_routes.py   # Monitoring API
│   ├── phase17_*.py            # Dataset integration
│   ├── phase18_*.py            # Promotion gate
│   ├── phase19_*.py            # Hardening
│   ├── phase20_*.py            # Production promotion & adapters
│   ├── phase21_*.py            # Monitoring modules
│   ├── phase22_*.py            # Final delivery & audit
│   ├── forecasting/            # Inference and registry
│   ├── monitoring/             # Phase 11 monitoring (legacy)
│   └── run_phase*.py           # Phase pipeline entry points
│
├── dashboard/
│   ├── phase20_production.py   # Production operational view
│   ├── phase21_monitoring.py   # Observability dashboard
│   └── phase22_executive_dashboard.py  # Executive business view
│
├── docs/
│   ├── phase17–phase21 reports   # Phase deliverables
│   ├── phase22_*.md              # Final delivery documentation
│   ├── PROJECT_FORESIGHT_FINAL_REPORT.md
│   ├── phase20_feature_contract.json
│   └── phase20_production_registry.json
│
├── tests/
│   ├── test_phase17_*.py       # Phase 17 tests
│   ├── test_phase20_*.py       # Phase 20 tests
│   ├── test_phase21_monitoring.py
│   └── test_phase22_final_delivery.py
│
├── outputs/
│   ├── bi/                     # BI parquet exports
│   └── monitoring/             # Phase 11 monitoring snapshots
│
├── requirements.txt
├── README.md
└── Dockerfile                  # Local Docker (cloud not deployed)
```

## Directory Roles

| Directory | Role |
|-----------|------|
| `data/` | All datasets and phase-specific artifacts |
| `models/` | Frozen and promoted model files (do not retrain) |
| `src/` | Application logic, APIs, pipelines |
| `dashboard/` | Streamlit dashboards |
| `docs/` | Reports, guides, contracts, audit results |
| `tests/` | Automated test suite |
| `outputs/` | Generated exports and monitoring snapshots |

## Entry Points

| Command | Purpose |
|---------|---------|
| `python src/run_phase21.py` | Run monitoring |
| `python src/phase22_final_audit.py` | Final delivery audit |
| `uvicorn src.api.app:app` | Start API |
| `streamlit run dashboard/phase22_executive_dashboard.py` | Executive dashboard |
| `python -m pytest tests -q` | Full regression |
