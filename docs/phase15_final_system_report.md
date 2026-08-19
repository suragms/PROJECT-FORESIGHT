# Phase 15 — Final system report

This is the project-completion report for **Demand & Inventory Intelligence (Project FORESIGHT)**. Forecasting models remain frozen at Phase 11 selection. Phase 15 adds executive BI, KPI exports, and documentation.

## Architecture (as implemented)

```
                    ┌──────────────────────┐
                    │ Historical Data      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Data Validation      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Frozen ML Models     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Forecast API         │
                    └──────────┬───────────┘
              ┌────────────────┴────────────────┐
              ↓                                 ↓
     ┌──────────────────┐             ┌──────────────────┐
     │ Forecast Output  │             │ Monitoring       │
     └────────┬─────────┘             └──────────────────┘
              ↓
     ┌──────────────────┐
     │ Inventory Risk   │
     └────────┬─────────┘
              ↓
     ┌──────────────────┐
     │ Recommendations  │
     └────────┬─────────┘
              ↓
     ┌──────────────────┐
     │ BI / Dashboard    │
     └──────────────────┘
```

### Component status

| Component | Status |
| --- | --- |
| Historical UCI + SYNTHETIC extracts | Implemented (local files) |
| Data validation / cleaning | Implemented (Phase 3) |
| Feature engineering | Implemented (frozen Phase 6 matrix) |
| Frozen ML models + registry hashes | Implemented (Phase 11; hashes unchanged through Phase 15) |
| Forecast API (`/health`, `/ready`, `/forecast`) | Implemented (local; auth configurable) |
| Forecast output parquet | Implemented |
| Monitoring JSON snapshots | Implemented (file-based; retraining disabled) |
| Inventory risk extract | Reference (1,000 rows) |
| Recommendations | Implemented (decision support only) |
| Streamlit dashboards | Implemented (local) |
| BI parquet exports | Implemented (`outputs/bi/`) |
| Power BI semantic model / workspace | Not deployed |
| Cloud hosting / TLS / IdP / secrets manager / autoscaling | Not deployed |
| Automated retraining | Not deployed (disabled) |
| Live warehouse / ERP write-back | Not deployed |

## Data sources

* **UCI Online Retail II** — invoice lines 2009–2011; forecast grain product × invoice-day, entity `ONLINE`.
* **SYNTHETIC relational retail** — stores, SKUs, daily sales and inventory; forecast grain store × SKU × day.
* Inventory intelligence on disk uses `outputs/risk_scores/inventory_risk_matrix.parquet` only (1,000 rows).

## Forecasting (frozen)

| Dataset | Horizon 1 | Horizons 3/7/14/30 |
| --- | --- | --- |
| UCI | `uci_h1_phase8_lightgbm` SHA-256 `331909f0fe191c0b9cb0418884b25eb59012f479f61f8b3e2ad51b729273e90d` | Direct LightGBM |
| SYNTHETIC | `synthetic_h1_hurdle_th050` SHA-256 `59a2b72024861d7f9c827596a52256af95facabfa796bdae5955374221cf1bf4` | Direct LightGBM |

Quantile P10/P90 models are **interval companions**, not selected point forecasts.

## Model validation evidence

| Phase | Documented result |
| --- | --- |
| 8 | FROZEN LightGBM benchmark (57/57 in repo validation history) |
| 9 | COMPLETE (146/146 PASS) |
| 10 | COMPLETE (experimental comparators; Phase 8 unchanged) |
| 11 | COMPLETE (140/140 PASS; READY WITH MONITORING) |
| 12 | COMPLETE (42/42 PASS; academic/reference packaging) |
| 13 | COMPLETE (42/42 PASS; security/operationalization; cloud not executed) |
| 14 | COMPLETE (19/19 PASS; local E2E simulation; Docker local image) |
| 15 | See `docs/phase15_metadata.json` after `python src/validate_phase15.py` |

## Inventory intelligence

Existing scorer labels on the extract: 733 CRITICAL / HIGH stockout, 887 replenishment-review flags (shelf at/below ROP), 0 SEVERE OVERSTOCK, 1 MODERATE OVERSTOCK. UCI keys do not join this extract → NOT AVAILABLE.

## API and security

FastAPI with optional `FORESIGHT_API_AUTH_ENABLED`, API-key header, in-process rate limits, audit log lines, `/health` and `/ready`. This is **not** an enterprise identity provider. TLS is not terminated in-repo.

## Monitoring

`python -m src.monitoring.run_monitoring` writes JSON under `outputs/monitoring/`. Snapshots are labelled as file snapshots. Automatic retraining is disabled.

## Dashboard

* `dashboard/app.py` — executive/risk/scenario UI; in-session trainer does not write `models/final`.
* `dashboard/forecast_analytics.py` — frozen forecast analytics.
* `dashboard/executive_intelligence.py` — Phase 15 BI over `outputs/bi/` with freshness captions.

## BI exports and Power BI preparation

See `docs/powerbi_data_model.md`. Exports are parquet. Power BI Desktop is not required for tests. No Power BI deployment was performed.

## Data-quality scorecard (forecast file)

From `outputs/monitoring/data_quality_report.json` (no invented composite score):

| Check | Result |
| --- | --- |
| n_rows | 957,949 |
| n_columns | 12 |
| missing required columns | [] |
| duplicate_rate_pct | 0.0 |
| n_duplicates | 0 |
| invalid_negative_counts | {} |
| leakage_columns_present | [] |
| date_gaps | null (not scored in this snapshot) |
| lower/upper missing_value_rate_pct | 76.351 (intervals are h=1 only) |
| schema_validity | PASS if required columns present |

Phase 3 cleaning metrics remain in `docs/data_quality_report.json`.

## Business value mapping

| Technical output | Business use |
| --- | --- |
| Demand forecast | Planning |
| Prediction interval | Uncertainty awareness |
| Demand growth | Capacity planning (where the documented rule is identified) |
| Stockout risk | Replenishment review |
| Overstock risk | Inventory review |
| Product ranking | Prioritization |
| Monitoring | Operational awareness |
| Recommendations | Decision support |

The system is **decision support**, not an autonomous decision-maker.

## Testing and validation commands

```bash
python src/validate_phase12.py
python src/validate_phase13.py
python -m pytest tests -q
python src/validate_phase14.py
python src/validate_phase15.py
```

## Final phase status matrix

| Phase | Status | Evidence |
| --- | --- | --- |
| Phase 1 | COMPLETE | README business questions; notebooks/01 |
| Phase 2 | COMPLETE | `src/inspect_datasets.py`; notebooks/01 executed |
| Phase 3 | COMPLETE | `src/data_cleaning.py`; `docs/data_quality_report.json` |
| Phase 4 | COMPLETE | `src/data_integration.py` CAM |
| Phase 5 | COMPLETE | `docs/eda_report.md`; `notebooks/04_eda.ipynb` |
| Phase 6 | COMPLETE | `src/feature_engineering.py`; frozen feature hash in Phase 11 metadata |
| Phase 7 | COMPLETE | `src/forecasting` baselines |
| Phase 8 | FROZEN | LightGBM benchmark; joblib + SHA-256 unchanged |
| Phase 9 | COMPLETE | `docs/phase9_analysis_report.md`; 146/146 PASS |
| Phase 10 | COMPLETE | `docs/phase10_analysis_report.md` |
| Phase 11 | COMPLETE | `docs/phase11_metadata.json`; 140/140 PASS |
| Phase 12 | COMPLETE | `docs/phase12_metadata.json`; 42/42 PASS |
| Phase 13 | COMPLETE | `docs/phase13_metadata.json`; 42/42 PASS |
| Phase 14 | COMPLETE | `docs/phase14_metadata.json`; 19/19 PASS |
| Phase 15 | COMPLETE | `docs/phase15_metadata.json` (12/12 PASS) |

Phase 5 was previously unchecked in the README even though `docs/eda_report.md` exists; the matrix follows repository evidence, not the stale checkbox.

## Known limitations

See `docs/phase15_known_limitations.md`. Cloud, TLS, identity provider, secrets manager, autoscaling, automated retraining, full-universe inventory, and Power BI publication are **not** claimed.
