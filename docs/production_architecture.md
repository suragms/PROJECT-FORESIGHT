# Production architecture (Phase 12)

This repository is an **academic/reference implementation** of a demand-forecasting stack. Phase 12 packages the Phase 11 final models for inference, API access, monitoring, and a read-only dashboard. It is **not** a claim that the system has been deployed to a production cloud.

## Layout

| Path | Role |
| --- | --- |
| `src/forecasting/` | Production inference package. `baselines.py` is the unchanged Phase 7/8 dashboard engine (moved from `src/forecasting.py`). |
| `src/final_forecasting.py` | Phase 11 inference core (hash check, schema, non-negative clip). Not modified in behavior. |
| `src/api/` | FastAPI app. Registered models only. |
| `src/monitoring/` | Data quality, drift, forecast distribution, accuracy when actuals exist. |
| `src/config.py` | Paths and thresholds via repo-relative paths and env vars. |
| `models/final/` | Phase 11 selected models + interval companions. |
| `dashboard/forecast_analytics.py` | Read-only Streamlit analytics. |
| `dashboard/app.py` | Existing executive inventory dashboard (unchanged). |
| `tests/` | API, inference, validation, regression, dashboard smoke tests. |

## Request flow

```
client → FastAPI validation → ForecastEngine
      → registry resolve (dataset, horizon)
      → SHA-256 check
      → Phase 11 FinalForecaster (fitted preprocessor)
      → non-negative demand
      → optional P10/P90 companion on h=1
      → schema + generated_at
```

Callers cannot pass a filesystem model path. Only `docs/final_model_registry.json` IDs are used.

## Final models (Phase 11)

- UCI h=1: `uci_h1_phase8_lightgbm`
- SYNTHETIC h=1: `synthetic_h1_hurdle_th050`
- Both datasets h=3,7,14,30: `*_h{h}_direct_lightgbm`
- Interval companions (h=1): `*_h1_quantile_p10p90`

Grain: `date + source_dataset + entity_id + product_key`. Target: `units_sold`.

## Authentication

Authentication is not included in this academic/reference implementation.
