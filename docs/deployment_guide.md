# Deployment guide

This is a local/reference deployment. Cloud deployment has **not** been performed.

## Local workflow

```text
install dependencies
↓
validate Phase 11 registry (hashes)
↓
configuration validation
↓
run tests
↓
start API (auth/rate-limit as configured)
↓
GET /ready
↓
start dashboard
↓
generate a batch forecast
↓
run monitoring
```

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Do not upgrade LightGBM / scikit-learn blindly; Phase 11 joblibs were trained with the pinned versions in `requirements.txt`.

### 2. Validate models

```bash
python src/validate_final_forecasting.py
python src/validate_phase12.py
```

### 3. Tests

```bash
python -m pytest tests -q
```

### 4. API

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Health: `GET http://127.0.0.1:8000/health`

### 5. Dashboard

```bash
streamlit run dashboard/forecast_analytics.py
```

Read-only. Does not retrain. The executive inventory app remains `streamlit run dashboard/app.py`.

### 6. Batch forecast

```bash
python -m src.forecasting.batch_forecast --help
python -m src.forecasting.batch_forecast --input data/samples/uci_h1_sample.parquet --output outputs/forecasts/uci_h1.parquet --dataset UCI --horizon 1
```

### 7. Monitoring

```bash
python -m src.monitoring.run_monitoring
```

Writes `outputs/monitoring/*.json`.

## Docker (API only)

The image runs as a non-root user, copies `src/`, `models/final/`, and the model registry, and does **not** include raw retail extracts or secrets. Production image defaults require `FORESIGHT_API_API_KEY` at runtime.

```bash
docker build -t foresight-api .
docker run --rm -p 8000:8000 ^
  -e FORESIGHT_API_API_KEY=replace-me ^
  foresight-api
```

Health check probes `GET /health`. Feature parquet for batch jobs must be mounted if needed.

## Environment variables

Copy `.env.example` to a local `.env` (never commit secrets).

| Variable | Default | Purpose |
| --- | --- | --- |
| `FORESIGHT_ENV` | development | `development` or `production` |
| `FORESIGHT_PROJECT_ROOT` | repo root | Override path root |
| `FORESIGHT_API_HOST` | 127.0.0.1 | Bind host |
| `FORESIGHT_API_PORT` | 8000 | Bind port |
| `FORESIGHT_API_AUTH_ENABLED` | false | Require API key |
| `FORESIGHT_API_API_KEY` | empty | Secret; not committed |
| `FORESIGHT_RATE_LIMIT_ENABLED` | false | Enable in-memory rate limits |
| `FORESIGHT_RATE_LIMIT_REQUESTS` | 60 | General window quota |
| `FORESIGHT_RATE_LIMIT_WINDOW_SECONDS` | 60 | Window length |
| `FORESIGHT_RATE_LIMIT_FORECAST_REQUESTS` | 20 | Stricter POST /forecast quota |
| `FORESIGHT_API_MAX_BATCH` | 500 | Batch size cap |
| `FORESIGHT_API_MAX_PAYLOAD_BYTES` | 2000000 | Body size cap |
| `FORESIGHT_LOG_LEVEL` | INFO | Logging |
| `FORESIGHT_APP_VERSION` | 0.13.0 | /health version |
| `FORECAST_DASHBOARD_SKIP_MAIN` | unset | Set to `1` in unit tests |

## Cloud

Cloud deployment has **not** been executed in this repository. A practical reference architecture:

```text
User
 ↓
HTTPS / TLS (load balancer / API gateway — not configured here)
 ↓
Cloud Load Balancer
 ↓
FastAPI container (this Docker image)
 ↓
Model registry + models/final (image or mounted volume)
 ↓
Monitoring reports (outputs/monitoring; not a managed APM)
```

| Concern | Where it would live | Status in this repo |
| --- | --- | --- |
| Secrets | Cloud secret manager → env | Documented only; use `FORESIGHT_API_API_KEY` |
| Models | Container image or object storage mount | Bundled under `models/final/` |
| Logs | stdout JSON audit lines → log drain | Local stdout |
| Monitoring | Cron/job for `run_monitoring` | Local files |
| TLS | Load balancer termination | Not executed |
| Autoscaling | Container service CPU/RPS | Not executed |

Do not treat this table as a completed deployment.
