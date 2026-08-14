# Deployment guide

This is a local/reference deployment. Cloud deployment has **not** been performed.

## Local workflow

```text
install dependencies
↓
validate Phase 11 registry (hashes)
↓
run tests
↓
start API
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

```bash
docker build -t foresight-api .
docker run --rm -p 8000:8000 foresight-api
```

The image copies `src/`, `models/final/`, and the model registry. It does **not** include raw retail extracts. Feature parquet for batch jobs must be mounted if needed.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `FORESIGHT_PROJECT_ROOT` | repo root | Override path root |
| `FORESIGHT_API_HOST` | 127.0.0.1 | Bind host (docs) |
| `FORESIGHT_API_PORT` | 8000 | Bind port (docs) |
| `FORESIGHT_API_MAX_BATCH` | 500 | Batch size cap |
| `FORESIGHT_API_MAX_PAYLOAD_BYTES` | 2000000 | Body size cap |
| `FORESIGHT_LOG_LEVEL` | INFO | Logging |
| `FORESIGHT_APP_VERSION` | 0.12.0 | /health version |
| `FORECAST_DASHBOARD_SKIP_MAIN` | unset | Set to `1` in unit tests |

## Cloud

Options (not executed here): container on a VM, Cloud Run / App Service with the Docker image, plus object storage for `models/final`. Add authentication, TLS, and secrets management before any real deployment.
