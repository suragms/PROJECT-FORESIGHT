# Phase 22 — API Documentation

Base URL (local): `http://127.0.0.1:8000`

Start server:
```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Interactive docs: `http://127.0.0.1:8000/docs`

---

## Phase 20 Production API

Prefix: `/phase20`

### GET /phase20/model

**Purpose:** Return production model metadata.

**Input:** None

**Output:** Model ID, source dataset, horizon, validation metrics, limitations.

**Example:**
```bash
curl http://127.0.0.1:8000/phase20/model
```

**Errors:** 500 if model metadata cannot be loaded.

---

### POST /phase20/forecast

**Purpose:** Generate weekly SKU forecasts using the promoted Phase 20 model.

**Input (JSON):**
```json
{
  "source_dataset": "SYNTHETIC",
  "forecast_origin": "2025-12-29",
  "include_extended": false,
  "records": [
    {
      "product_key": "SYN_00001",
      "features": { "lag_1": 120.5, "...": "..." }
    }
  ]
}
```

**Validation:**
- `source_dataset` must be `SYNTHETIC` (UCI rejected)
- Features must match 45-feature contract
- `include_extended`: if true, may include h7–h8 (PARTIAL accuracy)

**Output:** Forecast list with horizon, demand, status labels.

**Errors:**
- `400` — invalid source, missing features, schema errors
- `422` — request validation failure

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/phase20/forecast \
  -H "Content-Type: application/json" \
  -d '{"source_dataset":"SYNTHETIC","records":[{"product_key":"SYN_00001","features":{}}]}'
```

---

### POST /phase20/risk/explain

**Purpose:** Explain inventory risk for a single SKU.

**Input (JSON):**
```json
{
  "sku_id": "00001",
  "forecast_weekly_demand": 150.0,
  "on_hand_units": 200,
  "on_order_units": 50,
  "lead_time_weeks": 2,
  "safety_stock": 30,
  "reorder_point": 100
}
```

**Output:** Stockout/overstock levels, recommended action, projected balance.

**Errors:** `400` for invalid inputs.

---

### GET /phase20/contract

**Purpose:** Return the 45-feature production contract.

**Input:** None

**Output:** Feature names, dtypes, leakage status, required flags.

---

## Phase 21 Monitoring API

Prefix: `/phase21`

### GET /phase21/health

**Purpose:** High-level monitoring health score and component statuses.

**Output:**
```json
{
  "health_score": "HEALTHY|WATCH|DEGRADED|CRITICAL",
  "components": { "data_quality": "PASS", "...": "..." },
  "timestamp": "..."
}
```

**Note:** Returns `NOT_RUN` if monitoring has not been executed.

---

### GET /phase21/monitoring/latest

**Purpose:** Full latest monitoring summary including drift, alerts, simulations.

**Errors:** `404` if `python src/run_phase21.py` has not been run.

---

### GET /phase21/alerts

**Purpose:** Structured alerts with severity, evidence, recommended actions.

---

### GET /phase21/integrity

**Purpose:** Model integrity baseline and current hash verification.

**Output:** Baseline SHA-256 records + current integrity status for frozen 12 models and Phase 20 production model.

---

## Authentication

Configurable via environment variables (`FORESIGHT_API_AUTH_ENABLED`, `FORESIGHT_API_API_KEY`). See `docs/phase22_deployment_guide.md`. Do not expose API keys in documentation or commits.

## Legacy API

Original Phase 11 endpoints (`/forecast`, `/health`) remain available and are separate from Phase 20 production contract.
