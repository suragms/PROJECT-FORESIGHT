# Forecast API

Base URL (local): `http://127.0.0.1:8000`

Version: `0.13.0` (Phase 13). Forecast math is unchanged from Phase 11/12.

Authentication is **configurable**. Development defaults to off. Production requires `FORESIGHT_API_AUTH_ENABLED=true` and `FORESIGHT_API_API_KEY`. Send `X-API-Key` or `Authorization: Bearer <key>`.

Public (no auth): `GET /health`, `GET /ready`. Protected in production: `/model`, `/forecast`, `/forecast/batch`.

Rate limiting is optional (`FORESIGHT_RATE_LIMIT_ENABLED`). Forecast POST routes use a stricter bucket than general API traffic. Exceeding the window returns HTTP 429 `{"detail":"Rate limit exceeded"}`.

Payload limit: 2,000,000 bytes (`FORESIGHT_API_MAX_PAYLOAD_BYTES`). Batch limit: 500 records (`FORESIGHT_API_MAX_BATCH`). Errors return JSON `{"detail": "..."}` without stack traces. Security headers: `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Content-Security-Policy`.

## GET /health

Lightweight liveness. Does not hash models.

**Response**

```json
{"status": "ok", "version": "0.13.0", "timestamp": "2026-08-16T00:00:00+00:00"}
```

## GET /ready

Verifies process init, registry load, SHA-256 of registered joblibs, and configuration. Returns HTTP 503 when not ready.

```json
{"status": "ready", "version": "0.13.0", "models_verified": true, "registry_verified": true, "config_valid": true}
```

## GET /model

Optional query: `dataset`, `horizon`.

**Response:** list of selected registry models with SHA-256 hashes, supported datasets `UCI` / `SYNTHETIC`, horizons `1,3,7,14,30`.

**Errors:** 400 if dataset/horizon is unsupported.

## POST /forecast

**Request**

```json
{
  "source_dataset": "UCI",
  "horizon": 1,
  "date": "2011-09-26",
  "entity_id": "ONLINE",
  "product_key": "<sku>",
  "features": {
    "units_sold_lag_1": 4.0,
    "rolling_mean_7": 3.2
  },
  "include_actual": false
}
```

`features` must include the Phase 11 required columns for that model (see the loaded joblib `numeric_features` / `categorical_features`). Direct models (`horizon` 3/7/14/30) also require `hcal_*` target-calendar fields.

**Response:** `{ "forecasts": [...], "metadata": {...}, "n": 1 }`

Each forecast row:

`forecast_date, source_dataset, entity_id, product_key, horizon, prediction, lower_bound, upper_bound, model_name, model_version, generated_at` and `actual` only when `include_actual` is true and a genuine actual exists.

**Validation errors (400):** missing columns, missing `units_sold_lag_1`, invalid dates, duplicate keys, negative prohibited fields, unsupported dataset/horizon.

**Schema errors (422):** missing required JSON fields.

## POST /forecast/batch

**Request**

```json
{
  "source_dataset": "UCI",
  "horizon": 1,
  "records": [ { "date": "...", "entity_id": "...", "product_key": "...", "features": {} } ]
}
```

**Errors:** 400 if `len(records) > 500`.

## Security notes

- No arbitrary model file paths.
- No parent-directory traversal in registry `model_file`.
- Hash mismatch refuses to load.
- Unhandled exceptions return `{"detail":"Internal server error"}`.
- API keys are read from the environment and are never logged.
- Production process start fails if auth is disabled or the API key is missing.
