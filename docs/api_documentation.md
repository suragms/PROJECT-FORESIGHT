# Forecast API

Base URL (local): `http://127.0.0.1:8000`

Authentication is not included in this academic/reference implementation.

Payload limit: 2,000,000 bytes (`FORESIGHT_API_MAX_PAYLOAD_BYTES`). Batch limit: 500 records (`FORESIGHT_API_MAX_BATCH`). Errors return JSON `{"detail": "..."}` without stack traces.

## GET /health

**Response**

```json
{"status": "ok", "version": "0.12.0", "timestamp": "2026-08-14T00:00:00+00:00"}
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
