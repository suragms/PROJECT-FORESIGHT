# Phase 13 — Security report

## Scope

Harden the Phase 12 FastAPI serving layer. Forecast models, features, and hashes were not modified.

## Authentication

| Setting | Role |
| --- | --- |
| `FORESIGHT_ENV` | `development` (default) or `production` |
| `FORESIGHT_API_AUTH_ENABLED` | Enable API-key checks in development |
| `FORESIGHT_API_API_KEY` | Secret from the environment; never committed |

Production **refuses to start** if auth is disabled or the key is missing (`src/production/config_validation.py`).

Development may bypass auth so existing local tests and `uvicorn` keep working.

Clients send `X-API-Key` or `Authorization: Bearer <key>`. Comparison uses `hmac.compare_digest`.

Public endpoints: `GET /health`, `GET /ready`.  
`GET /model`, `POST /forecast`, and `POST /forecast/batch` require authentication when auth is required.

This is **not** an enterprise identity provider.

## Rate limiting

Optional in-memory sliding window:

* `FORESIGHT_RATE_LIMIT_ENABLED`
* `FORESIGHT_RATE_LIMIT_REQUESTS` / `FORESIGHT_RATE_LIMIT_WINDOW_SECONDS`
* Stricter bucket for `POST /forecast` and `POST /forecast/batch`

Health and readiness skip the limiter. Exceeding the window returns HTTP **429** `{"detail":"Rate limit exceeded"}` with `Retry-After`.

Limits are per API process, not a shared cluster quota.

## Request validation

Existing Phase 11 `FinalForecaster.validate_input` is unchanged. The API layer additionally rejects:

* unsupported dataset / horizon
* empty or path-like entity/product identifiers
* client-supplied model filesystem fields (`model_path`, `.joblib`, `models/final`)
* non-finite numerics (`NaN` / `Inf` encodings)
* prohibited negatives on lag/price fields
* duplicate forecasting keys
* oversized batch (`FORESIGHT_API_MAX_BATCH`)
* oversized payload (`FORESIGHT_API_MAX_PAYLOAD_BYTES`)
* nested feature objects

Unhandled errors still return `{"detail":"Internal server error"}` without stack traces.

## Security headers

Applied to API responses:

* `X-Content-Type-Options: nosniff`
* `X-Frame-Options: DENY`
* `Referrer-Policy: no-referrer`
* `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'; base-uri 'none'`
* `Cache-Control: no-store`

HSTS is **not** set on the app. It belongs at a TLS terminator that does not exist in this repository.

## Audit logging

JSON audit events (`forecast_service.audit`):

* `application_startup` / `application_shutdown`
* `forecast_request` / `batch_forecast_request`
* `authentication_success` / `authentication_failure`
* `validation_failure`
* `rate_limit_rejection`
* `model_loaded` / `model_hash_verification`

Never logged: API keys, passwords, secrets, or full feature payloads.

## Docker

* non-root `appuser` (uid 10001)
* pinned `requirements.txt` install
* no `.env` / raw data in the image
* `HEALTHCHECK` against `/health`
* production image expects `FORESIGHT_API_API_KEY` at runtime

## Tests

`tests/test_security.py` covers unauthenticated rejection, valid/invalid keys, public health, 429, readiness failure modes, payload/batch limits, path traversal, and absence of tracebacks/secrets.

## Residual risk

* API keys in environment variables are weaker than a managed IdP + mTLS.
* In-memory rate limits reset on process restart and do not coordinate across replicas.
* `/docs` is unauthenticated in development; enable auth in production if exposing OpenAPI.
