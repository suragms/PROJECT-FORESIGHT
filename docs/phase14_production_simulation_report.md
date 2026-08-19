# Phase 14 — Production simulation report

This is a **local** simulation. No cloud VM was started.

## Emulated flow

```text
Production Client (FastAPI TestClient + Docker runtime)
        ↓
Authenticated API (X-API-Key)
        ↓
Forecast Service
        ↓
Frozen Model Registry + SHA-256
        ↓
Forecast Result
        ↓
Inventory Risk Engine (join or NOT AVAILABLE)
        ↓
Business Recommendation (non-executing)
        ↓
Monitoring files
        ↓
Audit capture
```

The simulation calls the existing FastAPI app. It does not duplicate the service.

## Health and readiness

| Call | Result |
| --- | --- |
| `GET /health` | HTTP 200 |
| `GET /ready` | HTTP 200 (`status=ready`, models and registry verified) |
| `GET /model` without key | HTTP 401 |
| `GET /model` with valid key | HTTP 200 |

## Single forecast (measured)

| Field | Value |
| --- | --- |
| dataset | UCI |
| entity | ONLINE |
| product | UCI_10135 |
| horizon | 1 |
| model | `uci_h1_phase8_lightgbm` |
| prediction | 10.749200294243758 |
| P10 | 0.9803840782755803 |
| P90 | 12.034108979147309 |
| latency | 0.1002 s |
| request id | `f961d78d-8b14-4d1f-b246-8e6d9ffccfb0` (`X-Request-ID`) |
| model hash used | `331909f0fe191c0b9cb0418884b25eb59012f479f61f8b3e2ad51b729273e90d` |

Stages: request → validation → registry model selection → prediction → response → audit `forecast_request`. No silent failure.

## Batch forecast (measured)

| n | latency_s | rows/s | success | failure | response_bytes |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.0833 | 120.05 | 10 | 0 | 4301 |
| 100 | 0.1295 | 772.13 | 100 | 0 | 39312 |
| 500 | 0.1424 | 3510.89 | 500 | 0 | 195312 |

Phase 12 freeze reference (not an SLO): single 0.0681 s, batch10 0.0607 s, 164.8 rows/s. Phase 14 numbers include production auth and TestClient overhead.

## Registry verification

Registry exists → selected UCI h=1 model exists → SHA-256 matches Phase 12/13 freeze → model loads → same hash is returned on the inference metadata.

## Docker (local)

Image tag: `foresight-phase14:local`.

Measured inside the container:

| Check | Result |
| --- | --- |
| `GET /health` | HTTP 200 |
| `GET /ready` | HTTP 200 |
| unauthenticated `/model` | HTTP 401 |
| authenticated `/model` | HTTP 200 |
| process uid | 10001 (`appuser`) |
| UCI h=1 hash | `331909f0…` |
| SYNTHETIC h=1 hash | `59a2b720…` |
| API key baked into image | no |

The image is not published to a registry and is not running in a cloud account.

## Failure recovery

| Fault | Result |
| --- | --- |
| Missing model file | `/ready` HTTP 503 |
| Missing registry | `/ready` HTTP 503 |
| Hash mismatch | `/ready` HTTP 503 |
| Invalid request | HTTP 4xx, no traceback |
| Internal predict error | HTTP 500 `Internal server error`, no traceback |
| Rate limit | HTTP 429 |

Monkeypatches are restored after each controlled failure.

## Audit

Captured events included startup, shutdown, readiness, auth success/failure, forecast, batch, validation failure, model load/hash, rate-limit, unhandled error. The simulation API key was not present in the captured JSON.

## Monitoring

`run_monitoring` still writes quality, accuracy, drift, and API counter files. `retraining` remains `disabled`.
