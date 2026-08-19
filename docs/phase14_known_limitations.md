# Phase 14 — Known limitations

## Implemented locally

* Forecast API with configurable API-key auth and in-process rate limits
* `/health` and `/ready`
* Frozen registry inference (no retraining)
* File-based monitoring and JSON audit capture
* Reference inventory scoring on a **1000-row** extract
* Streamlit dashboards over saved artifacts
* Local Docker image `foresight-phase14:local` (non-root uid 10001)

## Production-ready design, not provisioned

* Docker image (non-root, healthcheck)
* Environment-variable configuration (see `.env.example`)
* Readiness/liveness probes for an orchestrator that does not exist here

## Not deployed

Cloud deployment was **not executed** on Render, Azure, AWS, or GCP.

### Minimum requirements if someone deploys later

| Item | Guidance |
| --- | --- |
| CPU | 2 vCPU is a practical starting point for LightGBM inference |
| RAM | 4 GB minimum; 8 GB more comfortable with pandas + 12 joblibs |
| Storage | Image + `models/final` (do not copy raw retail extracts) |
| Port | `FORESIGHT_API_PORT` (default 8000) |
| Health | `GET /health` |
| Readiness | `GET /ready` |
| Models | `models/final` + `docs/final_model_registry.json` |
| Logging | stdout JSON audit lines |
| Secrets | inject `FORESIGHT_API_API_KEY` at runtime; never bake into the image |
| Env | `FORESIGHT_ENV=production`, `FORESIGHT_API_AUTH_ENABLED=true` |

TLS, secret managers, and autoscaling belong to the platform, not this repo.

## Reference logic (analytical, not operational)

* Stockout / overstock scores
* Reorder recommendations
* The 10 business questions

No supplier API, no ERP write-back, no live warehouse feed.

## Contract notes

* Unseen categorical `season` is **accepted** by the existing frequency encoder (HTTP 200). That is the frozen Phase 11 contract, not a Phase 14 model change. Changing it would alter serving behaviour for those inputs.
* UCI forecast keys do not join the synthetic inventory matrix → **NOT AVAILABLE**, not imputed.
* Rate limits are per process.
* API keys are not an enterprise identity provider.
* Automatic retraining is disabled.
* Registry `model_file` values use Windows separators in the frozen JSON. Serving now normalizes `\` → `/` so Linux containers resolve the same artifacts without rewriting the registry.

## Docker notes

* `docker build` can take a long time on a cold pip layer because `requirements.txt` includes Streamlit/XGBoost.
* Docker Hub TLS timeouts can prevent a **fresh** base-image pull. A local image already built from this Dockerfile can be reused (`FORESIGHT_DOCKER_REBUILD=1` forces a rebuild).
* The container user cannot create `/app/outputs` unless the image pre-creates that directory (Dockerfile now does).

## Future work

* Managed IdP / OIDC
* Shared API gateway rate limits
* Production APM
* Full-universe inventory scoring refresh (the on-disk matrix is a 1000-row extract)
* Human-approved retraining workflow (not automatic)
