# Phase 20 — Forecast Contract

## Forecast Grain
Weekly SKU-level

## Source Dataset
SYNTHETIC only (UCI remains RESEARCH CANDIDATE)

## Supported Horizon
**6 weeks** — production-validated

## Extended Horizon
Weeks 7–8 — PARTIAL accuracy — NOT DEFAULT

Extended forecasts must be labeled `EXTENDED_PARTIAL` in API output.
They must NOT be labeled `PRODUCTION_VALIDATED`.

## Model
- **Model ID:** `phase20_synthetic_lightgbm`
- **Parent Candidate:** `phase19_synthetic_lightgbm`
- **Overall WAPE:** 13.96%
- **Supported Horizon WAPE (h1-h6):** 11.03%

## Known Limitation
Holiday bias in Nov–Dec remains partially unresolved. Holiday-period forecasts should be interpreted with additional review. This does not invalidate the model for general production use.

## API Default
All production API and dashboard views default to **6-week** horizon with `forecast_status: PRODUCTION`.

## UCI Rule
UCI models are NOT promoted. API rejects UCI source for Phase 20 endpoints.
