# Phase 15 — Executive summary

**Demand & Inventory Intelligence (Project FORESIGHT)** is a validated demand-forecasting and inventory-risk **decision-support** system. Phase 15 turns the frozen technical stack into an executive BI view. It does not retrain models and does not make autonomous supply decisions.

## What the system does

* Forecasts store/SKU (SYNTHETIC) and invoice-day product (UCI) demand at horizons 1, 3, 7, 14, and 30 using **frozen** Phase 11 registry models.
* Scores inventory **risk** on a **1,000-row reference extract** (not the live warehouse).
* Serves forecasts through a local FastAPI with optional API-key auth and rate limits.
* Exposes Streamlit dashboards and Power BI–ready parquet extracts.

## What has been validated (baseline this phase)

| Gate | Historical recorded result | Current runtime validation result |
| --- | --- | --- |
| Phase 12 | 42/42 PASS | **41/42 PASS** — 1 failure: legacy `models\lightgbm_forecaster.joblib` hash mismatch (stale artifact; all 12 frozen `models/final/` hashes PASS) |
| Phase 13 | 42/42 PASS | **41/42 PASS** — same cascading failure |
| Phase 14 | 19/19 PASS | **17/19 PASS** — 2 failures: Phase 12/13 regression gates cascade from the above |
| Pytest (pre–Phase 15 tests) | 59/59 PASS | **87/88 PASS** — 1 failure: `test_phase8_hashes_match_phase11_snapshot` (same root cause); 19 Phase 16 tests added |
| Model hashes (frozen `models/final/`) | UNCHANGED | **UNCHANGED** — all 12 frozen model SHA-256 hashes verified |

Phase 15 validation counts are written by `python src/validate_phase15.py` and are **not** hardcoded here.

## Business problems addressed

Planning (forecast), uncertainty awareness (P10/P90 companions), replenishment **review** (stockout labels), inventory **review** (overstock labels), and SKU prioritization (revenue ranking). The system does **not** place orders.

## Key findings (measured)

* Held-out TEST accuracy snapshot: MAE **8.8075**, RMSE **37.7012**, WAPE **73.4244**, bias **−0.2285** on 957,949 rows with actuals.
* Operational h=1 models remain `uci_h1_phase8_lightgbm` and `synthetic_h1_hurdle_th050`.
* Inventory extract: **733** CRITICAL / HIGH stockout labels, **887** replenishment-review flags, **1** MODERATE OVERSTOCK, **0** SEVERE OVERSTOCK.
* Extract ranking: 10 TOP_REVENUE and 10 LOW DEMAND SKU-store rows. Low demand is **not** interpreted as a bad product.
* SYNTHETIC monthly seasonality of TEST actuals is **weak** (CV 0.0429). UCI TEST shows a stronger monthly CV (0.3534) on a short 2011 sample.
* Recommendations on the extract: 733 replenishment reviews, 266 uncertainty reviews, 1 inventory-exposure review. None are purchase orders.

## Forecasting (frozen)

* Grain: UCI product × invoice-day (`ONLINE`); SYNTHETIC store × SKU × day.
* Horizon: 1 (operational), plus 3/7/14/30 direct LightGBM.
* Uncertainty: quantile P10/P90 **interval companions** on h=1 only — not guaranteed coverage bands and not actuals.

## Inventory

Stockout and overstock labels come from the existing risk scorer on the 1,000-row extract. Mean days of supply on the extract is 3.54. Replenishment is a **review**, not an executed ROP workflow (0 flags).

## Recommendations

Decision-support only: Review replenishment, Review inventory exposure, Monitor demand growth, Review forecast uncertainty, or No exceptional intervention indicated. Each row carries evidence, reason, recommended review, and limitation.

## Limitations (must read)

Cloud, TLS, identity provider, secrets manager, autoscaling, and automated retraining were **not** deployed. Power BI was **prepared**, not published. The risk matrix is a **1000-row reference extract**. Monitoring files are snapshots, not live APM. See `docs/phase15_known_limitations.md`.
