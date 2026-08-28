# Phase 18 — Candidate Model Explainability

**Method:** LightGBM split-count feature importance  
**Note:** Feature importance reflects model *reliance* on each feature during tree construction. It does not imply that a feature *causes* demand.

---

## UCI Candidate — Top Features

| Rank | Feature | Importance | Feature Type | Available at Prediction Time | Leakage Status |
|------|---------|-----------|-------------|------------------------------|----------------|
| 1 | rolling_min_4 | 294 | rolling | Yes — shifted by 1 week | PASS |
| 2 | lag_1 | 262 | lag | Yes — 1 week prior | PASS |
| 3 | lag_13 | 243 | lag | Yes — 13 weeks prior | PASS |
| 4 | lag_26 | 243 | lag | Yes — 26 weeks prior | PASS |
| 5 | lag_4 | 228 | lag | Yes — 4 weeks prior | PASS |
| 6 | lag_2 | 216 | lag | Yes — 2 weeks prior | PASS |
| 7 | price_lag1 | 216 | price | Yes — lagged 1 week | PASS |
| 8 | lag_52 | 211 | lag | Yes — 52 weeks prior | PASS |

**Interpretation:** The UCI model relies heavily on recent demand history (`lag_1`, `rolling_min_4`) and medium-term lags (`lag_13`, `lag_26`) which capture quarterly and semi-annual patterns. The 52-week lag provides the same-week-last-year signal. Price from last week is also important, consistent with the UK retail context. All features are available at prediction time. No leakage detected.

---

## SYNTHETIC Candidate — Top Features

| Rank | Feature | Importance | Feature Type | Available at Prediction Time | Leakage Status |
|------|---------|-----------|-------------|------------------------------|----------------|
| 1 | rolling_mean_26 | 439 | rolling | Yes — shifted by 1 week | PASS |
| 2 | lag_2 | 428 | lag | Yes — 2 weeks prior | PASS |
| 3 | lag_1 | 415 | lag | Yes — 1 week prior | PASS |
| 4 | ewm_4 | 288 | ewm | Yes — shifted by 1 week | PASS |
| 5 | lag_4 | 271 | lag | Yes — 4 weeks prior | PASS |
| 6 | rolling_mean_4 | 257 | rolling | Yes — shifted by 1 week | PASS |
| 7 | price_lag1 | 210 | price | Yes — lagged 1 week | PASS |
| 8 | ewm_26 | 203 | ewm | Yes — shifted by 1 week | PASS |

**Interpretation:** The Synthetic model relies primarily on the 26-week rolling mean, which captures the medium-term level of demand — appropriate for a structured 4-year dataset with moderate seasonality. The combination of recent lags (`lag_1`, `lag_2`) and exponential smoothing (`ewm_4`) captures trend and momentum. Price history is a secondary signal. All features are available at prediction time. No leakage detected.

---

## Leakage Revalidation Summary

| Check | Result |
|-------|--------|
| Leakage audit entries | 36 |
| PASS | 36 |
| FAIL | 0 |
| REVIEW | 0 |
| Lag spot-check (manual comparison) | OK — all lags verified as `.shift(n)` |
| Rolling features verified | Shifted by 1 before window | PASS |
| EWM features verified | Shifted by 1 before ewm | PASS |
| Calendar features | Derived from `forecast_week` date only | PASS |
| Price features | `price_lag1` = price shifted 1 week | PASS |
| Promotion features | `promo_lag1` = flag shifted 1 week | PASS |

**Overall leakage verdict: PASS**
