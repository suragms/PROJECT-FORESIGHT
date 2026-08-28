# Phase 18 — Final Promotion Gate Report

**Project:** Demand & Inventory Intelligence · **Phase:** 18  
**Date:** 2026-08-19 · **Title:** Candidate Promotion Gate & Independent Validation

---

## Executive Decision

| Candidate | Decision |
|-----------|----------|
| **SYNTHETIC LightGBM** | **PROMOTE WITH LIMITATIONS** |
| **UCI LightGBM** | **KEEP AS RESEARCH CANDIDATE** |

Neither candidate replaces the frozen production stack automatically. The SYNTHETIC candidate is eligible for a formal promotion phase subject to the limitations documented below. The UCI candidate is not yet ready for production consideration.

---

## Candidate Results

| Metric | SYNTHETIC | UCI |
|--------|-----------|-----|
| Seasonal-naive WAPE | 25.51% | 91.63% |
| Candidate LightGBM WAPE | **14.42%** | **64.19%** |
| Improvement | 11.09 pp | 27.44 pp |
| Fold stability | STRONG (5/5) | STRONG (5/5) |
| Reproducibility | REPRODUCIBLE | REPRODUCIBLE |

**Important UCI note:** The relative improvement for UCI is large (27.44 pp), but the absolute WAPE of 64.19% remains high. This does not represent reliable forecast accuracy for an operational inventory system. For 1,041 out of 2,675 UCI SKUs, the candidate WAPE exceeds 100%. This is not hidden.

---

## Reproducibility

Both candidates score **REPRODUCIBLE** under independent scoring verification:
- Same model loaded fresh produces identical predictions (max diff = 0.00 across 500-sample spot check).
- Stored backtest WAPEs are consistent with independently computed fold WAPEs.

---

## Fold Stability

**SYNTHETIC:** Improvement consistent across all 5 folds (10.58–11.88 pp). STRONG.  
**UCI:** Improvement consistent across all 5 folds (27.68–31.85 pp). STRONG.  

However, fold stability alone is not sufficient for promotion. See Bias section.

---

## Bias

| Dataset | Overall Bias | Direction | Severity |
|---------|-------------|-----------|----------|
| SYNTHETIC | +19.78 | OVER_FORECAST | ACCEPTABLE (relative 4.1%) |
| UCI | -10.61 | UNDER_FORECAST | **HIGH** (relative 15.2%) |

**SYNTHETIC:** The aggregate relative bias is acceptable (4.1%). However, folds 3 and 4 show a bias spike (+40 to +47) in the holiday period. This is a documented limitation that must be addressed before production deployment.

**UCI:** Systematic under-forecast bias that grows at longer horizons (−7 to −22 at h=6). In an inventory context, systematic under-forecasting directly drives stockout risk. This is a material concern.

---

## Horizon Stability

**SYNTHETIC:** HIGH_DEGRADATION. WAPE stable at 11–12% from h=1 to h=6, then spikes to 25–28% at h=7 and h=8. Effective reliable horizon: **6 weeks**, not 8. Within Zidio spec range (6–8w).

**UCI:** MODERATE_DEGRADATION (by WAPE; the degradation pattern is unusual — WAPE improves at longer horizons due to sparse-series behavior). Bias worsens at longer horizons.

---

## Leakage Review

| Check | Result |
|-------|--------|
| Formal audit: 36 features | 36 PASS, 0 FAIL |
| Lag spot-check (3 SKUs per source) | PASS — `.shift(n)` confirmed |
| Rolling features | PASS — shifted by 1 before window |
| Calendar features | PASS — derived from week date only |
| Price/promo features | PASS — lagged 1 week |

**LEAKAGE STATUS: PASS**

---

## Feature Explainability

**SYNTHETIC top features:** `rolling_mean_26`, `lag_2`, `lag_1`, `ewm_4` — capturing medium-term level and recent momentum. All available at prediction time. All PASS leakage audit.

**UCI top features:** `rolling_min_4`, `lag_1`, `lag_13`, `lag_26` — capturing recent minimum demand and semi-annual patterns. All available at prediction time. All PASS leakage audit.

Model reliance on rolling statistics and lags is appropriate for weekly demand forecasting. No calendar or price feature dominates, which is healthy.

---

## Risk Validation

| Component | Status |
|-----------|--------|
| Forecast-driven demand | PASS |
| Required columns present | PASS (0 missing) |
| Decision grid consistency | PASS |
| WoS = on_hand / forecast_demand | PASS |
| Rupee impact non-negative | PASS |
| UCI risk | NOT AVAILABLE — documented |

Decision grid verified: all REORDER NOW rows have CRITICAL stockout; all HEALTHY rows have LOW stockout and OPTIMAL overstock.

---

## Rupee Impact

| Metric | SYNTHETIC | UCI |
|--------|-----------|-----|
| Sales at risk | Computed from `forecast × lead_time × base_price` | NOT AVAILABLE |
| Locked capital | ₹0.00 (no overstock detected) | NOT AVAILABLE |
| Assumptions | None — all from sku_master.csv | — |

---

## Zidio Acceptance Matrix

| Requirement | SYNTHETIC | UCI |
|-------------|-----------|-----|
| Reproducible pipeline | PASS | PASS |
| Data quality handling | PASS | PASS |
| Weekly SKU forecast | PASS | PASS |
| Seasonal-naive baseline | PASS | PASS |
| Rolling-origin CV | PASS | PASS |
| WAPE primary metric | PASS | PASS |
| No leakage | PASS | PASS |
| Stockout risk | PASS | NOT APPLICABLE |
| Overstock risk | PASS | NOT APPLICABLE |
| Recommended action | PASS | NOT APPLICABLE |
| Rupee impact | PARTIAL | NOT APPLICABLE |
| Dashboard compatibility | PARTIAL (adapter needed) | PARTIAL |
| Scoring service compatibility | PARTIAL (adapter needed) | PARTIAL |

---

## Frozen Production Compatibility

| Component | Assessment |
|-----------|-----------|
| Forecast output schema | COMPATIBLE WITH ADAPTER (weekly vs daily grain) |
| Streamlit dashboard | COMPATIBLE WITH ADAPTER (column mapping needed) |
| FastAPI scoring service | COMPATIBLE WITH ADAPTER (horizon and grain alignment) |
| Risk engine | COMPATIBLE — Phase 17 risk engine can replace Phase 10 logic |
| Monitoring | COMPATIBLE WITH ADAPTER |

---

## Candidate Hashes

Stored in `docs/phase18_candidate_hashes.json`. Not added to production registry.

---

## Promotion Decision

### UCI LightGBM

**KEEP AS RESEARCH CANDIDATE**

Reasons:
1. Absolute WAPE 64.19% — high; 1,041/2,675 SKUs have WAPE > 100%
2. Systematic HIGH severity under-forecast bias (−15.2% relative bias)
3. No native inventory data — cannot function as end-to-end stockout/overstock model
4. Effective use: demand-forecasting research only

### SYNTHETIC LightGBM

**PROMOTE WITH LIMITATIONS**

Evidence in favour:
1. WAPE 14.42% — beats baseline 25.51% by 11.09 pp
2. STRONG fold stability (5/5 folds)
3. REPRODUCIBLE (max prediction diff = 0.00)
4. No leakage (36/36 features PASS)
5. Forecast-driven risk engine with verified decision grid
6. Rupee impact traceable to source data

Limitations that must be addressed in promotion phase:
1. **Horizon degradation:** Effective horizon is 6 weeks, not 8; h=7,8 show WAPE spike +16 pp
2. **Holiday bias spike:** Folds 3–4 show bias of +40 to +47; over-forecasting in high-demand periods
3. **API/dashboard adapters:** Weekly grain requires schema adapters for existing production components
4. **No quantile/hurdle:** Frozen production includes quantile (P10/P90) and hurdle models; Phase 17 provides point forecast only

---

## Limitations

- The SYNTHETIC candidate improvement (11.09 pp) is meaningful but the absolute WAPE (14.42%) may still be insufficient for certain high-value SKU decisions without quantile forecasts.
- Phase 17 used only LightGBM. XGBoost, Random Forest, and classical time-series alternatives were not evaluated.
- UCI lacks inventory — the UCI candidate cannot support the full Zidio risk pipeline and must not be presented as doing so.
- Both datasets differ from the Kaggle 10M Synthetic dataset (confirmed in Phase 16).

---

## Required Next Phase

**Phase 19 — Controlled Promotion:** If the SYNTHETIC candidate proceeds:
1. Address holiday-window bias (feature engineering or post-processing correction)
2. Reduce effective horizon to 6 weeks and document
3. Build quantile companion models (P10/P90) for uncertainty quantification
4. Build API/dashboard schema adapters
5. Run full regression suite on candidate in production schema
6. Update `docs/final_model_registry.json` with candidate hashes under new model IDs
7. Formal sign-off before replacing any `models/final/` artifact
