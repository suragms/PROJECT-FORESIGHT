# Phase 19 — Final Hardening Report

**Project:** Demand & Inventory Intelligence · **Phase:** 19  
**Date:** 2026-08-28 · **Title:** Synthetic Candidate Hardening & Final Production Readiness

---

## 1. Executive Summary

Phase 19 hardened the Synthetic LightGBM candidate with holiday calendar features, validated a 6-week supported forecast horizon, and re-validated the forecast-driven risk engine. Overall WAPE improved from **14.42%** (Phase 17) to **13.96%** (Phase 19) with no material regression. Holiday-period bias in folds 3-4 was reduced but not eliminated. Risk stress tests: **6/6 PASS**.

**Final Decision: READY FOR FINAL PRODUCTION PROMOTION REVIEW**

---

## 2. Phase 18 Limitations Addressed

| Limitation | Phase 19 Action | Result |
|------------|----------------|--------|
| Holiday bias spike (folds 3-4) | Investigated root cause; added calendar holiday features | WAPE improved; bias reduced but not eliminated |
| h7-h8 horizon degradation | Analyzed per-horizon; established 6-week boundary | h1-h6 PASS; h7-h8 PARTIAL |
| Forecast-driven risk | Re-validated with Phase 19 forecasts | PASS |
| No quantile companions | Not addressed in Phase 19 | Remains for promotion phase |

---

## 3. Holiday Diagnostics

Folds 3-4 validation periods (Nov–Dec 2025) show elevated demand and systematic over-forecasting. Evidence supports seasonal/holiday shopping period as contributing factor. See `docs/phase19_holiday_diagnostics.md`.

---

## 4. Holiday Mitigation Decision

Added calendar-derived features from `data/raw/calendar.csv`:
- `is_holiday_week`, `holiday_count`, `weeks_to_next_holiday`, `weeks_since_last_holiday`, `holiday_x_promo`, season one-hot encodings.

All features verified available at prediction time. No fabricated holiday data.

---

## 5. Horizon Diagnostics

| Horizon | Phase 17 WAPE | Phase 19 WAPE | Status |
|---------|--------------|--------------|--------|
| h1-h6 | 11.1–11.9% | 10.8–11.3% | PASS |
| h7 | 24.64% | 24.42% | PARTIAL |
| h8 | 27.64% | 27.24% | PARTIAL |

**Validated forecast horizon: 6 weeks**

---

## 6. Long-Horizon Strategy

Pre-defined hybrid rule: LightGBM for all horizons (seasonal-naive does not outperform at any horizon in historical validation). Production use limited to **6-week supported horizon**. h7-h8 reported as extended horizon with degraded accuracy.

---

## 7. Forecast Performance

| Metric | Value |
|--------|-------|
| Seasonal-Naive WAPE | 25.51% |
| Phase 17 WAPE | 14.42% |
| Phase 19 WAPE | **13.96%** |
| Supported horizon WAPE (h1-h6) | **11.03%** |
| Improvement vs Phase 17 | +0.46 pp |

---

## 8. Rolling-Origin Evidence

5 folds, all beat baseline. Fold stability: STRONG. See `docs/phase19_backtest_report.md`.

---

## 9. Bias

Folds 0-2: low bias (+1 to +3). Folds 3-4: elevated positive bias (+37 to +45), improved from Phase 17 (+40 to +47).

---

## 10. Leakage Audit

45 features audited. 0 FAIL. Holiday features from calendar only.

---

## 11. Reproducibility

Model scoring: REPRODUCIBLE (max prediction diff = 0.00).

---

## 12. Risk Validation

Forecast-driven risk using Phase 19 forecasts. Decision grid: PASS. Stress tests: 6/6 PASS. See `docs/phase19_risk_validation.md`.

---

## 13. Risk Stress Tests

All 6 scenarios pass: severe stockout, moderate stockout, healthy, moderate overstock, severe overstock, high volatility.

---

## 14. Rupee Impact

`sales_at_risk` traced to `sku_master.base_price`. `locked_capital` from `cost_price`. No fabricated assumptions.

---

## 15. Phase 17 vs Phase 19 Comparison

See `docs/phase19_candidate_comparison.md`.

---

## 16. Production Readiness Matrix

See `docs/phase19_production_readiness_matrix.md`.

---

## 17. Frozen Production Integrity

12/12 production model hashes unchanged. Phase 17 artifacts preserved.

---

## 18. Final Recommendation

**READY FOR FINAL PRODUCTION PROMOTION REVIEW**

The Synthetic candidate has been hardened with measurable improvement, validated 6-week horizon, and passing risk validation. Promotion requires a separate controlled phase addressing API adapters, quantile companions, and formal registry update.

---

## 19. Remaining Limitations

1. Holiday bias in Nov-Dec folds not fully eliminated
2. h7-h8 extended horizon with PARTIAL accuracy
3. API/dashboard schema adapters required for weekly grain
4. No quantile/hurdle companion models
5. UCI remains research candidate only
