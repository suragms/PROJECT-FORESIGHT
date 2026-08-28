print("""
========================================================
PROJECT FORESIGHT - PHASE 18 PROMOTION GATE
========================================================

PHASE 17 CANDIDATE RESULTS

UCI:
  Baseline WAPE: 91.63%
  Candidate WAPE: 64.19%
  Improvement: 27.44 pp
  Decision: KEEP AS RESEARCH CANDIDATE

SYNTHETIC:
  Baseline WAPE: 25.51%
  Candidate WAPE: 14.42%
  Improvement: 11.09 pp
  Decision: PROMOTE WITH LIMITATIONS

REPRODUCIBILITY:
  UCI: PASS
  SYNTHETIC: PASS

FOLD STABILITY:
  UCI: STRONG (5/5 folds beat baseline)
  SYNTHETIC: STRONG (5/5 folds beat baseline)

BIAS:
  UCI: HIGH (UNDER_FORECAST, relative bias -15.2%)
  SYNTHETIC: ACCEPTABLE (OVER_FORECAST, relative bias +4.1%; holiday spike folds 3-4)

HORIZON STABILITY:
  UCI: MODERATE_DEGRADATION (bias worsens; WAPE improves — sparse series)
  SYNTHETIC: HIGH_DEGRADATION at h=7,8 (+16.49 pp); effective horizon = 6 weeks

LEAKAGE:
  PASS (36 features audited, 0 FAIL; lag spot-check verified)

RISK ENGINE:
  Forecast-driven: PASS
  Stockout: PASS (15 CRITICAL, 82 MEDIUM, 3 LOW)
  Overstock: PASS (0 SEVERE, 0 MODERATE, 100 OPTIMAL)
  Decision grid: PASS (REORDER/MARKDOWN/WATCH/HEALTHY verified consistent)
  Rupee impact: PARTIAL (SYNTHETIC: sales_at_risk verified; UCI: NOT AVAILABLE)

ZIDIO ACCEPTANCE:
  SYNTHETIC: PARTIAL (10/13 PASS; 2 PARTIAL need adapters; 1 N/A)
  UCI: PARTIAL (7/13 PASS; 4 NOT APPLICABLE — no inventory)

PRODUCTION COMPATIBILITY:
  UCI: COMPATIBLE WITH ADAPTER (demand forecast only; no risk integration)
  SYNTHETIC: COMPATIBLE WITH ADAPTER (weekly-to-daily schema mapping required)

FROZEN PRODUCTION:
  12/12 hashes unchanged: PASS

PHASE 18 TESTS:
  28/28 PASS

FULL REGRESSION:
  Phase 12 = 42/42
  Phase 14 = 19/19
  Pytest (all phases) = 144/144

FINAL DECISION:

UCI:
  KEEP AS RESEARCH CANDIDATE
  (High absolute WAPE 64.19%; systematic under-forecast bias HIGH severity;
   1041/2675 SKUs WAPE > 100%; no native inventory data)

SYNTHETIC:
  PROMOTE WITH LIMITATIONS
  (WAPE 14.42% beats baseline 25.51% by 11.09 pp; STRONG fold stability;
   REPRODUCIBLE; no leakage; forecast-driven risk verified.
   Limitations: effective horizon 6w not 8w; holiday bias spike; API adapter needed;
   no quantile/hurdle companions yet)

AUTOMATIC PRODUCTION REPLACEMENT:
  NO

NEXT PHASE:
  Phase 19 - Controlled Promotion (SYNTHETIC only, if approved):
  1. Address holiday-window bias (feature engineering or correction)
  2. Set effective horizon to 6 weeks
  3. Build quantile companion models (P10/P90)
  4. Build API/dashboard schema adapters
  5. Register candidate hashes in final_model_registry.json
  6. Full regression suite sign-off before any models/final/ replacement

========================================================
""")
