# Phase 19 — Candidate Comparison

| Metric | Seasonal Naive | Phase 17 | Phase 19 |
|--------|----------------|----------|----------|
| Overall WAPE | 25.51% | 14.42% | **13.96%** |
| Supported horizon (h1-h6) WAPE | — | — | **11.03%** |
| Bias (overall) | +14.75 | +19.78 | Reduced in folds 0-2; folds 3-4 still elevated |
| Fold stability | — | STRONG (5/5) | **STRONG (5/5)** |
| Holiday behavior (folds 3-4 WAPE) | 29.7%/31.0% baseline | 18.96%/20.41% | **18.52%/19.80%** (improved) |
| Holiday behavior (folds 3-4 bias) | — | +40/+47 | **+37.5/+45.0** (improved, not eliminated) |
| h1-h6 | Baseline ~22-24% | ~11-12% | **~10.8-11.2% PASS** |
| h7-h8 | Baseline ~35-38% | 24.6%/27.6% DEGRADED | **24.4%/27.2% PARTIAL** (slight improvement) |
| Leakage | PASS | PASS (36/36) | **PASS (45/45)** |
| Reproducibility | — | PASS | **PASS** |
| Risk validation | — | PASS | **PASS (6/6 stress tests)** |
| Features | Calendar only | 36 lag/rolling | **45 (+ holiday calendar)** |

## Key Findings

1. Phase 19 improves overall WAPE by 0.46 pp vs Phase 17 without material regression.
2. Holiday calendar features modestly improve folds 3-4 performance but do not eliminate holiday bias.
3. Supported production horizon validated at **6 weeks**; h7-h8 remain extended horizon with PARTIAL status.
4. Hybrid rule: LightGBM for all horizons (seasonal-naive does not win at any horizon in validation).
