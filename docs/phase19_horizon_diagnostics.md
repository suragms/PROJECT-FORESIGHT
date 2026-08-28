# Phase 19 — Horizon Diagnostics

**Supported production horizon:** 6 weeks

**Hybrid rule:** horizon_model_selection_from_historical_validation


| Horizon | Baseline WAPE | Phase 17 WAPE | Phase 19 WAPE | Hybrid WAPE | Phase 19 Bias | Status |
|---------|--------------|--------------|--------------|------------|--------------|--------|
| h=1 | 23.712% | 11.143% | 10.925% | 10.925% | 2.879 | PASS |
| h=2 | 23.101% | 11.175% | 10.805% | 10.805% | -2.685 | PASS |
| h=3 | 23.192% | 11.804% | 11.27% | 11.27% | 0.862 | PASS |
| h=4 | 22.67% | 11.429% | 10.843% | 10.843% | -2.82 | PASS |
| h=5 | 22.244% | 11.908% | 11.191% | 11.191% | 0.493 | PASS |
| h=6 | 22.204% | 11.747% | 11.16% | 11.16% | 2.697 | PASS |
| h=7 | 34.901% | 24.643% | 24.418% | 24.418% | 67.762 | PARTIAL |
| h=8 | 37.648% | 27.637% | 27.236% | 27.236% | 79.311 | PARTIAL |

## Degradation Analysis

- h1-h6: Stable performance within supported horizon

- h7-h8: Extended horizon with documented degradation (PARTIAL status)

- Seasonal-naive does NOT outperform LightGBM at h7-h8 in validation

- **Validated forecast horizon for production: 6 weeks**
