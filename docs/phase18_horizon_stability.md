# Phase 18 — Horizon Stability Report

**Horizon evaluated:** 1 to 8 weeks ahead  
**Metric:** WAPE (%) and bias per horizon step

---

## SYNTHETIC Dataset

**Overall horizon status: HIGH_DEGRADATION (+16.49 pp from h=1 to h=8)**

| Horizon | Baseline WAPE | Candidate WAPE | Candidate Bias |
|---------|--------------|---------------|----------------|
| h=1 | 23.71% | 11.14% | +5.97 |
| h=2 | 23.10% | 11.18% | -2.48 |
| h=3 | 23.19% | 11.80% | +3.99 |
| h=4 | 22.67% | 11.43% | +1.11 |
| h=5 | 22.24% | 11.91% | +5.04 |
| h=6 | 22.20% | 11.75% | +3.28 |
| h=7 | 34.90% | 24.64% | +71.21 |
| h=8 | 37.65% | 27.64% | +82.73 |

**Interpretation:** Performance is stable from h=1 to h=6 (WAPE 11–12%). There is a sharp increase at h=7 and h=8 (WAPE 24–28%), accompanied by very high positive bias (+71 to +83). This pattern indicates the model struggles at the end of the 8-week window, likely during a high-demand holiday period (late November/December in the 4-year synthetic dataset).

**Classification: HIGH_DEGRADATION**

**Production implication:** If used in production, a 6-week rather than 8-week horizon would avoid the sharp degradation zone. This is within the Zidio specification (6–8 weeks). This limitation must be documented and the effective horizon set to 6 weeks for production use.

---

## UCI Dataset

**Overall horizon status: MODERATE_DEGRADATION (−14.67 pp from h=1 to h=8)**

| Horizon | Baseline WAPE | Candidate WAPE | Candidate Bias |
|---------|--------------|---------------|----------------|
| h=1 | 96.99% | 71.28% | +1.70 |
| h=2 | 94.21% | 68.61% | -3.24 |
| h=3 | 93.29% | 65.42% | -7.26 |
| h=4 | 91.66% | 61.42% | -12.96 |
| h=5 | 88.53% | 58.32% | -17.55 |
| h=6 | 86.07% | 56.91% | -22.02 |
| h=7 | 90.82% | 57.97% | -19.95 |
| h=8 | 92.47% | 56.61% | -16.91 |

**Interpretation:** UCI shows an unusual pattern — WAPE *decreases* at longer horizons. This is the opposite of the expected degradation pattern. It most likely reflects the sparse nature of UCI demand: many SKUs have zero demand at h=1 which the model misses, but at longer horizons the model captures more of the aggregated non-zero demand. The underlying bias is consistently negative and grows to −22 at h=6 before recovering slightly. This systematic under-forecasting is a material concern.

**Classification: MODERATE_DEGRADATION** (driven by bias, not by WAPE trend)
