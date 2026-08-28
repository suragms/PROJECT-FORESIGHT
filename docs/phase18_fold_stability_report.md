# Phase 18 — Fold Stability Report

**Validation method:** Rolling-origin cross-validation, 5 folds  
**Horizon per fold:** 8 weeks  
**Minimum training history required:** 52 weeks

---

## SYNTHETIC Dataset

| Fold | Train Origin | Val Start | Val End | Baseline WAPE | Candidate WAPE | Candidate Bias | Improvement | Candidate Wins |
|------|-------------|-----------|---------|--------------|---------------|----------------|-------------|----------------|
| 0 | 2025-10-14 | +1w | +8w | 23.35% | 11.46% | +3.10 | +11.88 pp | ✓ |
| 1 | 2025-10-21 | +1w | +8w | 22.54% | 11.63% | +4.31 | +10.91 pp | ✓ |
| 2 | 2025-10-28 | +1w | +8w | 22.42% | 11.24% | +7.50 | +11.18 pp | ✓ |
| 3 | 2025-11-04 | +1w | +8w | 29.70% | 18.96% | +40.21 | +10.75 pp | ✓ |
| 4 | 2025-11-11 | +1w | +8w | 30.99% | 20.41% | +47.23 | +10.58 pp | ✓ |

**Folds beating baseline: 5/5**  
**Stability classification: STRONG**

**Observation:** The improvement is consistent across all five folds, ranging from 10.58 to 11.88 percentage points. The baseline WAPE rises in folds 3-4 (holiday period), but the candidate also rises proportionally — indicating both are capturing the same seasonal difficulty. The candidate maintains a consistent relative advantage.

**Bias trend:** Bias is low and acceptable for folds 0-2 (+3 to +8). It spikes materially in folds 3-4 (+40 to +47), coinciding with the holiday-season period. This over-forecasting in the holiday window is a documented limitation requiring attention before production use.

---

## UCI Dataset

| Fold | Train Origin | Val Start | Val End | Baseline WAPE | Candidate WAPE | Candidate Bias | Improvement | Candidate Wins |
|------|-------------|-----------|---------|--------------|---------------|----------------|-------------|----------------|
| 0 | 2011-09-20 | +1w | +8w | 96.82% | 64.97% | -7.04 | +31.85 pp | ✓ |
| 1 | 2011-09-27 | +1w | +8w | 90.71% | 62.40% | -11.98 | +28.31 pp | ✓ |
| 2 | 2011-10-04 | +1w | +8w | 91.56% | 61.32% | -7.58 | +30.25 pp | ✓ |
| 3 | 2011-10-11 | +1w | +8w | 88.96% | 61.28% | -14.17 | +27.68 pp | ✓ |
| 4 | 2011-10-18 | +1w | +8w | 90.13% | 60.83% | -12.42 | +29.31 pp | ✓ |

**Folds beating baseline: 5/5**  
**Stability classification: STRONG**

**Observation:** The UCI candidate consistently beats the seasonal-naive baseline by 27–32 percentage points across all five folds. However, the absolute WAPE of 60–65% remains very high. This is not hidden: it reflects the genuine sparsity and long-tail behavior of 4,917 weekly SKU series over ~2 years.

**Bias trend:** UCI shows a persistent negative bias (under-forecast) that worsens from -7 to -14 across folds 0-4. This is a systematic pattern, not random noise, and constitutes a **HIGH severity** finding. The model under-predicts demand. In a real inventory context, systematic under-forecasting leads to stockout risk.
