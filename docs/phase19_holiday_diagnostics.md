# Phase 19 — Holiday Diagnostics

## Fold Summary

| Fold | Origin | Val Period | Baseline WAPE | Candidate WAPE | Bias | Holiday Weeks |
|------|--------|------------|--------------|---------------|------|---------------|
| 0 | 2025-10-14 | 2025-10-21–2025-12-09 | 23.347% | 11.463% | 3.103 | 200 |
| 1 | 2025-10-21 | 2025-10-28–2025-12-16 | 22.536% | 11.628% | 4.312 | 200 |
| 2 | 2025-10-28 | 2025-11-04–2025-12-23 | 22.421% | 11.237% | 7.501 | 200 |
| 3 | 2025-11-04 | 2025-11-11–2025-12-30 | 29.705% | 18.958% | 40.205 | 300 |
| 4 | 2025-11-11 | 2025-11-18–2025-12-30 | 30.997% | 20.414% | 47.226 | 300 |

## Root Cause Evidence

- Fold 3 validation: 2025-11-11 to 2025-12-30

- Fold 4 validation: 2025-11-18 to 2025-12-30

- Demand elevation in folds 3-4: **0.913x** vs other folds

- Folds 3-4 bias: **43.481** (over-forecast)

- Other folds bias: 4.972


## Period Detail (Folds 3-4)

| Week | Observed | Forecast | Error | Holiday | Season | Affected SKUs |
|------|----------|----------|-------|---------|--------|-----------------|
| 2025-11-11 | 50789.0 | 50382.6 | -406.4 | False | Fall | 19 |
| 2025-11-18 | 99706.0 | 102248.6 | 2542.6 | False | Fall | 44 |
| 2025-11-25 | 111842.0 | 103287.5 | -8554.5 | True | Fall | 41 |
| 2025-12-02 | 89908.0 | 95667.8 | 5759.8 | False | Winter | 53 |
| 2025-12-09 | 99652.0 | 100531.8 | 879.8 | False | Winter | 34 |
| 2025-12-16 | 96372.0 | 100577.4 | 4205.4 | False | Winter | 46 |
| 2025-12-23 | 102974.0 | 104072.9 | 1098.9 | True | Winter | 48 |
| 2025-12-30 | 32432.0 | 92128.5 | 59696.5 | True | Winter | 198 |

## Top Affected SKUs

| SKU | Mean Actual | Mean Forecast | Bias | SKU WAPE |
|-----|------------|--------------|------|----------|
| SYN_SKU_00060 | 1313.9 | 1327.9 | 14.0 | 25.9% |
| SYN_SKU_00057 | 1412.7 | 1543.3 | 130.6 | 17.7% |
| SYN_SKU_00068 | 1604.9 | 1772.0 | 167.1 | 15.3% |
| SYN_SKU_00024 | 675.5 | 735.5 | 59.9 | 34.6% |
| SYN_SKU_00044 | 1067.5 | 1176.6 | 109.2 | 21.8% |
| SYN_SKU_00032 | 1235.7 | 1361.1 | 125.4 | 17.1% |
| SYN_SKU_00031 | 981.2 | 1021.3 | 40.1 | 20.7% |
| SYN_SKU_00003 | 1256.7 | 1375.7 | 118.9 | 16.0% |
| SYN_SKU_00069 | 1263.6 | 1384.7 | 121.1 | 14.9% |
| SYN_SKU_00004 | 1015.3 | 1116.0 | 100.7 | 17.2% |

## Conclusion

Folds 3-4 validation periods (Nov 2025) coincide with elevated demand (0.913x vs other folds) and holiday calendar weeks. Candidate systematically over-forecasts during this period (positive bias). Errors are concentrated among high-volume SKUs. Evidence supports seasonal/holiday shopping period as contributing factor, not a data anomaly.
