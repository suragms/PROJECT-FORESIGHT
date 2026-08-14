# Phase 9 — Advanced Forecast Validation, Stability, Residual & Horizon Analysis

**Status:** COMPLETE  
**Validation:** 146/146 PASS

## 1. Executive Summary

**Conclusion (Option B — Moderately Stable):** LightGBM is promising but requires targeted improvements before productionization.

Walk-forward stability labels:

- **UCI:** Stable (mean WAPE=85.3126, CV=0.1354, max/min=1.3691)

- **SYNTHETIC:** Stable (mean WAPE=39.394, CV=0.0132, max/min=1.037)

Phase 8 TEST LightGBM remains the frozen production candidate. Phase 9 did not retrain or replace those artifacts; walk-forward retrains *copies* of the same configuration on earlier windows only.

## 2. Methodology

### Walk-forward

Expanding-window validation: each fold trains LightGBM (Phase 8 hyperparameters, `random_state=42`) on all observations with `date <= train_end`, evaluates on `val_start..val_end`. Training always precedes validation; sources are never mixed. Grain: `date + source_dataset + entity_id + product_key`.

### Residual analysis

`residual = actual - prediction` on frozen Phase 8 TEST predictions. `bias = mean(prediction - actual)` matches Phase 8.

### Horizon analysis

Phase 8 is a **1-observation-ahead** model. Horizons [1, 3, 7, 14, 30] are evaluated with **iterated recursive** forecasts from the frozen LightGBM (same approach as `generate_multi_step_forecast` in the legacy engine). Lag/rolling/trend features are updated from the prediction buffer. Price/promo/inventory are held at origin values (no future leakage). Recursive h=1 is therefore **not identical** to Phase 8 TEST: Phase 8 uses target-row operational fields (especially SYNTHETIC inventory). UCI has no inventory features, so recursive h=1 WAPE should closely match Phase 8 TEST. Horizon unit = observation step (calendar day for SYNTHETIC; next observed date for gappy UCI). UCI recursive evaluation uses the 400 longest eligible test series (deterministic compute cap).

### Zero-demand

Split TEST rows into `actual == 0` vs `actual != 0`. WAPE is undefined/zero-safe when `sum(|actual|)=0`. sMAPE treats (0,0) as 0 and remains large when the model predicts positive demand against true zeros.

### Store/entity

Per-entity MAE/RMSE/WAPE/bias recomputed from Phase 8 TEST predictions.

### Stability rule

{'stable_cv_max': 0.15, 'stable_range_ratio_max': 1.5, 'moderate_cv_max': 0.35, 'moderate_range_ratio_max': 2.0}

## 3. Walk-Forward Results

```
source_dataset  fold  train_end  val_start    val_end  train_rows  val_rows     MAE    RMSE    sMAPE     WAPE    bias  overprediction_pct  underprediction_pct
           UCI     1 2010-07-13 2010-07-14 2010-12-31      142504    131081 19.0763 91.5864  84.2453  83.8348  0.3652               71.41                28.59
           UCI     2 2010-12-31 2011-01-01 2011-04-30      273585     74220 18.3322 92.7428  86.8999 105.3143  5.5927               76.01                23.99
           UCI     3 2011-04-30 2011-05-01 2011-07-13      347805     49142 15.6943 52.5210  84.1959  82.4168  1.2356               74.13                25.87
           UCI     4 2011-07-13 2011-07-14 2011-09-25      396947     53020 16.5035 55.8952  80.2343  76.9218 -1.0227               70.65                29.35
           UCI     5 2011-09-25 2011-09-26 2011-12-09      449967     79545 17.0402 70.1618  82.7665  78.0755 -0.1727               72.01                27.99
     SYNTHETIC     1 2023-12-31 2024-01-01 2024-06-30      729000    182000  2.9280  5.4709 119.2801  39.5758 -0.0015               68.89                22.02
     SYNTHETIC     2 2024-06-30 2024-07-01 2024-12-31      911000    184000  2.8840  5.3304 115.5737  39.3596  0.0951               67.58                21.51
     SYNTHETIC     3 2024-12-31 2025-01-01 2025-03-13     1095000     72000  2.6918  4.6846 108.4006  39.2559 -0.0034               65.01                25.05
     SYNTHETIC     4 2025-03-13 2025-03-14 2025-08-06     1167000    146000  3.1115  6.0244 126.2837  40.1044 -0.0288               71.02                19.95
     SYNTHETIC     5 2025-08-06 2025-08-07 2025-12-31     1313000    147000  2.7998  5.1142 114.4868  38.6743  0.0957               67.69                22.07
```


### UCI fold summary

- Stability: **Stable** — CV_WAPE=0.135 < 0.15 and max/min=1.369 < 1.5

- WAPE: mean=85.3126, median=82.4168, std=11.5479, min=76.9218, max=105.3143

- MAE: mean=17.3293, median=17.0402, std=1.369, min=15.6943, max=19.0763

- RMSE: mean=72.5814, median=70.1618, std=19.0681, min=52.521, max=92.7428

- sMAPE: mean=83.6684, median=84.1959, std=2.4321, min=80.2343, max=86.8999

- bias: mean=1.1996, median=0.3652, std=2.5893, min=-1.0227, max=5.5927

### SYNTHETIC fold summary

- Stability: **Stable** — CV_WAPE=0.013 < 0.15 and max/min=1.037 < 1.5

- WAPE: mean=39.394, median=39.3596, std=0.5186, min=38.6743, max=40.1044

- MAE: mean=2.883, median=2.884, std=0.1563, min=2.6918, max=3.1115

- RMSE: mean=5.3249, median=5.3304, std=0.4912, min=4.6846, max=6.0244

- sMAPE: mean=116.805, median=115.5737, std=6.5859, min=108.4006, max=126.2837

- bias: mean=0.0314, median=-0.0015, std=0.0594, min=-0.0288, max=0.0957

## 4. Residual Analysis

```
source_dataset      n     MAE    RMSE    sMAPE    WAPE     MAPE   bias  mean_residual  median_residual  residual_std  overprediction_pct  underprediction_pct
     SYNTHETIC 147000  2.8156  5.1469 113.6813 38.8923  30.4648 0.0873        -0.0873          -0.5393        5.1461               67.24                22.12
           UCI  79545 17.3447 70.8952  82.8734 79.4710 283.0189 0.0335        -0.0335          -3.8346       70.8956               71.72                28.28
```

Demand regimes (zero = actual 0; low/medium/high = tertiles of positive demand):

```
source_dataset demand_regime     n  n_share_pct  mean_actual  mean_prediction     MAE     RMSE    sMAPE     WAPE     bias  overprediction_pct  underprediction_pct  q33_positive  q66_positive
     SYNTHETIC          high 17723        12.06      31.8255          25.2808  8.2840  11.6919  33.6972  26.0294  -6.5447               23.28                76.72          13.0          22.0
     SYNTHETIC           low 21135        14.38       8.5730           9.6583  2.8395   3.8457  33.9435  33.1215   1.0853               60.94                39.06          13.0          22.0
     SYNTHETIC        medium 18072        12.29      17.6500          16.3109  4.2590   5.7185  28.2278  24.1302  -1.3391               40.97                59.03          13.0          22.0
     SYNTHETIC          zero 90070        61.27       0.0000           1.4444  1.4444   2.5107 165.2759   0.0000   1.4444               82.64                 0.00          13.0          22.0
           UCI          high 25580        32.16      57.5975          39.5428 34.5175 122.7346  55.7353  59.9288 -18.0547               35.37                64.63           3.0          14.0
           UCI           low 27436        34.49       1.6710          10.5330  8.8832  14.0414 128.6864 531.6161   8.8620               98.99                 1.01           3.0          14.0
           UCI        medium 26529        33.35       8.1759          16.5203  9.5371  18.4822  61.6615 116.6489   8.3444               78.55                21.45           3.0          14.0
```

Residual vs actual / prediction diagnostics (heteroscedasticity):

```
source_dataset      n  corr_abs_error_actual  corr_abs_error_prediction  corr_residual_actual  corr_residual_prediction  low_pred_mean_residual  high_pred_mean_residual  low_actual_mae  high_actual_mae
     SYNTHETIC 147000                 0.5587                     0.3599                0.5334                    0.0954                 -0.0718                   0.6400          1.8231           6.8783
           UCI  79545                 0.9267                     0.3895                0.9098                   -0.0420                 -0.5572                   1.6326          8.7647          49.1843
```


### SYNTHETIC

- Mean residual (actual-pred)=-0.0873, bias (pred-actual)=0.0873

- Over-prediction 67.24% / under-prediction 22.12%

- |error| vs actual correlation=0.5587; |error| vs prediction correlation=0.3599. Low-actual MAE=1.8231, high-actual MAE=6.8783.

### UCI

- Mean residual (actual-pred)=-0.0335, bias (pred-actual)=0.0335

- Over-prediction 71.72% / under-prediction 28.28%

- |error| vs actual correlation=0.9267; |error| vs prediction correlation=0.3895. Low-actual MAE=8.7647, high-actual MAE=49.1843.

## 5. Forecast Horizon Results

```
source_dataset  horizon     n     MAE     RMSE    sMAPE     WAPE    bias
     SYNTHETIC        1 21000  4.1399   7.1323 127.4303  64.3163  0.1631
     SYNTHETIC        3 21000  8.1481  13.1673 145.1359  82.8176 -0.2312
     SYNTHETIC        7 21000  7.4099  11.2998 151.6665 112.0417  0.4548
     SYNTHETIC       14 20000  7.3424  11.2173 153.7303 111.5234  0.2880
     SYNTHETIC       30 17000  6.5509  10.3880 155.4533 100.9901 -0.7547
           UCI        1  4839 28.5693  77.8076  76.9311  79.4761 -2.4823
           UCI        3  4759 29.6548  66.6666  76.6289  79.5917 -1.6682
           UCI        7  4439 33.9680 111.0133  80.6896  89.7171 -0.3643
           UCI       14  3875 37.2220 116.8579  81.3363  96.4946  0.7922
           UCI       30  2595 38.8311  91.6766  82.7562 100.2543  2.0394
```


- **UCI:** WAPE from h=1 (79.4761) to h=30 (100.2543); monotone non-decreasing=True; ratio h30/h1=1.2614396025975103

- **SYNTHETIC:** WAPE from h=1 (64.3163) to h=30 (100.9901); monotone non-decreasing=False; ratio h30/h1=1.5702100400676033

## 6. Zero-Demand Results

```
source_dataset  n_total  n_zero  n_nonzero  zero_share_pct  zero_actual_prediction_mae  zero_actual_mean_prediction  zero_actual_median_prediction  zero_actual_positive_prediction_rate  zero_WAPE  zero_sMAPE  nonzero_MAE  nonzero_RMSE  nonzero_WAPE  nonzero_sMAPE  nonzero_mean_actual  nonzero_mean_prediction  counts_reconcile
     SYNTHETIC   147000   90070      56930           61.27                      1.4444                       1.4444                         0.7548                                 82.64        0.0    165.2759       4.9850        7.6438       26.6677        32.0524              18.6932                  16.6336                 1
           UCI    79545       0      79545            0.00                         NaN                          NaN                            NaN                                   NaN        NaN         NaN      17.3447       70.8952       79.4710        82.8734              21.8252                  21.8588                 1
```


sMAPE is unstable when actuals are zero because the denominator is `|y|+|ŷ|`. If ŷ>0 and y=0, each term equals 2, so sMAPE can approach 200% even when absolute errors are small. WAPE for a pure-zero subset has denominator 0 and is reported as 0.0 by the Phase 7/8 zero-safe rule (not a claim of perfect accuracy). Prefer MAE on the zero-demand slice.

## 7. Store/Entity Stability

```
entity_id     n    MAE   RMSE    WAPE   bias
STORE_005 14700 2.7886 5.0175 37.8151 0.0782
STORE_009 14700 2.7362 4.9127 38.0589 0.1256
STORE_001 14700 2.7713 5.1224 38.1943 0.0959
STORE_006 14700 2.8593 5.2000 38.4936 0.0877
STORE_002 14700 2.7693 5.0403 38.8348 0.1258
STORE_008 14700 2.9047 5.3696 38.8528 0.0397
STORE_007 14700 2.8874 5.3858 39.1453 0.0206
STORE_004 14700 2.9101 5.3979 39.2797 0.0566
STORE_010 14700 2.7314 4.8924 40.1581 0.1377
STORE_003 14700 2.7978 5.0986 40.2323 0.1055
```

Best: **STORE_005** WAPE=37.8151, MAE=2.7886. Worst: **STORE_003** WAPE=40.2323, MAE=2.7978.

Phase 8 reported STORE_005 WAPE≈37.82 and STORE_003 WAPE≈40.23. Phase 9 recomputes from the same TEST predictions; small differences can arise from rounding. Values above are the recalculated figures.

## 8. Metric Interpretation

- **MAE:** scale of typical absolute error in units; comparable within a source, not across UCI vs SYNTHETIC volume scales.
- **RMSE:** penalizes large misses (UCI bulk orders inflate RMSE).
- **WAPE:** volume-weighted; robust to zeros in the mixed set because the denominator is total actual volume. Uninformative on an all-zero slice.
- **sMAPE:** bounded but inflates on sparse/zero demand when predictions are positive. High SYNTHETIC sMAPE does not by itself mean the model is unusable.

## 9. Phase 9 Findings

### Confirmed finding

- Frozen Phase 8 TEST WAPE remains UCI 79.471, SYNTHETIC 38.8923.

- Walk-forward labels: UCI=Stable, SYNTHETIC=Stable.

- UCI fold 2 (2011-01-01..2011-04-30) WAPE=105.3143 is the worst UCI window (post-holiday wholesale gap).

- SYNTHETIC TEST zero-demand share=61.27%; P(pred>0|actual=0)=82.64%.

- Recalculated SYNTHETIC store range: STORE_005 WAPE=37.8151 through STORE_003 WAPE=40.2323.

### Possible explanation

- SYNTHETIC intermittency (~60% zero-demand TEST days) drives high sMAPE and positive predictions on zero days.
- Recursive error accumulation can increase WAPE with horizon even if 1-step WAPE is acceptable.
- UCI low-demand regime over-prediction (mean actual << mean prediction) is consistent with a model pulled toward the mean of a heavy-tailed wholesale series.
- UCI fold-2 WAPE spike is consistent with a thinner post-holiday invoice calendar, not necessarily a broken model class.
- SYNTHETIC recursive h=1 WAPE is worse than Phase 8 TEST because origin-held inventory/promo features remove same-day operational information the 1-step model uses.

### Limitation

- Recursive multi-step is not a direct h-step model; it is the only leakage-safe multi-horizon method supported by the Phase 8 1-step architecture.
- UCI horizon analysis uses the 400 longest series (documented cap).
- SYNTHETIC recursive WAPE is not monotone in horizon (origins near the TEST end drop out).
- No prediction intervals in this phase.
- Walk-forward retrains LightGBM on each fold (same config) but does not tune hyperparameters.

## 10. Phase 10 Recommendations

Do **not** implement these here. Evidence-based next steps:

1. Intermittent-demand / hurdle / Croston-style methods for SYNTHETIC zeros.

2. Quantile regression or LightGBM quantile loss for prediction intervals.

3. Direct multi-horizon models (separate heads for h=7/14/30) vs recursive.

4. Hyperparameter optimization only after residual/zero-demand fixes.

5. Hierarchical store-SKU reconciliation for SYNTHETIC.


## Overall decision

**Option B — Moderately Stable**

LightGBM is promising but requires targeted improvements before productionization.

## Charts

- `outputs\figures\forecasting\phase9\fold_stability.png`

- `outputs\figures\forecasting\phase9\horizon_performance_synthetic.png`

- `outputs\figures\forecasting\phase9\horizon_performance_uci.png`

- `outputs\figures\forecasting\phase9\residual_distribution_synthetic.png`

- `outputs\figures\forecasting\phase9\residual_distribution_uci.png`

- `outputs\figures\forecasting\phase9\residual_over_time_synthetic.png`

- `outputs\figures\forecasting\phase9\residual_over_time_uci.png`

- `outputs\figures\forecasting\phase9\residual_vs_actual_synthetic.png`

- `outputs\figures\forecasting\phase9\residual_vs_actual_uci.png`

- `outputs\figures\forecasting\phase9\store_stability.png`

- `outputs\figures\forecasting\phase9\walk_forward_mae.png`

- `outputs\figures\forecasting\phase9\walk_forward_wape.png`

- `outputs\figures\forecasting\phase9\zero_demand_analysis.png`
