# Phase 10 — Forecasting Strategy Improvement

**Status:** COMPLETE  
**Validation:** 87/87 PASS

## 1. Executive Summary

**Decision (Option A — Major improvement):** Structural models substantially outperform frozen LightGBM.

SYNTHETIC hurdle vs Phase 8 TEST: WAPE 38.8923 -> 26.2505 (+32.50%). Zero-day positive prediction rate 82.64 -> 1.42.

Phase 8 LightGBM and Phase 9 stability artifacts were not modified.

## 2. Phase 8 Baseline

- UCI LightGBM TEST: WAPE=79.471, MAE=17.3447, sMAPE=82.8734
- SYNTHETIC LightGBM TEST: WAPE=38.8923, MAE=2.8156, sMAPE=113.6813
- Selection remains frozen; Phase 10 models are experimental comparators.

## 3. Zero-Demand Problem

SYNTHETIC TEST is ~61.27% zero-demand. Phase 8 LightGBM predicted positive demand on 82.64% of those zeros (zero-day MAE 1.44). UCI TEST has no coded zero-demand rows (invoice-day grain), so a hurdle model is not identified there.

## 4. Hurdle Model

Stage 1: LightGBM classifier P(demand>0), `is_unbalance=True`. Stage 2: LightGBM regressor trained only on actual demand>0. Thresholds tried on validation only: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]. Objective: min validation WAPE, then MAE, then nonzero MAE.

### UCI
Skipped: Train zero-demand share=0.00% < 5%. Hurdle is for intermittent demand; UCI rows are predominantly positive-demand invoice days (missing days are absent, not coded as zero).

### SYNTHETIC

- Best threshold (validation): **0.50**

- Train zeros: 62.36%

- Val classifier at selected threshold: ROC-AUC=0.9969, PR-AUC=0.9943, F1=0.9564, precision=0.9731, recall=0.9402, Brier=0.0224

- TEST hurdle: WAPE=26.2505 MAE=1.9004 sMAPE=13.5443 bias=-0.2557 zero_MAE=0.3 nonzero_MAE=4.4324 zero_pos_rate=1.42

Validation threshold table:

```
source_dataset      split  threshold    MAE   RMSE   sMAPE    WAPE    bias      n  zero_mae  nonzero_mae  zero_positive_prediction_rate  precision  recall     f1  false_positive_demand_rate  false_negative_demand_rate
     SYNTHETIC validation        0.5 1.9786 5.9627 11.4365 25.5028 -0.3437 146000    0.3296       5.3219                           1.28     0.9731  0.9402 0.9564                        1.28                        5.98
     SYNTHETIC validation        0.6 2.0055 6.1626 11.6751 25.8490 -0.6300 146000    0.1620       5.7430                           0.64     0.9860  0.9217 0.9528                        0.64                        7.83
     SYNTHETIC validation        0.7 2.1051 6.5242 12.5246 27.1332 -0.8951 146000    0.0682       6.2347                           0.28     0.9938  0.8990 0.9440                        0.28                       10.10
     SYNTHETIC validation        0.4 2.1183 6.1883 12.2149 27.3027  0.3011 146000    0.8742       4.6406                           3.17     0.9379  0.9697 0.9536                        3.17                        3.03
     SYNTHETIC validation        0.8 2.2600 6.9813 13.8382 29.1290 -1.1548 146000    0.0216       6.7981                           0.09     0.9979  0.8726 0.9311                        0.09                       12.74
     SYNTHETIC validation        0.3 2.3615 6.7285 13.9025 30.4376  0.7187 146000    1.3544       4.4032                           4.90     0.9080  0.9804 0.9428                        4.90                        1.96
     SYNTHETIC validation        0.2 3.2071 8.4836 19.5411 41.3374  1.8161 146000    2.7880       4.0569                           9.69     0.8349  0.9935 0.9073                        9.69                        0.65
```


Matched TEST comparison vs Phase 8:

- Phase 8 WAPE=38.8923 vs hurdle 26.2505
- Phase 8 zero-pos-rate=82.64 vs hurdle 1.42
- Phase 8 nonzero MAE=4.985 vs hurdle 4.4324

## 5. Intermittent Baselines

Croston / SBA / TSB / Naive implemented as rolling 1-step per series (alpha=0.1, TSB beta=0.1). Forecast uses only history before the origin, then updates with the realized actual. UCI skipped: zeros are not coded.

```
source_dataset   model  alpha  beta    MAE    RMSE    sMAPE     WAPE    bias      n  zero_mae  nonzero_mae  zero_positive_prediction_rate
     SYNTHETIC   naive    0.1   NaN 5.2717 10.4688  45.1750  72.8181 -0.0188 147000    1.7379      10.8625                          12.67
     SYNTHETIC croston    0.1   NaN 7.7901 10.4353 150.5177 107.6062  0.4565 147000    6.1796      10.3382                         100.00
     SYNTHETIC     sba    0.1   NaN 7.6817 10.3991 151.3849 106.1088  0.0717 147000    5.8706      10.5472                         100.00
     SYNTHETIC     tsb    0.1   0.1 7.4758 10.3056 151.0911 103.2639  0.1679 147000    5.6061      10.4339                         100.00
```
- **UCI:** UCI zero-row share=0.00%. Croston/SBA/TSB require observed zero demand intervals; UCI grain omits no-sale days, so intermittent baselines are not identified. Naive last-observation remains available via Phase 7.

## 6. Direct Multi-Horizon

Each horizon has its own LightGBM predicting `units_sold` at t+h from origin-t features. Target calendar (known in advance) is attached; demand lags stay at t. Train rows require target_date within the train window (no test labels in training).

```
source_dataset  horizon  train_n  val_n  test_n  training_time_sec  train_target_within_train  val_target_within_val  origin_precedes_target     MAE    RMSE    sMAPE    WAPE    bias
           UCI        1   392492  50006   76377              1.777                       True                   True                    True 18.8664 73.8456  87.4581 84.9963 -0.2918
           UCI        3   383909  44682   70571              1.838                       True                   True                    True 19.3957 76.1078  88.0596 85.1797 -0.5018
           UCI        7   367751  35894   60364              1.650                       True                   True                    True 20.3824 62.3516  87.2182 86.1841 -0.3912
           UCI       14   341975  24092   45526              1.594                       True                   True                    True 21.4391 63.2018  86.3329 84.8453 -1.2741
           UCI       30   291980   8345   20943              1.380                       True                   True                    True 25.4371 70.6540  82.9292 80.0420 -5.3536
     SYNTHETIC        1  1166000 145000  146000              9.881                       True                   True                    True  3.7274  6.0586 114.6869 51.4966  0.1204
     SYNTHETIC        3  1164000 143000  144000              8.738                       True                   True                    True  4.3730  6.6823 123.4012 60.6812  0.1261
     SYNTHETIC        7  1160000 139000  140000              8.636                       True                   True                    True  5.0214  7.2194 136.3732 69.8290  0.0871
     SYNTHETIC       14  1153000 132000  133000              8.191                       True                   True                    True  5.0694  7.3095 138.3648 70.4621  0.0611
     SYNTHETIC       30  1137000 116000  117000              8.044                       True                   True                    True  5.6689  7.9337 142.7820 79.0687  0.0574
```

Recursive (Phase 9) vs direct:

```
source_dataset  horizon  recursive_WAPE  direct_WAPE  wape_improvement_pct  recursive_MAE  direct_MAE  recursive_sMAPE  direct_sMAPE  recursive_bias  direct_bias  direct_n  recursive_n                                                                                                                 note
           UCI        1         79.4761      84.9963               -6.9457        28.5693     18.8664          76.9311       87.4581         -2.4823      -0.2918     76377         4839 Populations differ: Phase 9 recursive used strided origins (UCI 400-series cap); direct uses all valid test origins.
           UCI        3         79.5917      85.1797               -7.0208        29.6548     19.3957          76.6289       88.0596         -1.6682      -0.5018     70571         4759 Populations differ: Phase 9 recursive used strided origins (UCI 400-series cap); direct uses all valid test origins.
           UCI        7         89.7171      86.1841                3.9379        33.9680     20.3824          80.6896       87.2182         -0.3643      -0.3912     60364         4439 Populations differ: Phase 9 recursive used strided origins (UCI 400-series cap); direct uses all valid test origins.
           UCI       14         96.4946      84.8453               12.0725        37.2220     21.4391          81.3363       86.3329          0.7922      -1.2741     45526         3875 Populations differ: Phase 9 recursive used strided origins (UCI 400-series cap); direct uses all valid test origins.
           UCI       30        100.2543      80.0420               20.1610        38.8311     25.4371          82.7562       82.9292          2.0394      -5.3536     20943         2595 Populations differ: Phase 9 recursive used strided origins (UCI 400-series cap); direct uses all valid test origins.
     SYNTHETIC        1         64.3163      51.4966               19.9323         4.1399      3.7274         127.4303      114.6869          0.1631       0.1204    146000        21000 Populations differ: Phase 9 recursive used strided origins (UCI 400-series cap); direct uses all valid test origins.
     SYNTHETIC        3         82.8176      60.6812               26.7291         8.1481      4.3730         145.1359      123.4012         -0.2312       0.1261    144000        21000 Populations differ: Phase 9 recursive used strided origins (UCI 400-series cap); direct uses all valid test origins.
     SYNTHETIC        7        112.0417      69.8290               37.6759         7.4099      5.0214         151.6665      136.3732          0.4548       0.0871    140000        21000 Populations differ: Phase 9 recursive used strided origins (UCI 400-series cap); direct uses all valid test origins.
     SYNTHETIC       14        111.5234      70.4621               36.8186         7.3424      5.0694         153.7303      138.3648          0.2880       0.0611    133000        20000 Populations differ: Phase 9 recursive used strided origins (UCI 400-series cap); direct uses all valid test origins.
     SYNTHETIC       30        100.9901      79.0687               21.7065         6.5509      5.6689         155.4533      142.7820         -0.7547       0.0574    117000        17000 Populations differ: Phase 9 recursive used strided origins (UCI 400-series cap); direct uses all valid test origins.
```

Populations are not identical: Phase 9 used strided origins and a 400-series UCI cap. Directional comparison is still informative.

## 7. Prediction Intervals

Quantile LightGBM (objective=quantile) at 0.10/0.50/0.90. Raw quantiles may be negative; they are clipped at 0 because demand cannot be negative, then sorted to restore P10<=P50<=P90. Crossing counts are reported before reorder.

```
source_dataset  n_test  coverage_pct  mean_width  pinball_p10  pinball_p50  pinball_p90  p10_below_pct  p50_below_pct  p90_below_pct  n_crossed_before_reorder  n_negative_raw_p10  n_negative_raw_p50  n_negative_raw_p90  nonneg_clip  training_time_sec
           UCI   79545       82.3672     39.4081       2.0347       7.4040       6.3538          14.08          50.73          89.94                      1942                   0                 125                   6         True              8.332
     SYNTHETIC  147000       89.7673     10.0079       0.4652       1.2379       0.8975          64.04          78.77          93.80                      4264               25112                8201                2512         True             25.977
```
## 8. Hyperparameter Optimization

Eight LightGBM configs, selected by chronological validation WAPE then MAE. TEST evaluated once after selection. Config 0 is the Phase 8 frozen hyperparameter set.

```
source_dataset  config_id  is_phase8_config  num_leaves  learning_rate  n_estimators  min_child_samples  subsample  colsample_bytree  reg_alpha  reg_lambda  val_WAPE  val_MAE  val_RMSE  val_sMAPE  training_time_sec
           UCI          0              True          31           0.05           150                 20        0.8               0.8        0.0         0.0   76.9218  16.5035   55.8952    80.2343              1.407
           UCI          1             False          15           0.05           150                 20        0.8               0.8        0.0         0.0   80.3484  17.2387   57.3189    82.4433              1.171
           UCI          2             False          63           0.05           150                 20        0.8               0.8        0.0         0.0   75.4238  16.1821   55.8280    78.3387              2.378
           UCI          3             False          31           0.03           250                 20        0.8               0.8        0.0         0.0   76.2178  16.3524   55.8448    79.8614              2.853
           UCI          4             False          31           0.10           100                 20        0.8               0.8        0.0         0.0   76.2773  16.3652   56.1255    78.9658              1.271
           UCI          5             False          31           0.05           150                 50        0.8               0.8        0.0         0.0   77.0284  16.5264   55.8886    80.1500              1.862
           UCI          6             False          31           0.05           150                 20        0.8               0.8        0.1         1.0   76.9092  16.5008   56.3656    80.1988              1.548
           UCI          7             False          63           0.05           150                 20        0.7               0.7        0.0         0.0   75.1562  16.1247   55.5317    78.2933              1.974
     SYNTHETIC          0              True          31           0.05           150                 20        0.8               0.8        0.0         0.0   40.1044   3.1115    6.0244   126.2837              5.664
     SYNTHETIC          1             False          15           0.05           150                 20        0.8               0.8        0.0         0.0   47.3240   3.6716    6.7757   134.7369              4.728
     SYNTHETIC          2             False          63           0.05           150                 20        0.8               0.8        0.0         0.0   34.0479   2.6416    5.4133   118.9993              8.225
     SYNTHETIC          3             False          31           0.03           250                 20        0.8               0.8        0.0         0.0   39.9152   3.0968    5.9982   127.7725              8.978
     SYNTHETIC          4             False          31           0.10           100                 20        0.8               0.8        0.0         0.0   38.0618   2.9530    5.7829   116.7435              4.237
     SYNTHETIC          5             False          31           0.05           150                 50        0.8               0.8        0.0         0.0   40.0794   3.1095    6.0269   126.1264              6.735
     SYNTHETIC          6             False          31           0.05           150                 20        0.8               0.8        0.1         1.0   40.0725   3.1090    6.0209   126.0084              5.964
     SYNTHETIC          7             False          63           0.05           150                 20        0.7               0.7        0.0         0.0   34.2993   2.6611    5.4352   119.0863              7.200
```

Selected TEST scores:

```
source_dataset  best_config_id  best_is_phase8_config     MAE    RMSE    sMAPE    WAPE   bias      n
           UCI               7                  False 17.0953 70.7181  80.7257 78.3282 0.2512  79545
     SYNTHETIC               2                  False  2.4594  4.7121 104.2466 33.9724 0.0617 147000
```
## 9. Final Model Comparison

```
  Dataset           Model  Horizon     MAE    RMSE    sMAPE    WAPE        Bias Zero-demand Training time
      UCI Phase8_LightGBM        1 17.3447 70.8952  82.8734 79.4710 see Phase 9 see Phase 9        frozen
      UCI        HPO_cfg7        1 17.0953 70.7181  80.7257 78.3282      0.2512         n/a      see grid
SYNTHETIC Phase8_LightGBM        1  2.8156  5.1469 113.6813 38.8923 see Phase 9 see Phase 9        frozen
SYNTHETIC   Hurdle_th0.50        1  1.9004  5.0573  13.5443 26.2505     -0.2557        1.42        16.282
SYNTHETIC        HPO_cfg2        1  2.4594  4.7121 104.2466 33.9724      0.0617         n/a      see grid
```

## 10. Recommended Model

SYNTHETIC: hurdle (threshold=0.5) is preferred when zero-day false positives matter and overall WAPE/nonzero MAE stay close. UCI: keep Phase 8 LightGBM (hurdle not applicable).

Do not auto-replace Phase 8 artifacts. Direct models may be used for longer-horizon planning if they improve WAPE at h>=7. Quantile P10/P90 are diagnostic intervals, not a replacement point forecast unless P50 beats LightGBM on WAPE.

## 11. Remaining Limitations

- Hurdle threshold is a hard zero; it does not produce calibrated expected demand E[y]=P(y>0)*E[y|y>0] unless used as a mixture (not selected here).
- Direct vs recursive comparison uses different origin samples.
- Quantile models are independent and can cross before reorder.
- HPO search is small; not a full AutoML sweep.
- UCI intermittency is not identified in the current grain.
- No prediction intervals around the hurdle mixture.

## 12. Phase 11 Recommendation

Do **not** implement Phase 11 here. Evidence-based next step: inventory risk scoring / replenishment using the frozen Phase 8 1-step LightGBM as the demand engine, with optional SYNTHETIC hurdle overlay for zero-demand SKUs, and P10/P90 bands for safety-stock diagnostics. Productionize only after stakeholder review of zero-demand false positives vs missed demand.


## Overall decision

**Option A — Major improvement**

Structural models substantially outperform frozen LightGBM.

## Charts

- `outputs\figures\forecasting\phase10\zero_demand_threshold_analysis.png`

- `outputs\figures\forecasting\phase10\classifier_confusion_matrix.png`

- `outputs\figures\forecasting\phase10\hurdle_vs_lightgbm.png`

- `outputs\figures\forecasting\phase10\intermittent_baseline_comparison.png`

- `outputs\figures\forecasting\phase10\direct_vs_recursive_horizon.png`

- `outputs\figures\forecasting\phase10\horizon_improvement.png`

- `outputs\figures\forecasting\phase10\prediction_interval_coverage.png`

- `outputs\figures\forecasting\phase10\prediction_interval_width.png`

- `outputs\figures\forecasting\phase10\quantile_calibration.png`

- `outputs\figures\forecasting\phase10\hyperparameter_comparison.png`
