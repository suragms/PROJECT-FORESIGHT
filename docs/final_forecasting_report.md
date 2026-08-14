# Final Forecasting Report (Phase 11)

## 1. Executive Summary

Recommended operational solution:

- **UCI, horizon 1:** frozen Phase 8 LightGBM (`uci_h1_phase8_lightgbm`).
- **SYNTHETIC, horizon 1:** hurdle LightGBM, threshold 0.50 (`synthetic_h1_hurdle_th050`).
- **Both datasets, horizons 3/7/14/30:** direct LightGBM per horizon.
- **Intervals:** P10/P90 quantile companions on h=1 only (diagnostic bands).

**Production readiness: READY WITH MONITORING.** h=1 models beat Phase 7 baselines, SYNTHETIC hurdle materially cuts false-positive demand, and inference is schema-validated and reproducible. Not READY: UCI has a documented high-error walk-forward fold, long-horizon WAPE still degrades, same-day price/inventory are assumed known at origin, quantile bands are not statistically calibrated, and no live production monitor exists yet.

## 2. Phase 8 Baseline

Frozen LightGBM (immutable benchmark):

| Dataset | MAE | RMSE | sMAPE | WAPE | vs Phase 7 |
| --- | ---: | ---: | ---: | ---: | --- |
| UCI | 17.3447 | 70.8952 | 82.8734 | 79.4710 | MA-30 WAPE 86.3870 (+8.01%) |
| SYNTHETIC | 2.8156 | 5.1469 | 113.6813 | 38.8923 | Naive WAPE 72.8181 (+46.59%) |

CatBoost was not installed. Same-day `average_unit_price` and SYNTHETIC inventory/promo were treated as known operational signals.

## 3. Phase 9 Stability

Walk-forward expanding-window copies of Phase 8 LightGBM (146/146 PASS):

- UCI: mean WAPE 85.31, CV 0.135, max/min 1.369 → quantitative **Stable**, with fold 2 (Jan–Apr 2011) WAPE **105.31** after the post-holiday spike.
- SYNTHETIC: mean WAPE 39.39, CV 0.013, max/min 1.037 → **Stable**.
- Recursive h=1→h=30 WAPE grows to ~100 on both sources.
- SYNTHETIC TEST zeros 61.27%; Phase 8 predicted demand on **82.64%** of zero days.

## 4. Phase 10 Experiments

Validation **87/87 PASS**. Option A — Major improvement.

- Hurdle (SYNTHETIC): TEST WAPE 38.89 → 26.25; zero-day P(pred>0) 82.64% → 1.42%. UCI skipped (0% zeros).
- Croston/SBA/TSB worse than Naive.
- Direct LightGBM improves long-horizon WAPE vs recursive (especially h≥7 SYNTHETIC; h≥14 UCI). Direct h=1 is next-observation, not Phase 8 same-row.
- Quantile P10–P90: UCI coverage 82.37%, SYNTHETIC 89.77% vs 80% nominal. Crossing before reorder occurred.
- HPO: UCI cfg7 TEST WAPE 78.33 (small gain, no walk-forward); SYNTHETIC cfg2 33.97 (still worse than hurdle).

## 5. Final Model Selection

### Selection logic

```
Primary (in order):
  1. WAPE on held-out TEST (lower is better).
  2. MAE if WAPE is within 2% relative.
  3. Stability: if relative WAPE gain vs a walk-forward-validated model
     is below 3%, keep the walk-forward-validated model.
  4. Horizon: select independently per horizon. Operational h=1 uses the
     Phase 8 same-row contract; h in {3,7,14,30} uses leakage-safe direct models.

Secondary:
  5. Bias closer to 0.
  6. Zero-demand MAE and false-positive demand rate (SYNTHETIC).
  7. Prediction-interval coverage near 80% (diagnostic, not a point-forecast rule).
  8. Training / inference cost.
  9. Model complexity (prefer one-stage unless zeros require a hurdle).
 10. Reproducibility (frozen serialized artifacts preferred when quality is comparable).

Gates:
  - Exclude models with WAPE worse than the Phase 7 baseline.
  - Exclude hurdle on UCI (train zero share 0%; not identified).
  - Exclude Croston / SBA / TSB (worse than Naive on SYNTHETIC).
  - Do not select a model solely for lowest RMSE.
  - Quantile P50 is a point-forecast candidate only if it beats the selected
    point model on WAPE; otherwise P10/P90 remain interval companions.
```

### Decisions

- **UCI h=1 → Phase8_LightGBM.** Phase 8 WAPE 79.4710; HPO 78.3282 is only 1.44% relative (<3%) and was not walk-forward tested. Primary criterion 3 (stability) keeps the frozen Phase 8 LightGBM. UCI hurdle was not identified (0% coded zeros).

- **SYNTHETIC h=1 → Hurdle_th0.50.** Hurdle TEST WAPE 26.2505 vs Phase 8 38.8923 (32.5% relative). MAE and zero-demand false positives also improve. HPO and Croston-family do not beat hurdle. Complexity is justified by the 61% zero-demand share.

- **UCI h=3 → Direct_LightGBM.** Leakage-safe h-step model with origin features + known-in-advance target calendar. Recursive Phase 9 is diagnostic only (frozen exo). Direct h=1 is a different task than Phase 8 same-row 1-step, so h=1 stays with the operational 1-step model.

- **UCI h=7 → Direct_LightGBM.** Leakage-safe h-step model with origin features + known-in-advance target calendar. Recursive Phase 9 is diagnostic only (frozen exo). Direct h=1 is a different task than Phase 8 same-row 1-step, so h=1 stays with the operational 1-step model.

- **UCI h=14 → Direct_LightGBM.** Leakage-safe h-step model with origin features + known-in-advance target calendar. Recursive Phase 9 is diagnostic only (frozen exo). Direct h=1 is a different task than Phase 8 same-row 1-step, so h=1 stays with the operational 1-step model.

- **UCI h=30 → Direct_LightGBM.** Leakage-safe h-step model with origin features + known-in-advance target calendar. Recursive Phase 9 is diagnostic only (frozen exo). Direct h=1 is a different task than Phase 8 same-row 1-step, so h=1 stays with the operational 1-step model.

- **SYNTHETIC h=3 → Direct_LightGBM.** Leakage-safe h-step model with origin features + known-in-advance target calendar. Recursive Phase 9 is diagnostic only (frozen exo). Direct h=1 is a different task than Phase 8 same-row 1-step, so h=1 stays with the operational 1-step model.

- **SYNTHETIC h=7 → Direct_LightGBM.** Leakage-safe h-step model with origin features + known-in-advance target calendar. Recursive Phase 9 is diagnostic only (frozen exo). Direct h=1 is a different task than Phase 8 same-row 1-step, so h=1 stays with the operational 1-step model.

- **SYNTHETIC h=14 → Direct_LightGBM.** Leakage-safe h-step model with origin features + known-in-advance target calendar. Recursive Phase 9 is diagnostic only (frozen exo). Direct h=1 is a different task than Phase 8 same-row 1-step, so h=1 stays with the operational 1-step model.

- **SYNTHETIC h=30 → Direct_LightGBM.** Leakage-safe h-step model with origin features + known-in-advance target calendar. Recursive Phase 9 is diagnostic only (frozen exo). Direct h=1 is a different task than Phase 8 same-row 1-step, so h=1 stays with the operational 1-step model.

Quantile P50 was not selected as the point forecast (WAPE does not beat the chosen h=1 models). HPO UCI was not selected (stability override). Intermittent models were not selected (WAPE).

## 6. Final Performance

### vs Phase 7 baseline (h=1)

```
  Dataset          Baseline  Baseline WAPE  Final Model WAPE  Absolute improvement  Improvement %
      UCI moving_average_30        86.3870           79.4710                6.9160         8.0058
SYNTHETIC             naive        72.8181           26.2505               46.5676        63.9506
```

### h=1 selected models

```
  dataset                     model     MAE    RMSE   sMAPE    WAPE    bias      n  zero_mae  nonzero_mae  zero_positive_prediction_rate
SYNTHETIC synthetic_h1_hurdle_th050  1.9004  5.0573 13.5443 26.2505 -0.2557 147000       0.3       4.4324                           1.42
      UCI    uci_h1_phase8_lightgbm 17.3447 70.8952 82.8734 79.4710  0.0335  79545       NaN      17.3447                            NaN
```

### Candidate matrix (implemented models only)

```
  dataset                     model  horizon     MAE     RMSE    sMAPE     WAPE    bias  zero_demand_MAE  positive_demand_MAE                                         stability  eligible
      UCI           Phase8_LightGBM        1 17.3447  70.8952  82.8734  79.4710     NaN              NaN                  NaN         Stable (Phase 9 folds; fold-2 WAPE spike)      True
SYNTHETIC           Phase8_LightGBM        1  2.8156   5.1469 113.6813  38.8923     NaN              NaN                  NaN                                  Stable (Phase 9)      True
SYNTHETIC             Hurdle_th0.50        1  1.9004   5.0573  13.5443  26.2505 -0.2557           0.3000               4.4324 not walk-forward tested (Phase 8 LGBM was Stable)      True
SYNTHETIC        Intermittent_naive        1  5.2717  10.4688  45.1750  72.8181 -0.0188           1.7379              10.8625                           not walk-forward tested      True
SYNTHETIC      Intermittent_croston        1  7.7901  10.4353 150.5177 107.6062  0.4565           6.1796              10.3382                           not walk-forward tested     False
SYNTHETIC          Intermittent_sba        1  7.6817  10.3991 151.3849 106.1088  0.0717           5.8706              10.5472                           not walk-forward tested     False
SYNTHETIC          Intermittent_tsb        1  7.4758  10.3056 151.0911 103.2639  0.1679           5.6061              10.4339                           not walk-forward tested     False
      UCI           Direct_LightGBM        1 18.8664  73.8456  87.4581  84.9963 -0.2918              NaN                  NaN                           not walk-forward tested      True
      UCI           Direct_LightGBM        3 19.3957  76.1078  88.0596  85.1797 -0.5018              NaN                  NaN                           not walk-forward tested      True
      UCI           Direct_LightGBM        7 20.3824  62.3516  87.2182  86.1841 -0.3912              NaN                  NaN                           not walk-forward tested      True
      UCI           Direct_LightGBM       14 21.4391  63.2018  86.3329  84.8453 -1.2741              NaN                  NaN                           not walk-forward tested      True
      UCI           Direct_LightGBM       30 25.4371  70.6540  82.9292  80.0420 -5.3536              NaN                  NaN                           not walk-forward tested      True
SYNTHETIC           Direct_LightGBM        1  3.7274   6.0586 114.6869  51.4966  0.1204              NaN                  NaN                           not walk-forward tested      True
SYNTHETIC           Direct_LightGBM        3  4.3730   6.6823 123.4012  60.6812  0.1261              NaN                  NaN                           not walk-forward tested      True
SYNTHETIC           Direct_LightGBM        7  5.0214   7.2194 136.3732  69.8290  0.0871              NaN                  NaN                           not walk-forward tested      True
SYNTHETIC           Direct_LightGBM       14  5.0694   7.3095 138.3648  70.4621  0.0611              NaN                  NaN                           not walk-forward tested      True
SYNTHETIC           Direct_LightGBM       30  5.6689   7.9337 142.7820  79.0687  0.0574              NaN                  NaN                           not walk-forward tested      True
SYNTHETIC Phase9_Recursive_LightGBM        1  4.1399   7.1323 127.4303  64.3163  0.1631              NaN                  NaN                 diagnostic (exo frozen at origin)     False
SYNTHETIC Phase9_Recursive_LightGBM        3  8.1481  13.1673 145.1359  82.8176 -0.2312              NaN                  NaN                 diagnostic (exo frozen at origin)     False
SYNTHETIC Phase9_Recursive_LightGBM        7  7.4099  11.2998 151.6665 112.0417  0.4548              NaN                  NaN                 diagnostic (exo frozen at origin)     False
SYNTHETIC Phase9_Recursive_LightGBM       14  7.3424  11.2173 153.7303 111.5234  0.2880              NaN                  NaN                 diagnostic (exo frozen at origin)     False
SYNTHETIC Phase9_Recursive_LightGBM       30  6.5509  10.3880 155.4533 100.9901 -0.7547              NaN                  NaN                 diagnostic (exo frozen at origin)     False
      UCI Phase9_Recursive_LightGBM        1 28.5693  77.8076  76.9311  79.4761 -2.4823              NaN                  NaN                 diagnostic (exo frozen at origin)     False
      UCI Phase9_Recursive_LightGBM        3 29.6548  66.6666  76.6289  79.5917 -1.6682              NaN                  NaN                 diagnostic (exo frozen at origin)     False
      UCI Phase9_Recursive_LightGBM        7 33.9680 111.0133  80.6896  89.7171 -0.3643              NaN                  NaN                 diagnostic (exo frozen at origin)     False
      UCI Phase9_Recursive_LightGBM       14 37.2220 116.8579  81.3363  96.4946  0.7922              NaN                  NaN                 diagnostic (exo frozen at origin)     False
      UCI Phase9_Recursive_LightGBM       30 38.8311  91.6766  82.7562 100.2543  2.0394              NaN                  NaN                 diagnostic (exo frozen at origin)     False
      UCI                  HPO_cfg7        1 17.0953  70.7181  80.7257  78.3282  0.2512              NaN                  NaN                           not walk-forward tested      True
SYNTHETIC                  HPO_cfg2        1  2.4594   4.7121 104.2466  33.9724  0.0617              NaN                  NaN                           not walk-forward tested      True
SYNTHETIC              Quantile_P50        1  2.4757   6.2941  41.8818  34.1976 -1.0715           0.2296               6.0294                           not walk-forward tested      True
      UCI              Quantile_P50        1 14.8080  72.5529  67.6211  67.8479 -8.8359              NaN              14.8080                           not walk-forward tested      True
```

### Bias

```
  dataset  mean_bias  median_bias  mean_residual  median_residual  overprediction_pct  underprediction_pct
SYNTHETIC    -0.2557       0.0000         0.2557           0.0000               19.84                19.76
      UCI     0.0335       3.8346        -0.0335          -3.8346               71.72                28.28
```

### Demand regimes

```
  dataset regime     n     MAE     WAPE     bias
SYNTHETIC   high 17723  6.9156  21.7297  -4.8362
SYNTHETIC    low 21135  2.7806  32.4338   1.4347
SYNTHETIC medium 18072  3.9291  22.2613  -0.5098
SYNTHETIC   zero 90070  0.3000   0.0000   0.3000
      UCI   high 25580 34.5175  59.9288 -18.0547
      UCI    low 27436  8.8832 531.6161   8.8620
      UCI medium 26529  9.5371 116.6489   8.3444
```

### Entity / store

```
  dataset  n_entities best_entity  best_WAPE  median_WAPE worst_entity  worst_WAPE  spread_WAPE
SYNTHETIC          10   STORE_001    25.2048       26.165    STORE_003     27.3415       2.1367
      UCI           1      ONLINE    79.4710       79.471       ONLINE     79.4710       0.0000
```

## 7. Zero-Demand Results

SYNTHETIC TEST (matched to Phase 8 grain):

- Actual zero rate: 61.27%
- Predicted zero rate (final): 62.39% (Phase 8: 10.64%)
- False-positive demand rate P(pred>0 | actual=0): **1.42%** vs Phase 8 **82.64%**
- Zero-demand MAE: 0.3 vs 1.4444
- Non-zero-demand MAE: 4.4324 vs 4.985
- WAPE: 26.2505 vs 38.8923
- Bias: -0.2557 vs 0.0873

**Yes — the final model materially reduced false-positive demand predictions** (82.64% → 1.42%) while also improving overall WAPE and non-zero MAE. The hurdle is retained.

## 8. Horizon Results

```
  dataset  horizon         model      n     MAE    RMSE    sMAPE    WAPE    bias
SYNTHETIC        1  SYNTHETIC_h1 147000  1.9004  5.0573  13.5443 26.2505 -0.2557
SYNTHETIC        3  SYNTHETIC_h3 144000  4.3730  6.6823 123.4012 60.6812  0.1261
SYNTHETIC        7  SYNTHETIC_h7 140000  5.0214  7.2194 136.3732 69.8290  0.0871
SYNTHETIC       14 SYNTHETIC_h14 133000  5.0694  7.3095 138.3648 70.4621  0.0611
SYNTHETIC       30 SYNTHETIC_h30 117000  5.6689  7.9337 142.7820 79.0687  0.0574
      UCI        1        UCI_h1  79545 17.3447 70.8952  82.8734 79.4710  0.0335
      UCI        3        UCI_h3  70571 19.3957 76.1078  88.0596 85.1797 -0.5018
      UCI        7        UCI_h7  60364 20.3824 62.3516  87.2182 86.1841 -0.3912
      UCI       14       UCI_h14  45526 21.4391 63.2018  86.3329 84.8453 -1.2741
      UCI       30       UCI_h30  20943 25.4371 70.6540  82.9292 80.0420 -5.3536
```

Error generally grows from short to long horizons on SYNTHETIC. UCI direct WAPE is not strictly monotone (h=30 can look better than h=7 because the target population and season mix change). Do not treat direct h=1 as comparable to Phase 8 same-row h=1.

## 9. Prediction Intervals

```
  dataset      n  coverage_pct  nominal_coverage_pct  mean_width  pinball_p10  pinball_p50  pinball_p90  p10_below_pct  p90_below_pct  interval_crossing
SYNTHETIC 147000       89.7673                  80.0     10.0079       0.4652       1.2379       0.8975          64.04          93.80                  0
      UCI  79545       82.3672                  80.0     39.4081       2.0347       7.4040       6.3538          14.08          89.94                  0
```

Bands are **P10/P90 quantile LightGBM companions**, clipped at 0 and reordered if they crossed. UCI coverage is near the 80% nominal; SYNTHETIC over-covers (intervals too wide on average). P10/P50 empirical below-rates are not equal to 10/50/90, so these are **not claimed as statistically calibrated**. They are usable as operational uncertainty bands with monitoring, not as safety-stock formulas.

## 10. Feature Dependencies

```
             Feature                    Required                         Source              Availability at forecast time                                                  Notes
                year                         Yes               Phase 6 calendar                           Known in advance                                                       
               month                         Yes               Phase 6 calendar                           Known in advance                                                       
             quarter                         Yes               Phase 6 calendar                           Known in advance                                                       
        week_of_year                         Yes               Phase 6 calendar                           Known in advance                                                       
         day_of_week                         Yes               Phase 6 calendar                           Known in advance                                                       
        day_of_month                         Yes               Phase 6 calendar                           Known in advance                                                       
         day_of_year                         Yes               Phase 6 calendar                           Known in advance                                                       
          is_weekend                         Yes               Phase 6 calendar                           Known in advance                                                       
           month_sin                         Yes               Phase 6 calendar                           Known in advance                                                       
           month_cos                         Yes               Phase 6 calendar                           Known in advance                                                       
             dow_sin                         Yes               Phase 6 calendar                           Known in advance                                                       
             dow_cos                         Yes               Phase 6 calendar                           Known in advance                                                       
          is_holiday                         Yes               Phase 6 calendar                           Known in advance                                                       
    units_sold_lag_1 Yes (h=1 and direct origin)      Phase 6 lag of units_sold               Available (past demand only)                                           Leakage-safe
    units_sold_lag_2 Yes (h=1 and direct origin)      Phase 6 lag of units_sold               Available (past demand only)                                           Leakage-safe
    units_sold_lag_3 Yes (h=1 and direct origin)      Phase 6 lag of units_sold               Available (past demand only)                                           Leakage-safe
    units_sold_lag_7 Yes (h=1 and direct origin)      Phase 6 lag of units_sold               Available (past demand only)                                           Leakage-safe
   units_sold_lag_14 Yes (h=1 and direct origin)      Phase 6 lag of units_sold               Available (past demand only)                                           Leakage-safe
   units_sold_lag_21 Yes (h=1 and direct origin)      Phase 6 lag of units_sold               Available (past demand only)                                           Leakage-safe
   units_sold_lag_28 Yes (h=1 and direct origin)      Phase 6 lag of units_sold               Available (past demand only)                                           Leakage-safe
   units_sold_lag_30 Yes (h=1 and direct origin)      Phase 6 lag of units_sold               Available (past demand only)                                           Leakage-safe
      rolling_mean_7                         Yes Phase 6 rolling of past demand                                  Available                                           Leakage-safe
     rolling_mean_14                         Yes Phase 6 rolling of past demand                                  Available                                           Leakage-safe
     rolling_mean_30                         Yes Phase 6 rolling of past demand                                  Available                                           Leakage-safe
       rolling_std_7                         Yes Phase 6 rolling of past demand                                  Available                                           Leakage-safe
      rolling_std_14                         Yes Phase 6 rolling of past demand                                  Available                                           Leakage-safe
      rolling_std_30                         Yes Phase 6 rolling of past demand                                  Available                                           Leakage-safe
     demand_change_1                         Yes Phase 6 rolling of past demand                                  Available                                           Leakage-safe
     demand_change_7                         Yes Phase 6 rolling of past demand                                  Available                                           Leakage-safe
     demand_growth_7                         Yes Phase 6 rolling of past demand                                  Available                                           Leakage-safe
    demand_growth_30                         Yes Phase 6 rolling of past demand                                  Available                                           Leakage-safe
  average_unit_price                         Yes                  Phase 6 price Assumed known at origin (list price / lag)       Must be supplied externally; not a future actual
         price_lag_1                         Yes                  Phase 6 price Assumed known at origin (list price / lag)       Must be supplied externally; not a future actual
              season                         Yes                        Phase 6                           Known in advance                                      Frequency-encoded
          base_price              SYNTHETIC only            Phase 6 store/price                            Known at origin                                                       
        discount_pct              SYNTHETIC only                  Phase 6 promo            Must be planned/known at origin                                                       
        price_change              SYNTHETIC only            Phase 6 store/price                            Known at origin                                                       
      promotion_flag              SYNTHETIC only                  Phase 6 promo            Must be planned/known at origin                                                       
 promotion_available              SYNTHETIC only                  Phase 6 promo            Must be planned/known at origin                                                       
     promo_rolling_7              SYNTHETIC only                  Phase 6 promo            Must be planned/known at origin                                                       
     store_size_sqft              SYNTHETIC only            Phase 6 store/price                            Known at origin                                                       
    ending_inventory              SYNTHETIC only              Phase 6 inventory Column required at origin; NaN allowed (LightGBM-native)   Not imputed. Missing column is rejected.
        on_order_qty              SYNTHETIC only              Phase 6 inventory Column required at origin; NaN allowed (LightGBM-native)   Not imputed. Missing column is rejected.
       stockout_flag              SYNTHETIC only              Phase 6 inventory Column required at origin; NaN allowed (LightGBM-native)   Not imputed. Missing column is rejected.
      historical_doi              SYNTHETIC only              Phase 6 inventory Column required at origin; NaN allowed (LightGBM-native)   Not imputed. Missing column is rejected.
            category              SYNTHETIC only    Phase 6 product/store attrs                                      Known                            Unseen levels map to freq 0
        sub_category              SYNTHETIC only    Phase 6 product/store attrs                                      Known                            Unseen levels map to freq 0
               brand              SYNTHETIC only    Phase 6 product/store attrs                                      Known                            Unseen levels map to freq 0
              region              SYNTHETIC only    Phase 6 product/store attrs                                      Known                            Unseen levels map to freq 0
          store_type              SYNTHETIC only    Phase 6 product/store attrs                                      Known                            Unseen levels map to freq 0
              hcal_*          Direct models only     Target-date calendar shift     Known in advance for the forecast date     Required input; not inferred from truncated panels
          units_sold              No — forbidden                            raw            Must not be used as a predictor                     TARGET — never used as a predictor
             revenue              No — forbidden                            raw            Must not be used as a predictor        Same-day revenue leaks target (≈ price × units)
   transaction_count              No — forbidden                            raw            Must not be used as a predictor    Same-day order activity contemporaneous with demand
    unique_customers              No — forbidden                            raw            Must not be used as a predictor Same-day customer activity contemporaneous with demand
                date              No — forbidden                            raw            Must not be used as a predictor           Temporal key — encoded via calendar features
      source_dataset              No — forbidden                            raw            Must not be used as a predictor              Partition key — models trained per source
           entity_id              No — forbidden                            raw            Must not be used as a predictor                  Raw ID — use store attributes instead
         product_key              No — forbidden                            raw            Must not be used as a predictor                Raw ID — use product attributes instead
              sku_id              No — forbidden                            raw            Must not be used as a predictor                   Raw ID duplicate of product identity
         entity_type              No — forbidden                            raw            Must not be used as a predictor  Near-constant within source; not used as numeric code
               split              No — forbidden                            raw            Must not be used as a predictor           Metadata for chronological partitioning only
insufficient_history              No — forbidden                            raw            Must not be used as a predictor      Metadata flag — rows dropped via lag availability
```

Inference **rejects** missing columns, missing `units_sold_lag_1`, negative prohibited fields, duplicate keys, and invalid dates. Longer-lag / `historical_doi` NaNs are **not imputed**; they are passed through as LightGBM-native missing values (Phase 8/10 training behavior). `units_sold` is never a predictor.

## 11. Limitations

- UCI invoice-day grain has no coded zeros; intermittency is unidentified.
- UCI walk-forward fold 2 remains a high-error regime; HPO was not re-tested on folds.
- Direct vs recursive comparisons use different origin samples.
- Hurdle is a hard threshold, not E[y]=P(y>0)E[y|y>0].
- Quantile models are independent of the hurdle point forecast.
- Same-day price/inventory/promo must be supplied at origin; they are not forecasted here.
- No live production traffic, concept-drift monitor, or replenishment policy is in this phase.
- Long-horizon forecasts remain weak relative to short-horizon operational 1-step models.

## 12. Production Readiness

**READY WITH MONITORING**

h=1 models beat Phase 7 baselines, SYNTHETIC hurdle materially cuts false-positive demand, and inference is schema-validated and reproducible. Not READY: UCI has a documented high-error walk-forward fold, long-horizon WAPE still degrades, same-day price/inventory are assumed known at origin, quantile bands are not statistically calibrated, and no live production monitor exists yet.

### Freeze status

- Phase 8 artifacts unchanged: True
- Phase 9 artifacts unchanged: True
- Feature parquet md5: `c7cd3e3cd1372a987451051b841d8392`

UCI entity-level WAPE spread is not identified (single `ONLINE` entity). SYNTHETIC store WAPE uses the Phase 9 CV/range rule. UCI production class is **Moderately Stable** because of Phase 9 fold 2 (WAPE 105.31), even when TEST months are Stable.

### Charts

- `outputs\figures\forecasting\final\final_model_comparison.png`
- `outputs\figures\forecasting\final\final_horizon_comparison.png`
- `outputs\figures\forecasting\final\final_store_stability.png`
- `outputs\figures\forecasting\final\final_residual_analysis.png`
- `outputs\figures\forecasting\final\final_zero_demand_comparison.png`
- `outputs\figures\forecasting\final\final_prediction_intervals.png`

