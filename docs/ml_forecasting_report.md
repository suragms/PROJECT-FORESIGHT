# Phase 8 — Machine Learning Demand Forecasting Report

**Validation:** 57/57 PASS

## 1. Objective

Train source-specific ML demand models on Phase 6 features and beat Phase 7 baseline WAPE benchmarks on the untouched TEST window.

## 2. Input data

- Path: `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\features\forecast_features.parquet`

- Rows: **1,995,496** | Columns: **62**

## 3. Feature groups

Numeric: calendar, cyclical, lag, rolling, demand trend, price; Synthetic extras: promo, inventory, store size. Categorical (frequency-encoded): season; Synthetic also category/sub_category/brand/region/store_type.

### Excluded fields

- `units_sold`: TARGET — never used as a predictor

- `revenue`: Same-day revenue leaks target (≈ price × units)

- `transaction_count`: Same-day order activity contemporaneous with demand

- `unique_customers`: Same-day customer activity contemporaneous with demand

- `date`: Temporal key — encoded via calendar features

- `source_dataset`: Partition key — models trained per source

- `entity_id`: Raw ID — use store attributes instead

- `product_key`: Raw ID — use product attributes instead

- `sku_id`: Raw ID duplicate of product identity

- `entity_type`: Near-constant within source; not used as numeric code

- `split`: Metadata for chronological partitioning only

- `insufficient_history`: Metadata flag — rows dropped via lag availability

## 4. Models evaluated

random_forest, hist_gradient_boosting, lightgbm, xgboost (CatBoost not installed — skipped).

## 5. Training configuration

- `random_state=42`

- Selection metric order: WAPE → MAE → RMSE → sMAPE on VALIDATION

- TEST used only after selection

## 6. Validation results

source_dataset                  model     MAE    RMSE    sMAPE    WAPE  training_time
     SYNTHETIC               lightgbm  3.1115  6.0244 126.2837 40.1044          7.631
     SYNTHETIC          random_forest  3.2346  6.3700 144.7665 41.6907          9.148
     SYNTHETIC hist_gradient_boosting  3.2370  6.2381 122.2162 41.7219         13.145
     SYNTHETIC                xgboost  3.4757  6.5580 120.3094 44.7993         11.799
           UCI               lightgbm 16.5035 55.8952  80.2343 76.9218          1.749
           UCI                xgboost 16.6620 59.6071  78.2229 77.6605          3.051
           UCI hist_gradient_boosting 16.6774 58.9465  79.3097 77.7322          5.004
           UCI          random_forest 18.3963 57.6516  84.4070 85.7442          4.076


## 7. Test results

source_dataset                  model     MAE    RMSE    sMAPE    WAPE  baseline_WAPE  wape_improvement_pct selected
     SYNTHETIC               lightgbm  2.8156  5.1469 113.6813 38.8923        72.8181               46.5898     True
     SYNTHETIC hist_gradient_boosting  2.9036  5.2873 110.4674 40.1072        72.8181               44.9214    False
     SYNTHETIC          random_forest  2.9845  5.4106 134.9162 41.2255        72.8181               43.3856    False
     SYNTHETIC                xgboost  3.0637  5.4736 109.0139 42.3187        72.8181               41.8844    False
           UCI hist_gradient_boosting 17.2792 71.9372  82.0480 79.1708        86.3870                8.3533    False
           UCI               lightgbm 17.3447 70.8952  82.8734 79.4710        86.3870                8.0058     True
           UCI                xgboost 17.5616 71.4968  80.6523 80.4644        86.3870                6.8559    False
           UCI          random_forest 18.5642 71.6098  85.9231 85.0585        86.3870                1.5378    False


## 8–11. Baseline comparison & best models

### UCI

- Best ML model: **lightgbm**

- Baseline: moving_average_30 WAPE=86.387

- ML TEST WAPE=79.4710 | MAE=17.3447 | RMSE=70.8952 | sMAPE=82.8734

- WAPE improvement %: **8.0058**

- Beat baseline on TEST: **YES**

OBSERVATION: lightgbm reduced TEST WAPE vs moving_average_30.

EVIDENCE: baseline WAPE=86.387, ML WAPE=79.4710, improvement=8.0058%.

BUSINESS INTERPRETATION: Lag/calendar/price features add predictive structure beyond the baseline.

BUSINESS ACTION: Prefer the selected ML model for planning with inventory constraints.

### SYNTHETIC

- Best ML model: **lightgbm**

- Baseline: naive WAPE=72.8181

- ML TEST WAPE=38.8923 | MAE=2.8156 | RMSE=5.1469 | sMAPE=113.6813

- WAPE improvement %: **46.5898**

- Beat baseline on TEST: **YES**

OBSERVATION: lightgbm reduced TEST WAPE vs naive.

EVIDENCE: baseline WAPE=72.8181, ML WAPE=38.8923, improvement=46.5898%.

BUSINESS INTERPRETATION: Lag/calendar/price features add predictive structure beyond the baseline.

BUSINESS ACTION: Prefer the selected ML model for planning with inventory constraints.

## 12. Product-level performance

### UCI

Best WAPE SKUs:

- `UCI_40046A` (ONLINE): WAPE=0.32, MAE=0.04

- `UCI_47586A` (ONLINE): WAPE=2.43, MAE=0.15

- `UCI_21458` (ONLINE): WAPE=3.10, MAE=0.37

Worst WAPE SKUs:

- `UCI_79063D` (ONLINE): WAPE=17554.51, MAE=234.06

- `UCI_90214U` (ONLINE): WAPE=4455.56, MAE=534.67

- `UCI_90071` (ONLINE): WAPE=2821.86, MAE=34.49

source_dataset       segment  n_skus  revenue_share_pct    model     MAE    RMSE     WAPE
           UCI  high_revenue     671              80.02 lightgbm 22.7186 43.4074  74.9877
           UCI lower_revenue    2497              19.98 lightgbm 12.5598 19.4243 187.3979


### SYNTHETIC

Best WAPE SKUs:

- `SYN_SKU_00068` (STORE_005): WAPE=14.56, MAE=5.12

- `SYN_SKU_00017` (STORE_001): WAPE=15.66, MAE=1.48

- `SYN_SKU_00068` (STORE_006): WAPE=15.68, MAE=4.64

Worst WAPE SKUs:

- `SYN_SKU_00043` (STORE_002): WAPE=158.74, MAE=3.79

- `SYN_SKU_00085` (STORE_008): WAPE=156.27, MAE=3.72

- `SYN_SKU_00043` (STORE_009): WAPE=147.23, MAE=3.52

source_dataset       segment  n_skus  revenue_share_pct    model    MAE   RMSE    WAPE
     SYNTHETIC  high_revenue      35              80.31 lightgbm 3.1530 5.3406 40.5558
     SYNTHETIC lower_revenue      65              19.69 lightgbm 2.6339 4.6271 55.7156


## 13. Store-level performance (Synthetic)

entity_id    MAE   RMSE    WAPE
STORE_005 2.7886 5.0175 37.8151
STORE_009 2.7362 4.9127 38.0589
STORE_001 2.7713 5.1224 38.1943
STORE_006 2.8593 5.2000 38.4936
STORE_002 2.7693 5.0403 38.8348
STORE_008 2.9047 5.3696 38.8528
STORE_007 2.8874 5.3858 39.1453
STORE_004 2.9101 5.3979 39.2797
STORE_010 2.7314 4.8924 40.1581
STORE_003 2.7978 5.0986 40.2323

Best store: STORE_005 WAPE=37.8151; Worst: STORE_003 WAPE=40.2323

## 14. Feature importance (top 10 native)

### UCI

 rank            feature  importance
    1 average_unit_price       795.0
    2        price_lag_1       491.0
    3   units_sold_lag_1       270.0
    4    rolling_mean_30       219.0
    5    demand_change_7       209.0
    6     rolling_std_30       207.0
    7  units_sold_lag_14       195.0
    8     rolling_mean_7       170.0
    9        day_of_year       163.0
   10   units_sold_lag_3       145.0


### SYNTHETIC

 rank           feature  importance
    1   rolling_mean_30       416.0
    2      on_order_qty       348.0
    3   rolling_mean_14       296.0
    4   demand_change_7       231.0
    5  ending_inventory       214.0
    6    rolling_mean_7       211.0
    7  demand_growth_30       208.0
    8    rolling_std_30       204.0
    9        base_price       182.0
   10 units_sold_lag_14       170.0


Feature importance is correlational — not causal.

## 15. Error analysis

### UCI

- source_dataset: UCI

- model: lightgbm

- bias_mean_pred_minus_actual: 0.0335

- median_error: 3.8346

- mean_abs_error: 17.3447

- p95_abs_error: 55.0873

- zero_demand_share_pct: 0.0

- zero_demand_mae: nan

- high_demand_mae: 75.4367

- pct_overpredict: 71.72

- pct_underpredict: 28.28

### SYNTHETIC

- source_dataset: SYNTHETIC

- model: lightgbm

- bias_mean_pred_minus_actual: 0.0873

- median_error: 0.5393

- mean_abs_error: 2.8156

- p95_abs_error: 10.4314

- zero_demand_share_pct: 61.27

- zero_demand_mae: 1.4444

- high_demand_mae: 8.5931

- pct_overpredict: 67.24

- pct_underpredict: 22.12

## 16. Model limitations

- No walk-forward CV in this phase (deferred to Phase 9).

- RF uses row subsampling on large Synthetic train for runtime.

- Frequency encoding collapses rare categories.

- Same-day average_unit_price treated as known price signal.

- No prediction intervals.

## 17. Phase 9 recommendations

1. Walk-forward / rolling origin validation for stability.

2. Residual diagnostics and horizon-specific metrics.

3. Intermittent-demand methods for sparse UCI SKUs.

4. Statistical comparison of ML vs baseline (Diebold-Mariano).

5. Calibrate inventory-aware decision thresholds.

## Charts

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\feature_importance.png`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\uci_high_volume.png`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\uci_high_revenue.png`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\syn_store_sku.png`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\syn_high_revenue.png`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\ml_vs_baseline_wape.png`

## Files

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\ml\ml_model_metrics.parquet`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\ml\feature_importance.parquet`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\ml\ml_metrics_by_product_uci.parquet`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\ml\ml_metrics_by_entity_uci.parquet`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\ml\ml_high_value_uci.parquet`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\ml\ml_metrics_by_product_synthetic.parquet`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\ml\ml_metrics_by_entity_synthetic.parquet`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\ml\ml_high_value_synthetic.parquet`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\ml\ml_error_analysis.parquet`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\data\processed\forecasts\ml\ml_predictions.parquet`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\models\uci_best_model.joblib`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\models\synthetic_best_model.joblib`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\docs\model_training_metadata.json`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\feature_importance.png`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\uci_high_volume.png`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\uci_high_revenue.png`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\syn_store_sku.png`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\syn_high_revenue.png`

- `C:\Users\SURAG\Documents\zidio\Project_FORESIGHT\Demand-Inventory-Intelligence\outputs\figures\forecasting\ml\ml_vs_baseline_wape.png`
