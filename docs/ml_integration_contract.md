# ML / Risk / Streamlit — Integration Contract

**Project: FORESIGHT — Demand & Inventory Intelligence**
**Document: `docs/ml_integration_contract.md`**
**Phase: Pre-Integration validation (before Phase 4 CAM)**

This contract defines the exact input and output schema the existing **ML
Forecasting Engine**, **Inventory Risk Engine**, and **Streamlit Application**
currently consume and produce. Phase 4 (Common Analytical Model) must satisfy
this contract so the three components keep working without modification.

> **Source of truth.** These components are validated against the **Phase 3
> cleaned datasets** in `data/processed/` via `src/data_integration.py`. The
> raw datasets are never read by the runtime stack.

---

## 1. Current component → dependency map

| Component | Entry-point module | Cleaned datasets it reads |
|---|---|---|
| ML Forecasting Engine | `src/forecasting.py` | `sales_daily_clean.parquet`, `sku_master_clean.csv`, `calendar_clean.csv` |
| Feature Engineering | `src/feature_engineering.py` | (consumes the above via forecasting) |
| Inventory Risk Engine | `src/risk_scoring.py` | `sales_daily_clean.parquet`, `inventory_snapshots_clean.parquet`, `sku_master_clean.csv`, `store_master_clean.csv`, `calendar_clean.csv` |
| Streamlit App | `dashboard/app.py` | all of the above + `customer_master_clean.csv` (catalog display) |
| Validation harness | `src/validate_ml_stack.py` | all of the above |

---

## 2. Forecasting Input (required columns)

The feature matrix is built at a single entity grain. The current runtime grain
is **SKU level** (`date, sku_id`), i.e. daily demand summed across stores.

| Column | Type | Source | Notes |
|---|---|---|---|
| `date` | `datetime64` | sales | Primary time axis |
| `sku_id` | `str` | sales / sku_master | Forecasting entity |
| `units_sold` | `int` | sales | **Forecast target** |
| `total_revenue` | `float` | sales | financial + derives `avg_unit_price` |
| `avg_unit_price` | `float` | sales | revenue-weighted; used for `discount_pct` |
| `transaction_count` | `int` | sales | summed on aggregation |
| `unique_customers` | `int` | sales | summed on aggregation |
| `promotion_flag` | `int {0,1}` | sales | `max` on aggregation (promo if any store) |
| `category`, `sub_category`, `brand`, `cost_price`, `base_price`, `lead_time_days`, `reorder_point`, `safety_stock` | various | sku_master | used to derive `discount_pct` |
| `is_holiday`, `season`, `month`, `quarter`, `day_of_week` ... | various | calendar + derived | calendar features |

### Notes
- **Entity column.** The consumer currently expects `sku_id` only.
- **`source_dataset`.** The ledger intends a `source_dataset` discriminator
  (`SYNTHETIC` / `UCI-ONLINE`). The runtime stack is synthetic-only right now;
  Phase 4 must populate this column and the app must branch on it for
  entity selection (synthetic: `STORE_ID`; UCI: online-only entity `ONLINE`).
- **Feature set used by ML models** (`FEATURE_COLS`, 25 features):
  calendar (month, quarter, day_of_month, day_of_week, is_weekend,
  sin/cos day_of_week, sin/cos month), pricing (discount_pct, promotion_flag),
  lags (1,2,3,7,14,21,28,30), rolling mean/std 7/14/30, ewm 7/28.
- **Phase 4 compatibility.** The app trains at SKU-total grain today. If Phase 4
  moves forecasting to store-SKU grain, the app's training path
  (`get_cached_forecasting_models`) and the forecast history builder must be
  switched together so training and inference stay on the same grain.

---

## 3. Inventory Input (required columns)

| Column | Type | Source | Notes |
|---|---|---|---|
| `date` | `datetime64` | inventory | snapshot date |
| `store_id` | `str` | inventory / store_master | risk entity dimension |
| `sku_id` | `str` | inventory / sku_master | risk SKU dimension |
| `ending_inventory` | `int` | inventory | **basis for stockout/DOS** |
| `on_order_qty` | `int` | inventory | included in net inventory position for reorder logic |
| `stockout_flag` | `int {0,1}` | inventory | incident indicator |
| `lead_time_days` | `int` | sku_master | replenishment lead time |
| `reorder_point` | `int` | sku_master | ROP trigger |
| `safety_stock` | `int` | sku_master | safety buffer |
| `beginning_inventory_pre_receipts` | `int` | inventory (Phase 3 derived) | equals `beginning_inventory − receipts` |
| `inventory_balance_ok` | `bool` | inventory (Phase 3 derived) | canonical equation holds |

### ⚠️ Inventory semantic (Phase 3 REVIEW)
Per the Phase 3 data-quality finding: the generator's `beginning_inventory`
**already includes the day's receipts**, so the validated relationship is
`ending_inventory = beginning_inventory − units_sold`, **not**
`beginning_inventory + receipts − units_sold`.

The risk engine already complies: it computes **Days of Supply** and **stockout
scores from `ending_inventory` directly** and uses
`ending_inventory + on_order_qty` for reorder triggers. It **never re-adds
`receipts`**. Phase 4 must preserve this semantic and carry the `REVIEW` status
downstream — do not "correct" the data by silently re-adding receipts.

---

## 4. Forecast Output (recommended schema)

Produced by `generate_multi_step_forecast()` (and matched by the app's baseline
path). The app plots `date`, `forecast_units`, `forecast_lower`,
`forecast_upper`.

| Column | Type | Notes |
|---|---|---|
| `date` | `datetime64` | one row per future day |
| `entity_id` | `str` | recommended for Phase 4 (SKU id today) |
| `sku_id` | `str` | required |
| `forecast_horizon` | `int` | days ahead (7/14/30) |
| `forecast_demand` | `float` | point forecast (aliased `forecast_units`) |
| `lower_bound` | `float` | 95% CI lower (aliased `forecast_lower`) |
| `upper_bound` | `float` | 95% CI upper (aliased `forecast_upper`) |
| `model_name` | `str` | algorithm used |

---

## 5. Risk Output (recommended schema)

Produced by `calculate_inventory_risk_matrix()`.

| Column | Type | Notes |
|---|---|---|
| `date` | `datetime64` | latest snapshot date |
| `entity_id` / `store_id` | `str` | risk entity |
| `sku_id` | `str` | required |
| `forecast_demand` | `float` | avg daily demand (30-d lookback) |
| `current_inventory` | `int` | = `ending_inventory` |
| `days_of_inventory` | `float` | `ending / effective_daily_demand` |
| `stockout_gap` | `float` | net gap vs reorder/safety coverage |
| `overstock_gap` | `int` | `excess_units` beyond target coverage |
| `risk_score` | `float` | `stockout_risk_score` + `overstock_risk_score` |
| `risk_level` | `str` | `LOW/SAFE`, `MEDIUM (REORDER)`, `CRITICAL/HIGH` |
| `reorder_required` | `bool` | `reorder_triggered` |
| `recommendation` | `str` | derived business action |

Risk levels are currently **hard-coded** in `risk_scoring.py` (stockout:
`≥70` CRITICAL / `≥35` MEDIUM; overstock: `≥65` SEVERE / `≥30` MODERATE).
No reorder trigger fires when `on_order_qty` already covers ROP — this is
verified data-consistent behaviour, not a bug.

---

## 6. Phase 4 integration plan (compatibility, not yet applied)

1. **Header row rules** — Phase 4 must emit every column above with the exact
   names, or the app must be re-pointed to aliases. No renaming without touching
   `dashboard/app.py`.
2. **Grain** — Phase 4 CAM runs at `date + entity_id + sku_id`. The app's
   forecasting currently assumes `sku_id`-only. Add an entity selector that is a
   no-op when the synthetic dataset (entity = store) is not selected, and switch
   the training + history builders together to avoid a grain mismatch.
3. **`source_dataset`** — must be added and the app must expose it as the top-level
   "Select source dataset" control; UCI is a single `ONLINE` entity (no fake stores).
4. **Inventory REVIEW flag** — Phase 4 must keep `beginning_inventory_pre_receipts`
   + `inventory_balance_ok` and surface the REVIEW status in the app's data-quality tab.
5. **Models** — persist to `models/*.joblib` and load via `MLDemandForecaster.load`;
   the app must never retrain on every interaction (already cached).

---

*Contract version 1.0 — validated against `data/processed/` on 2026-08-12 via
`src/validate_ml_stack.py` (32/32 checks).*