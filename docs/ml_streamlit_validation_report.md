# ML / Risk / Streamlit — Validation Report

**Project: FORESIGHT — Demand & Inventory Intelligence**
**Document: `docs/ml_streamlit_validation_report.md`**
**Date:** 2026-08-12 · **Method:** code review + runtime smoke tests

## 1. Scope & method

Validated the existing (already-built) ML forecasting engine, inventory risk
engine, and Streamlit application against the **Phase 3 cleaned datasets**
(`data/processed/`). No components were rebuilt. Validation used:

1. Static code review of `src/forecasting.py`, `src/evaluation.py`,
   `src/feature_engineering.py`, `src/risk_scoring.py`, `src/data_integration.py`,
   `dashboard/app.py`.
2. Headless runtime via Streamlit's `AppTest` harness (8 tabs, default + widget
   interactions: baseline model, store, SKU, horizon).
3. A re-runnable harness: `src/validate_ml_stack.py` (32 checks, exit 0).

---

## 2. Validation matrix

| # | Component | Test | Result | Status | Notes |
|---|---|---|---|---|---|
| 1 | Data layer | Cleaned datasets load | `sales 1,461,000 · inv 1,461,000` | ✅ PASS | date span 2022-01-01 → 2025-12-31 |
| 2 | Data layer | Grain `(date, store, sku)` unique | 0 duplicates | ✅ PASS | |
| 3 | Data layer | Phase-3 inventory derived cols present | `beginning_inventory_pre_receipts`, `inventory_balance_ok` | ✅ PASS | |
| 4 | Models | `lightgbm_forecaster.joblib` exists / loads / predicts | 25 features, non-neg preds | ✅ PASS | |
| 5 | Models | `xgboost_forecaster.joblib` exists / loads / predicts | 25 features, non-neg preds | ✅ PASS | |
| 6 | Models | `random_forest_forecaster.joblib` exists / loads / predicts | 25 features, non-neg preds | ✅ PASS | |
| 7 | Feature eng | Feature matrix builds at SKU-total grain | 140,190 rows | ✅ PASS | full series, no random sampling |
| 8 | Feature eng | Lag/rolling/EWM features present | 8 lags, 3 rolling, 2 EWM | ✅ PASS | |
| 9 | Feature eng | **No target leakage** — `lag_1 == prev calendar-day actual` | verified | ✅ PASS | rolling shifted by 1; time-ordered split |
| 10 | Benchmark | Leaderboard has MAE/RMSE/WAPE/MAPE/R² | yes | ✅ PASS | |
| 11 | Benchmark | **Best model on WAPE is ML** (not naive) | LightGBM WAPE 27.95% | ✅ PASS | NAIVE 66.03% — ML is ~2.4× better |
| 12 | Forecast | Magnitude matches recent history | 211 vs 247 (ratio 1.17) | ✅ PASS | fixed from 8.9× under-scale |
| 13 | Forecast | CI bounds present & consistent | lower ≤ point ≤ upper | ✅ PASS | |
| 14 | Risk | Risk matrix computes | 1,000 rows | ✅ PASS | latest snapshot × 10 stores × 100 SKUs |
| 15 | Risk | Required fields present | DOS, scores, levels, ROQ, capital | ✅ PASS | |
| 16 | Risk | Scores bounded 0–100; DOS ≥ 0 | yes | ✅ PASS | |
| 17 | Risk | **Inventory semantic respected** | receipts never re-added; net position = ending + on_order | ✅ PASS | Phase-3 REVIEW carried |
| 18 | Risk | 10 Core Questions pipeline runs | 10 keys | ✅ PASS | 733 critical stockouts reported |
| 19 | App | Headless startup, all 8 tabs | 0 exceptions | ✅ PASS | AppTest, default selections |
| 20 | App | Switch model → Seasonal Naive | 0 exceptions | ✅ PASS | now a *real* baseline forecast |
| 21 | App | Select specific store | 0 exceptions | ✅ PASS | selectors restricted to 10 active stores |
| 22 | App | Select different SKU | 0 exceptions | ✅ PASS | selectors restricted to 100 tracked SKUs |
| 23 | App | Change forecast horizon 14 / 30 | 0 exceptions | ✅ PASS | |
| 24 | App | Insufficient-history path | user warning, no traceback | ✅ PASS | error-handling guard added |

---

## 3. Bugs found & fixes applied

| # | Severity | Bug | Fix |
|---|---|---|---|
| 1 | **Critical (analytical)** | **Grain mismatch.** Models trained on a random sample of store-SKU rows (~7 units/day) but applied to a SKU's store-total series (~74 units/day) → every forecast under-scaled **~9×** (top SKU: predicted 28 vs actual 247 units/day). | Train on the **full SKU-total series** (aggregate stores → `date, sku_id`) via new `aggregate_daily_sales()` in `feature_engineering.py`; app trains via no-arg `get_cached_forecasting_models()`. Models retrained & re-saved. Forecast now 211 vs 247. |
| 2 | **High (leakage-adjacent)** | **Random sampling broke the time series** — lags/rolling features were computed over a random 100k-row subset with non-contiguous dates. | Training now uses the complete contiguous series; split remains time-ordered (last 30 days = test). |
| 3 | Medium (misleading UI) | **Baseline selector was fake** — "Seasonal Naive (7D)" / "7-Day Moving Average" silently ran LightGBM. | Wired to `BaselineForecaster`; baseline forecasts now produced via a matched forecast frame. |
| 4 | Medium (integrity) | **Fabricated KPI** — executive card claimed "+12.4% vs prev cycle" (not computed). | Replaced with real YoY growth (2025 vs 2024 revenue). |
| 5 | Medium (robustness) | Selecting a non-tracking store/SKU (only 10 of 30 stores and 100 of 5,000 SKUs have data) could crash the forecast on empty history. | Store & SKU selectors restricted to active entities; `len(history) < 7` → `st.warning`, no traceback (Tab 2 + Tab 3). |
| 6 | Low (display) | Tab 8 mixed str/int `Records` column → Arrow serialization error (auto-recovered, mis-typed). | Records made uniformly strings; UCI status corrected to `REVIEW`. |
| 7 | Low (wiring) | `@st.cache_resource` took unhashable DataFrame args (worked, but fragile). | Cache function made arg-less. |

### Verified NOT bugs
- **`reorder_triggered = 0` despite 733 critical stockouts** — data-consistent:
  the generator only places an order when `inv ≤ ROP and on_order == 0`, so net
  position `ending + on_order` covers ROP. The engine correctly includes
  `on_order_qty` in the trigger.
- **Inventory equation** — engine already uses `ending_inventory` and never
  re-adds receipts (respects the Phase-3 REVIEW semantic).

---

## 4. Performance observations

| Area | Observation |
|---|---|
| First app load | ~28 s (includes one-time SKU-total model training, cached via `st.cache_resource`) |
| Subsequent loads | fast — datasets cached (`st.cache_data`) and models cached (no retrain per interaction) |
| Training | 3 models × 137k rows ≈ 26 s; feature matrix build ≈ 1 s |
| Risk engine | ~1–2 s per run |
| Data | Parquet for large facts; CSV only for small dimension tables |

---

## 5. Remaining limitations (documented, not blocking)

1. **MAPE is capped at 500** in `evaluation.py` — meaningless for intermittent
   retail demand; **WAPE is the primary metric** (already the leaderboard sort key).
2. **95% CI is a heuristic** (`±1.96·std_7·√(1+step/10)`) — not a calibrated
   prediction interval. Acceptable for a dashboard; flagged for Phase 9 upgrades.
3. **Risk levels are hard-coded** thresholds in `risk_scoring.py` (not yet
   configurable). Documented in the integration contract.
4. **UCI Online Retail II is not yet wired into the runtime stack** — the app is
   synthetic-only. Phase 4 must add `source_dataset` + the `ONLINE` entity.
5. **SKU-level (not store-SKU) forecasts** — chosen to match the app's UI; Phase 4
   should move training + inference together to `date + entity + sku`.
6. `use_container_width` deprecation warnings in Streamlit 1.59 (cosmetic, works).

---

## 6. Sign-off

All 24 validation tests pass. 7 issues fixed (1 critical analytical bug, 1
leakage-adjacent training flaw, 5 UI/robustness/integrity issues). Existing
working functionality preserved; models retrained at the correct grain and
re-persisted. Ready for **Phase 4 — Data Integration & Common Analytical Model**.
