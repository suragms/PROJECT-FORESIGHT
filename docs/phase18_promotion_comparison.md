# Phase 18 — Frozen Production vs Phase 17 Candidate Comparison

| Dimension | Frozen Production | Phase 17 Candidate |
|-----------|------------------|-------------------|
| **Data provenance** | Repository raw files; Phase 3 cleaning; SHA-256 locked | Same raw files; Phase 17 cleaning; SHA-256 verified |
| **Forecast grain** | Daily (date + source_dataset + entity_id + product_key) | **Weekly** (week + source_dataset + product_key) — closer to Zidio spec |
| **Horizon** | 1, 3, 7, 14, 30 days | 8 weeks (Zidio spec: 6–8 weeks) |
| **Features** | Phase 6 pipeline: 970 lines; daily lags/rolling/EWM; calendar; price; promotions | Phase 17 pipeline: 46 columns; weekly lags/rolling/EWM; calendar; price_lag1; promo_lag1 |
| **Model** | LightGBM point + hurdle + quantile (12 artifacts) | LightGBM point (2 artifacts) |
| **WAPE (SYNTHETIC)** | 26.25% at h=1 daily | 14.42% at h=8w weekly (not directly comparable; different grain) |
| **WAPE (UCI)** | 79.47% at h=1 daily | 64.19% at h=8w weekly (not directly comparable) |
| **Bias (SYNTHETIC)** | -0.26 (h=1 daily) | +19.78 (h=8w aggregate); +3 to +47 across folds (holiday spike) |
| **Bias (UCI)** | From Phase 8 report | -10.61 (under-forecast, HIGH severity) |
| **Backtesting** | Walk-forward daily splits | Rolling-origin weekly, 5 folds, verified temporal ordering |
| **Leakage status** | PASS (Phase 6 audit) | **PASS** (Phase 17/18 audit; 36 features; 0 FAIL) |
| **Risk logic** | Historical demand (30-day lookback avg) | **Forecast-driven demand** — genuine improvement |
| **Inventory integration** | Latest snapshot + historical avg | Latest snapshot + **forecast** demand |
| **Explainability** | Not independently audited in Phase 18 | Top features extracted; all PASS leakage review |
| **Reproducibility** | SHA-256 hashes locked in registry | **REPRODUCIBLE** — max_diff = 0.00 on 500-sample spot check |
| **Production compatibility** | Deployed; daily grain API; daily dashboard | Requires schema adapters for weekly grain |

## Key Differences

1. **Grain:** The Phase 17 candidate uses weekly grain, which better matches the Zidio weekly SKU-level specification. The frozen production models use daily grain.
2. **Risk logic:** Phase 17 uses forecast-driven risk rather than historical-demand risk — a methodological improvement.
3. **WAPE comparability:** The two WAPE values cannot be directly compared because the grain, horizon, and aggregation differ. Neither is definitively "better" without a controlled head-to-head on the same grain.
4. **Quantile/hurdle models:** The frozen production stack includes quantile and hurdle companions; Phase 17 provides point forecasts only.
5. **Horizon coverage:** The frozen stack covers 1–30 day horizons; Phase 17 covers 8 weeks only.

## Conclusion

Phase 17 is not a straightforward replacement. It addresses the weekly grain requirement and improves the risk engine, but introduces new concerns (holiday-window bias spike, no quantile/hurdle companions, API adapter requirement). Promotion requires a controlled promotion phase with these gaps explicitly addressed.
