# Phase 17 — Frozen Production vs Candidate Comparison

| Dimension | Frozen Production | Phase 17 Candidate |
|-----------|------------------|-------------------|
| Dataset | UCI + SYNTHETIC (daily grain) | UCI + SYNTHETIC (weekly grain) |
| Provenance | Repository raw files, Phase 3 cleaning | Same raw files, Phase 17 cleaning |
| Grain | date + source_dataset + entity_id + product_key | week + source_dataset + product_key |
| Features | Phase 6 (970 lines; daily lags, rolling, EWM, calendar, price, promo) | Phase 17 (46 cols; weekly lags, rolling, EWM, calendar, price_lag, promo_lag) |
| Model | LightGBM point + hurdle + quantile companions | LightGBM point |
| Horizon | 1, 3, 7, 14, 30 days | 8 weeks |
| WAPE (SYNTHETIC) | 26.25% (h=1 daily) | 14.42% (h=8 weekly) |
| WAPE (UCI) | 79.47% (h=1 daily) | 64.19% (h=8 weekly) |
| Bias (SYNTHETIC) | -0.26 (h=1) | From backtest |
| Validation | Walk-forward daily splits | Rolling-origin weekly, 5 folds |
| Risk logic | Historical demand (30-day lookback avg) | **Forecast-driven demand** |
| Inventory usage | Latest inventory snapshot + historical avg | Latest snapshot + forecast demand |
| Business alignment | Daily granularity | **Weekly granularity (Zidio spec)** |

**Note:** WAPE values across daily and weekly grain are not directly comparable.
