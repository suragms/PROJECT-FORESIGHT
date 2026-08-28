# Phase 18 — Zidio Acceptance Matrix

**Evaluated candidate:** Phase 17 LightGBM (SYNTHETIC + UCI)  
**Evaluation date:** 2026-08-19

| Requirement | Evidence | SYNTHETIC | UCI |
|-------------|----------|-----------|-----|
| Reproducible pipeline | `src/run_phase17.py`; fixed seed=42; deterministic preprocessing; SHA-256 verified | PASS | PASS |
| Data-quality handling | `docs/phase17_data_quality_report.md`; all cleaning decisions documented with Problem/Decision/Reason/Impact | PASS | PASS |
| Weekly SKU forecast | Weekly aggregation at Monday-anchored period; 8-week horizon; `weekly_features.parquet` | PASS | PASS |
| Seasonal-naive baseline | Built before ML candidate; same-week-last-year; saved to parquet | PASS | PASS |
| Rolling-origin CV | 5-fold rolling-origin; strict temporal order verified; `backtest_results.parquet` | PASS | PASS |
| WAPE as primary metric | WAPE used throughout; MAPE not used as primary | PASS | PASS |
| No temporal leakage | 36 features audited; 0 FAIL; lag spot-check PASS | PASS | PASS |
| Stockout risk | Forecast-driven; lead-time demand vs inventory position + safety stock | PASS | NOT APPLICABLE |
| Overstock risk | Forward forecast demand vs on-hand; excess units calculated | PASS | NOT APPLICABLE |
| Recommended action | 4-quadrant grid: REORDER NOW / MARKDOWN / WATCH / HEALTHY; all verified consistent | PASS | NOT APPLICABLE |
| Rupee impact | `sales_at_risk` from `base_price`; `locked_capital` from `cost_price`; no fabrication | PARTIAL (sales_at_risk only; no overstock) | NOT APPLICABLE |
| Dashboard compatibility | Output schema compatible with Streamlit dashboard column contracts | PARTIAL (adapter required for weekly grain) | PARTIAL |
| Scoring service compatibility | FastAPI schema uses daily grain; weekly candidate requires API adapter | PARTIAL (adapter required) | PARTIAL |

**Summary:**

| Dataset | Zidio Acceptance | Status |
|---------|-----------------|--------|
| SYNTHETIC | 10/13 PASS, 2 PARTIAL, 1 N/A | **PARTIAL** |
| UCI | 7/13 PASS, 2 PARTIAL, 4 N/A | **PARTIAL** |

Dashboard and scoring service require schema adapters for weekly grain. This is a known and expected integration step, not a fundamental failure.
