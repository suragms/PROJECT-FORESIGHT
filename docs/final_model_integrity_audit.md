# Project Foresight — Final Model Integrity Audit

**Audit date:** 2026-08-29  
**Method:** SHA-256 verification against registries + production routing code inspection

---

## Frozen 12 models (`models/final/`)

| Check | Result |
|-------|--------|
| Registry | `docs/final_model_registry.json` |
| Models on disk | 12/12 |
| Hash verification | **12/12 PASS** |
| Runtime test evidence | `tests/test_phase23_navigation.py::TestIntegrity`, `test_regression_artifacts.py` |

All frozen models unchanged from registered hashes.

---

## Phase 20 production model

| Field | Verified value |
|-------|----------------|
| Model ID | `phase20_synthetic_lightgbm` |
| Path | `models/final/phase20/phase20_synthetic_lightgbm.joblib` |
| Registry | `docs/phase20_production_registry.json` |
| Hash | `96a88f1dbb8e1904f2c0b79877afe7bfe30ef5336f8d4598dc07d6adf895e086` |
| Hash match | **PASS** |
| Parent | `phase19_synthetic_lightgbm` (same hash — copy promotion) |
| Forecast grain | Weekly SKU-level |
| Validated horizon | **6 weeks** (h1–h6) |
| Extended h7–h8 | Documented as **PARTIAL** accuracy |
| Feature contract | **45 features** (`docs/phase20_feature_contract.json`) |
| UCI in production | **Rejected** at adapter level |

E2E evidence: `docs/phase20_e2e_results.json` — 6/6 smoke tests pass, lineage match true.

---

## Legacy model: `models/lightgbm_forecaster.joblib`

| Check | Result |
|-------|--------|
| File exists | **YES** |
| In `models/final/` registry | **NO** |
| In Phase 20 production adapter | **NO** |
| In Phase 20 routes | **NO** |
| Referenced in | `src/phase9_common.py`, `src/phase10_common.py`, `src/validate_ml_stack.py`, `src/cam_adapter.py`, `src/config.py` |
| Production routing classification | **NOT USED IN PRODUCTION** |
| Issue class | **LEGACY NON-PRODUCTION ARTIFACT ISSUE** |
| Historical impact | Phase 12 validator 41/42 vs documented 42/42 |

Documentation correctly classifies this artifact. Hash mismatch vs Phase 11 snapshot is a **DOCUMENTATION MISMATCH / LEGACY ARTIFACT ISSUE**, not a production failure.

---

## Candidate models (non-production)

| Model | Location | Role |
|-------|----------|------|
| `phase17_synthetic_lightgbm` | `models/phase17/synthetic/` | Phase 17 candidate |
| `phase17_uci_lightgbm` | `models/phase17/uci/` | UCI research candidate |
| `phase19_synthetic_lightgbm` | `models/phase19/synthetic/` | Parent of Phase 20 promotion |

Phase 17 model hash recorded in `docs/phase22_final_audit.json` integrity snapshot.

---

## Monitoring integrity (Phase 21)

`data/phase21/monitoring/model_integrity_report.json` — frozen 12/12 and Phase 20 checks consumed by dashboards.  
Phase 21 does **not** modify model files (observability-only — verified in source: no retrain calls in `run_phase21.py` monitoring path).

---

## Summary

| Category | Status |
|----------|--------|
| Frozen 12/12 | **PASS** |
| Phase 20 production | **PASS** |
| Legacy `lightgbm_forecaster.joblib` | **LEGACY ISSUE** (not production) |
| UCI accidentally in production | **PASS** (explicitly rejected) |
