# Phase 16.1 — Validation Documentation Reconciliation

**Project:** Demand & Inventory Intelligence · **Phase:** 16.1
**Date:** 2026-08-19 · **Type:** Documentation-only — no model, data, or code changes

---

## Purpose

This phase reconciles project documentation with the actual executable validation results. Several documents (primarily `docs/phase15_executive_summary.md` and `docs/phase15_final_system_report.md`) reported optimistic validation counts that do not match the runtime output of the validation scripts.

---

## Before (documented values)

| Gate | Documented |
|------|-----------|
| Phase 12 | 42/42 PASS |
| Phase 13 | 42/42 PASS |
| Phase 14 | 19/19 PASS |
| Pytest | 59/59 PASS |
| Phase 15 | 12/12 PASS |

---

## Verified Runtime (2026-08-19)

| Gate | Command | Result |
|------|---------|--------|
| Phase 12 | `python src/validate_phase12.py` | **41/42 PASS** |
| Phase 13 | `python src/validate_phase13.py` | **41/42 PASS** |
| Phase 14 | `python src/validate_phase14.py` | **17/19 PASS** |
| Pytest | `python -m pytest tests -q` | **87 passed, 1 failed** (88 total; includes 19 Phase 16 tests) |
| Phase 15 | `python src/validate_phase15.py` | **9/12 PASS** |

---

## Discrepancies

### Discrepancy 1: Phase 12 — 42/42 vs 41/42

**Failing gate:** `Phase 8 unchanged: ['models\\lightgbm_forecaster.joblib']`

**Classification:** Stale artifact

**Root cause:** The file `models/lightgbm_forecaster.joblib` (a legacy Phase 8 intermediate model) has a different SHA-256 hash than what was recorded in the Phase 11 metadata snapshot. This legacy file sits outside `models/final/` and is **not** used by the frozen forecasting stack. All 12 frozen models in `models/final/` have verified, unchanged hashes.

**Reference:** `docs/phase15_known_limitations.md` does not explicitly document this failure, but the known limitations document states: "Frozen hashes must remain unchanged unless a human-approved Phase 11 replacement is executed." The failure is in a legacy intermediate model file, not a frozen production model.

### Discrepancy 2: Phase 13 — 42/42 vs 41/42

**Failing gate:** Same as Phase 12 — the Phase 13 validator runs the Phase 12 suite as a regression check and inherits its failure.

**Classification:** Expected cascading failure

### Discrepancy 3: Phase 14 — 19/19 vs 17/19

**Failing gates:**
1. `Phase 12 Regression FAIL` — cascades from Phase 12's 41/42
2. `Phase 13 Regression FAIL` — cascades from Phase 13's 41/42

All 17 other Phase 14 gates (API, security, inference, Docker, monitoring, dashboard, etc.) PASS.

**Classification:** Expected cascading failure

### Discrepancy 4: Pytest — 59/59 vs 87/88

**Change in total count:** Phase 16 added 19 new tests in `tests/test_phase16_dataset_alignment.py`. The pre-Phase-16 count was 69 tests (59 original + 10 added in various phases).

**Failing test:** `test_phase8_hashes_match_phase11_snapshot` in `tests/test_regression_artifacts.py`

**Classification:** Stale artifact (same root cause as Phase 12 failure)

### Discrepancy 5: Phase 15 — 12/12 vs 9/12

The `docs/phase15_final_system_report.md` status matrix previously reported "12/12 PASS". The runtime result is 9/12 PASS. The Phase 15 metadata file (`docs/phase15_metadata.json`) already correctly recorded "9/12 PASS" — the status matrix was inconsistent with its own metadata.

**Classification:** Documentation mismatch

---

## Root Cause Summary

All validation failures trace to a **single root cause**: the legacy file `models/lightgbm_forecaster.joblib` has a stale hash. This file is:

- An intermediate Phase 8 benchmark model
- Located in `models/` (NOT in `models/final/`)
- **Not used** by the frozen forecasting stack
- **Not referenced** by `docs/final_model_registry.json`
- **Not loaded** by `src/final_forecasting.py`

The 12 frozen production models in `models/final/` all have verified SHA-256 hashes that match the model registry exactly. No production model integrity issue exists.

---

## Changes Made

Documentation-only updates:

1. `docs/phase15_executive_summary.md` — replaced 42/42, 19/19, 59/59 with actual runtime results; added "Historical recorded result" vs "Current runtime validation result" columns
2. `docs/phase15_final_system_report.md` — updated model validation evidence table and final phase status matrix with actual runtime counts
3. `README.md` — updated Project Status table with actual validation counts

---

## Model Integrity

**CONFIRMED UNCHANGED**

All 12 frozen model SHA-256 hashes match `docs/final_model_registry.json`:

| Model | Hash (first 20) | Status |
|-------|-----------------|--------|
| uci_h1_phase8_lightgbm | `331909f0fe191c0b9cb0` | PASS |
| synthetic_h1_hurdle_th050 | `59a2b72024861d7f9c82` | PASS |
| uci_h3_direct_lightgbm | `28403041bd349d68993c` | PASS |
| uci_h7_direct_lightgbm | `ce215ffcd0cf3a5db148` | PASS |
| uci_h14_direct_lightgbm | `905b7859f163a9623cd3` | PASS |
| uci_h30_direct_lightgbm | `d87831a4aa65a28aad5f` | PASS |
| uci_h1_quantile_p10p90 | `294e1dbbda3633218204` | PASS |
| synthetic_h3_direct_lightgbm | `2c896d2c4589a24457ee` | PASS |
| synthetic_h7_direct_lightgbm | `f6df774aa0a9e287912e` | PASS |
| synthetic_h14_direct_lightgbm | `a31dbb6cc97618a068c5` | PASS |
| synthetic_h30_direct_lightgbm | `2b9124359964f112efac` | PASS |
| synthetic_h1_quantile_p10p90 | `9c09a25717cab87f2d59` | PASS |

---

## Data Integrity

**CONFIRMED UNCHANGED**

All 9 raw data file hashes match the Phase 16 baseline (`data/raw/.raw_hashes_phase16.json`). No processed data, forecast artifacts, or risk artifacts were modified.

---

## Final Status

```
PHASE 16.1 COMPLETE
```
