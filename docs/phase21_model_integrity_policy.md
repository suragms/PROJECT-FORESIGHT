# Phase 21 — Model Integrity Policy

## Frozen Artifacts

Phase 21 verifies integrity of:

1. **12 original frozen models** in `models/final/` (from `docs/final_model_registry.json`)
2. **Phase 20 production model** at `models/final/phase20/phase20_synthetic_lightgbm.joblib`

## Baseline

`docs/phase21_production_integrity_baseline.json` records:

- Model path, SHA-256, file size, model ID, registry status
- Phase 20 production model hash vs `docs/phase20_production_registry.json`

## Ongoing Monitoring

Each run produces `data/phase21/monitoring/model_integrity_report.json`:

| Status | Meaning |
|--------|---------|
| PASS | All hashes match baseline |
| FAIL | Hash mismatch or missing model |

## MODEL INTEGRITY ALERT

If any hash changes unexpectedly:

1. Report alert with severity CRITICAL
2. Do **not** repair or overwrite automatically
3. Investigate whether intentional promotion occurred outside Phase 20 process

## Phase 20 Provenance

Cross-checked against `docs/phase20_promotion_provenance.json` copy verification.

## Legacy Artifacts

`models/lightgbm_forecaster.joblib` is a **known legacy non-production artifact**. It is not part of the frozen 12-model stack and is not monitored as production.
