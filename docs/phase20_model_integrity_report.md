# Phase 20 — Model Integrity Report

**Date:** 2026-08-28

## Frozen Original Models (12)

| Check | Result |
|-------|--------|
| Pre-promotion hash snapshot | PASS — 12/12 match |
| Post-promotion verification | PASS — 12/12 unchanged |
| Registry entries modified | NO |

## Phase 17 Artifacts

| Artifact | Status |
|----------|--------|
| `models/phase17/synthetic/phase17_synthetic_lightgbm.joblib` | UNCHANGED |
| `data/phase17/` | UNCHANGED |

## Phase 19 Candidate

| Artifact | Status |
|----------|--------|
| `models/phase19/synthetic/phase19_synthetic_lightgbm.joblib` | UNCHANGED |
| SHA-256 matches promotion provenance source | VERIFIED |

## Phase 20 Promoted Artifact

| Field | Value |
|-------|-------|
| Path | `models/final/phase20/phase20_synthetic_lightgbm.joblib` |
| Parent | `phase19_synthetic_lightgbm` |
| Copy verified | YES (source SHA-256 == promoted SHA-256) |
| Registry | `docs/phase20_production_registry.json` (separate extension) |

## Lineage

```
phase17_synthetic_lightgbm (candidate)
        ↓
phase19_synthetic_lightgbm (hardened candidate)
        ↓ COPY (not overwrite)
phase20_synthetic_lightgbm (production)
```

Original 12 `models/final/*.joblib` files were NOT modified.
