# Phase 20 — Rollback Plan

## 1. Disable Phase 20 Promoted Model

- Remove or comment out the Phase 20 router include in `src/api/app.py`:
  ```python
  # application.include_router(phase20_router, prefix="/phase20", tags=["phase20"])
  ```
- Stop serving `dashboard/phase20_production.py` if deployed separately.
- Route all forecast requests back to existing `/forecast` endpoints using Phase 11 registered models.

## 2. Restore Previous Routing

- Existing `/forecast` and `/forecast/batch` routes remain unchanged and continue to use `models/final/` Phase 11 artifacts via `ForecastEngine`.
- No changes required to historical registry (`docs/final_model_registry.json`).

## 3. Preserve Historical Artifacts

Do NOT delete:
- `models/phase17/` — Phase 17 historical candidates
- `models/phase19/` — Phase 19 hardened candidate
- `data/phase17/`, `data/phase19/` — validation evidence
- `docs/phase17_*`, `docs/phase18_*`, `docs/phase19_*` — audit trail

Phase 20 promoted copy at `models/final/phase20/` may be archived but should not be deleted without documentation.

## 4. Rollback Verification

After rollback:
1. Run `python src/validate_phase12.py` — verify 12 frozen models (41/42 expected if legacy hash issue persists).
2. Verify `docs/final_model_registry.json` hashes match `models/final/*.joblib` (excluding `phase20/` subdirectory).
3. Confirm `/forecast` API responds with Phase 11 models.
4. Confirm Phase 20 endpoints return 404 or are unreachable.

## 5. No Retraining Required

Rollback is routing-only. No model retraining or data regeneration is needed.
