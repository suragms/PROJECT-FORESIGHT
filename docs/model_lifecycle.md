# Model lifecycle

## Current production candidates (Phase 11)

Selected models live in `models/final/` and `docs/final_model_registry.json`. Status values:

- `selected` — point-forecast model for a dataset × horizon
- `interval_companion` — P10/P90 quantile models for h=1 only

Do **not** overwrite Phase 8 files in `models/uci_best_model.joblib` or `models/synthetic_best_model.joblib`.

## Change process

1. Train/evaluate in a new experiment (do not mutate frozen Phase 8/9 artifacts).
2. Record TEST metrics with the same WAPE/MAE/bias definitions.
3. Walk-forward if replacing UCI h=1 (Phase 9 fold-2 is the known failure mode).
4. Assign a new `model_id`, write a new joblib under `models/final/`, SHA-256 it.
5. Append a registry record. Never edit hashes of live files in place.
6. Re-run `src/validate_final_forecasting.py` and `src/validate_phase12.py`.

## Retraining policy

Phase 12 inference **never retrains**. The API, batch CLI, and forecast dashboard only load registered joblibs.

## Rollback

Point the registry `model_file` + `hash` back to the previous joblib. Keep old files on disk.
