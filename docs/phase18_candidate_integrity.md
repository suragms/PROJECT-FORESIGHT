# Phase 18 — Candidate Artifact Integrity

## Phase 17 Candidate Models

| Property | UCI Candidate | SYNTHETIC Candidate |
|----------|--------------|---------------------|
| Path | `models/phase17/uci/phase17_uci_lightgbm.joblib` | `models/phase17/synthetic/phase17_synthetic_lightgbm.joblib` |
| File exists | Yes | Yes |
| Size | 414,024 bytes | 417,759 bytes |
| Loadable | Yes | Yes |
| Model type | LGBMRegressor | LGBMRegressor |
| n_estimators | 150 | 150 |
| Source dataset | UCI | SYNTHETIC |
| Forecast horizon | 8 weeks | 8 weeks |
| Training data | `data/phase17/features/weekly_features.parquet` (UCI rows) | `data/phase17/features/weekly_features.parquet` (SYNTHETIC rows) |
| Feature contract | 36+ weekly lag/rolling/calendar columns | 36+ weekly lag/rolling/calendar columns |
| SHA-256 | See `docs/phase18_candidate_hashes.json` | See `docs/phase18_candidate_hashes.json` |

## Candidate vs Production Registry

The candidate hashes are stored in `docs/phase18_candidate_hashes.json`.  
They have **not** been added to `docs/final_model_registry.json`.  
Production registry remains unchanged.

## Production Model Integrity

All 12 production model SHA-256 hashes verified against `docs/final_model_registry.json`.  
Snapshot saved: `docs/phase18_production_hash_snapshot.json`.  
**Result: 12/12 PASS — UNCHANGED.**
