# Project Foresight — Documentation Consistency Audit

**Audit date:** 2026-08-29  
**Method:** Cross-check docs vs runtime (pytest, hash verification, dataset inventory, HTTP probes)

---

## Consistent claims (verified)

| Claim | Documentation | Runtime evidence | Status |
|-------|---------------|------------------|--------|
| Production model `phase20_synthetic_lightgbm` | Phase 20/22 docs, README | Hash + adapter + tests | **CONSISTENT** |
| 6-week validated horizon | Phase 20 registry, dashboards | e2e results horizons 1–6 | **CONSISTENT** |
| WAPE 13.96% / h1–h6 11.03% | README, model card | Phase 20 registry + monitoring reports | **CONSISTENT** |
| Live performance PENDING ACTUALS | README, dashboards, monitoring | UI labels + monitoring JSON | **CONSISTENT** |
| 12 frozen models unchanged | Registries, Phase 21/22 | 12/12 hash PASS | **CONSISTENT** |
| Phase 21 observability-only | Phase 21 docs | No retrain in monitoring runners | **CONSISTENT** |
| No demo credentials in SPA | Phase 23 docs | Grep + `test_public_frontend_has_no_demo_credentials` | **CONSISTENT** |
| Test suite | Phase 22 audit | **286/286 PASS** (this audit) | **CONSISTENT** |

---

## Discrepancies found

| Issue | Documentation says | Actual evidence | Classification |
|-------|-------------------|-----------------|----------------|
| Phase 12 validation | Some docs: **42/42 PASS** | Historical **41/42** due to legacy `lightgbm_forecaster.joblib` | **DOCUMENTATION MISMATCH** — later docs correctly note 41/42 |
| Synthetic dataset source | README describes "Multi-Store Relational" dataset | Local generator (`generate_synthetic_retail.py`), **not** Kaggle 10M | **DOCUMENTATION MISMATCH** — Phase 16 report documents this; README should clarify Kaggle 10M not used |
| Kaggle extraction | Implied official datasets | UCI CSV present; Kaggle ZIPs absent; synthetic 10M **MANUAL DOWNLOAD REQUIRED** | **EXPECTED HISTORICAL DIFFERENCE** / incomplete external download |
| Phase 22 delivery status | `PROJECT DELIVERY READY` | Local audit PASS; Render `/health` **timeout** | **ENVIRONMENT ISSUE** — live backend not verified this run |
| Phase 17 feature audit count | Brief mentions "46-column features" | `weekly_features.parquet` has **46 columns**; leakage audit lists **36 feature entries** | **CONSISTENT** — audit covers engineered features, not all raw columns |
| Phase 19 feature count | Docs reference holiday features | **56 columns** in `phase19/features/synthetic_weekly_features.parquet` | **CONSISTENT** with hardening phase |

---

## Historical vs current runtime

| Metric | Historical doc | Current runtime (2026-08-29) | Notes |
|--------|----------------|------------------------------|-------|
| pytest total | Various phase reports cite subset counts | **286 passed** | Full suite green |
| Phase 22 audit | PASS | Re-run **PASS** | `python src/phase22_final_audit.py` |
| Frozen model hashes | Phase 18 snapshot | All match | No drift |
| Git dirty state | Phase 22 snapshot lists modified monitoring JSON | Still modified (expected from monitoring runs) | **EXPECTED HISTORICAL DIFFERENCE** — monitoring outputs regenerate |

---

## Recommendations (documentation only — not applied in this audit)

1. README § Datasets: explicitly state Kaggle 10M synthetic was **not** downloaded; local generator used.
2. Phase 12 references: standardize on **41/42 PASS** with legacy artifact footnote everywhere.
3. Deployment docs: note Render cold-start may cause intermittent health-check timeouts.

---

## Overall documentation consistency

**Status: PARTIAL** — core production/validation claims match runtime; dataset sourcing and historical Phase 12 counts need clearer alignment in a few older docs.
