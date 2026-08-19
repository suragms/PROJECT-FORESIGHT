# Phase 15 — Known limitations

Phase 15 does not hide operational gaps. The BI layer consumes frozen artifacts; it does not make them live.

## Cloud deployment status

**Not deployed.** No Render, Azure, AWS, or GCP production service was provisioned in this phase. Local Docker image `foresight-phase14:local` is a reference container only.

## TLS status

**Not deployed in this repository.** TLS termination belongs to a reverse proxy or platform that was not configured here.

## Identity provider status

**Not deployed.** API-key authentication (`FORESIGHT_API_AUTH_ENABLED` + `FORESIGHT_API_API_KEY`) is not an enterprise identity provider (no OIDC/SAML/SSO).

## Secrets-manager status

**Not deployed.** Secrets are expected via environment variables / `.env` (gitignored). No AWS Secrets Manager, Azure Key Vault, or GCP Secret Manager integration.

## Autoscaling status

**Not deployed.** No Kubernetes HPA, cloud autoscaling group, or multi-worker shared rate-limit store.

## Automated retraining status

**Disabled / not deployed.** Monitoring does not retrain. Frozen hashes must remain unchanged unless a human-approved Phase 11 replacement is executed.

## Inventory-data coverage

On-disk inventory intelligence is `outputs/risk_scores/inventory_risk_matrix.parquet`:

* **1000-row reference extract** (also written 1,000-row) — not the operational inventory universe
* **733** `CRITICAL / HIGH` stockout labels
* **887** replenishment-review flags (`reorder_triggered`: shelf at/below ROP) on the latest snapshot used by the risk matrix
* **0** `SEVERE OVERSTOCK`
* **1** `MODERATE OVERSTOCK`
* UCI forecast keys (for example `UCI_10135`) **do not join** this extract → **NOT AVAILABLE**, never imputed

## Monitoring snapshot limitations

`outputs/monitoring/*.json` files are **snapshots**. Captions must say “Monitoring snapshot” / “Data as of”, never “live”. In-process API counters (for example error_rate 0.4211 after Phase 14 negative tests) are not cloud APM. `n_alerts` on the last snapshot was 0.

## Power BI deployment status

**Not deployed.** `outputs/bi/*.parquet` and `docs/powerbi_data_model.md` prepare a semantic model. No Power BI workspace, gateway, or scheduled refresh was published. Tests do not require Power BI Desktop.

## Other BI limitations

* Forecast accuracy metrics use held-out TEST actuals already stored in `final_predictions.parquet`. Unknown future actuals are not scored.
* Interval coverage is **NOT AVAILABLE** (P10/P90 are companions; missing on 76.351% of rows because they are h=1 only).
* Extract growth is **Insufficient Evidence** (no independent YoY window on the extract). Series-level growth uses a documented TEST date split after a SYNTHETIC join only.
* SYNTHETIC monthly seasonality of TEST actuals is **weak** (CV 0.0429).
* Executive 2×2 uses a strict-median overlay; `ending_inventory` median is 0. It does not replace `stockout_risk_level`.
* Recommendations never generate purchase orders, supplier messages, or financial commitments.
* Dashboard filters change the view only; they do not change frozen models.
