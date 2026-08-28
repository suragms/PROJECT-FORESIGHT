# Phase 21 — Alerting Policy

## Principles

1. **Observe, do not act** — alerts inform human review; no automatic retraining or model replacement.
2. **Evidence-based** — every alert includes component, severity, message, evidence, and recommended action.
3. **Structured** — alerts stored in `data/phase21/monitoring/alerts.json` and embedded in monitoring summary.

## Alert Schema

```json
{
  "alert_id": "uuid-prefix",
  "timestamp": "ISO-8601",
  "component": "data_quality|feature_quality|data_drift|...",
  "severity": "INFO|WARNING|CRITICAL",
  "message": "Human-readable summary",
  "evidence": {},
  "recommended_action": "Review ..."
}
```

## Severity Levels

| Level | When Used |
|-------|-----------|
| INFO | Informational drift or monitoring notes |
| WARNING | Degraded quality, drift watch, elevated holiday WAPE |
| CRITICAL | FAIL status, model integrity alert, missing critical features |

## Recommended Actions (Examples)

- Review data ingestion pipeline
- Review feature pipeline
- Investigate drift in input features
- Investigate forecast distribution shift
- Review forecast performance vs baseline
- Investigate inventory inputs and risk logic
- **Investigate model integrity immediately** (hash mismatch)

## MODEL INTEGRITY ALERT

Triggered when SHA-256 of frozen models or Phase 20 production model does not match registry. **Do not repair automatically.**

## Non-Actions

Phase 21 will never:

- Retrain models
- Replace production models
- Modify `models/final/` artifacts
- Overwrite production forecasts
