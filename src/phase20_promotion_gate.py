"""
Phase 20 — Promotion Gate & Pre-Promotion Snapshot
===================================================
Validates eligibility and promotes Phase 19 candidate via COPY (never overwrite).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs")
MODELS_FIN = os.path.join(BASE, "models", "final")
MODELS19 = os.path.join(BASE, "models", "phase19", "synthetic")
MODELS20 = os.path.join(MODELS_FIN, "phase20")
P19_METRICS = os.path.join(BASE, "data", "phase19", "forecasts", "backtest_metrics.json")
P19_RISK = os.path.join(BASE, "data", "phase19", "risk", "risk_summary.json")
P19_FEAT_AUDIT = os.path.join(BASE, "data", "phase19", "features", "leakage_audit.json")
SOURCE_ARTIFACT = os.path.join(MODELS19, "phase19_synthetic_lightgbm.joblib")
PROMOTED_ARTIFACT = os.path.join(MODELS20, "phase20_synthetic_lightgbm.joblib")

SUPPORTED_HORIZON_WEEKS = 6


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return None


def record_pre_promotion_snapshot() -> dict:
    reg_path = os.path.join(DOCS, "final_model_registry.json")
    with open(reg_path) as f:
        registry = json.load(f)

    models = []
    all_match = True
    for e in registry:
        mf = os.path.join(BASE, e["model_file"].replace("\\", os.sep))
        actual = sha256(mf)
        ok = actual == e["hash"]
        if not ok:
            all_match = False
        models.append({
            "model_id": e["model_id"],
            "path": e["model_file"],
            "expected_hash": e["hash"],
            "actual_hash": actual,
            "size_bytes": os.path.getsize(mf),
            "match": ok,
            "status": e.get("status"),
        })

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "frozen_models_count": len(models),
        "all_frozen_match": all_match,
        "models": models,
    }
    out = os.path.join(DOCS, "phase20_pre_promotion_snapshot.json")
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot


def check_eligibility() -> dict:
    """Return eligibility result with pass/fail per criterion."""
    checks = {}

    checks["candidate_exists"] = os.path.exists(SOURCE_ARTIFACT)
    if checks["candidate_exists"]:
        checks["candidate_sha256"] = sha256(SOURCE_ARTIFACT)
        checks["candidate_size"] = os.path.getsize(SOURCE_ARTIFACT)
        try:
            import joblib
            m = joblib.load(SOURCE_ARTIFACT)
            checks["candidate_loadable"] = m is not None and hasattr(m, "predict")
        except Exception as ex:
            checks["candidate_loadable"] = False
            checks["load_error"] = str(ex)
    else:
        checks["candidate_loadable"] = False

    metrics = json.load(open(P19_METRICS)) if os.path.exists(P19_METRICS) else {}
    checks["phase19_wape_documented"] = metrics.get("phase19_wape_pct") is not None
    checks["beats_seasonal_naive"] = (
        metrics.get("phase19_wape_pct", 999) < metrics.get("seasonal_naive_wape_pct", 0)
    )
    checks["rolling_origin_pass"] = len(metrics.get("fold_metrics", [])) >= 5
    checks["all_folds_beat_baseline"] = all(
        f.get("p19_beats_baseline", False) for f in metrics.get("fold_metrics", [])
    ) if metrics.get("fold_metrics") else False

    audit = json.load(open(P19_FEAT_AUDIT)) if os.path.exists(P19_FEAT_AUDIT) else []
    checks["leakage_pass"] = all(a.get("leakage_status") != "FAIL" for a in audit)

    risk = json.load(open(P19_RISK)) if os.path.exists(P19_RISK) else {}
    checks["risk_stress_pass"] = risk.get("stress_tests_pass", False)

    snap = record_pre_promotion_snapshot()
    checks["frozen_intact"] = snap["all_frozen_match"]
    checks["horizon_six_weeks"] = metrics.get("supported_horizon_weeks") == SUPPORTED_HORIZON_WEEKS

    critical = [
        "candidate_exists", "candidate_loadable", "phase19_wape_documented",
        "beats_seasonal_naive", "rolling_origin_pass", "leakage_pass",
        "frozen_intact", "horizon_six_weeks", "risk_stress_pass",
    ]
    checks["promotion_eligible"] = all(checks.get(k) for k in critical)
    return checks


def promote_artifact() -> dict:
    """Copy Phase 19 candidate to promoted location. Never overwrite original 12 models."""
    os.makedirs(MODELS20, exist_ok=True)
    if not os.path.exists(SOURCE_ARTIFACT):
        raise FileNotFoundError(f"Source candidate not found: {SOURCE_ARTIFACT}")

    source_hash = sha256(SOURCE_ARTIFACT)
    shutil.copy2(SOURCE_ARTIFACT, PROMOTED_ARTIFACT)
    promoted_hash = sha256(PROMOTED_ARTIFACT)

    metrics = json.load(open(P19_METRICS))
    provenance = {
        "source_artifact": SOURCE_ARTIFACT.replace("\\", "/"),
        "source_sha256": source_hash,
        "promoted_artifact": PROMOTED_ARTIFACT.replace("\\", "/"),
        "promoted_sha256": promoted_hash,
        "copy_verified": source_hash == promoted_hash,
        "phase19_wape_pct": metrics.get("phase19_wape_pct"),
        "supported_horizon_wape_pct": metrics.get("supported_horizon_wape_pct"),
        "promotion_timestamp": datetime.now(timezone.utc).isoformat(),
        "promotion_phase": 20,
    }
    prov_path = os.path.join(DOCS, "phase20_promotion_provenance.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    return provenance


def register_production_model(provenance: dict) -> dict:
    """Add Phase 20 entry to separate production registry extension."""
    entry = {
        "model_id": "phase20_synthetic_lightgbm",
        "status": "production",
        "source_dataset": "SYNTHETIC",
        "parent_candidate": "phase19_synthetic_lightgbm",
        "forecast_grain": "weekly_sku",
        "supported_horizon_weeks": SUPPORTED_HORIZON_WEEKS,
        "extended_horizon": {
            "supported": True,
            "weeks": [7, 8],
            "accuracy_status": "PARTIAL",
        },
        "wape": provenance.get("phase19_wape_pct"),
        "supported_horizon_wape": provenance.get("supported_horizon_wape_pct"),
        "validation": "rolling_origin",
        "promotion_phase": 20,
        "model_file": "models/final/phase20/phase20_synthetic_lightgbm.joblib",
        "hash": provenance.get("promoted_sha256"),
        "limitations": [
            "Holiday bias in Nov-Dec not fully eliminated",
            "Horizon 7-8 has partial accuracy",
        ],
    }
    reg_path = os.path.join(DOCS, "phase20_production_registry.json")
    with open(reg_path, "w") as f:
        json.dump([entry], f, indent=2)
    return entry


def run_promotion_gate() -> dict:
    print("=" * 60)
    print("PHASE 20 — PROMOTION GATE")
    print("=" * 60)

    snapshot = record_pre_promotion_snapshot()
    print(f"Pre-promotion snapshot: {len(snapshot['models'])} models, all_match={snapshot['all_frozen_match']}")

    eligibility = check_eligibility()
    for k, v in eligibility.items():
        if k not in ("candidate_sha256", "candidate_size", "load_error", "promotion_eligible"):
            print(f"  {k}: {v}")

    if not eligibility["promotion_eligible"]:
        print("\nPROMOTION BLOCKED — eligibility check failed")
        return {"status": "BLOCKED", "eligibility": eligibility, "snapshot": snapshot}

    provenance = promote_artifact()
    registry_entry = register_production_model(provenance)
    print(f"\nPromoted: {PROMOTED_ARTIFACT}")
    print(f"SHA-256: {provenance['promoted_sha256'][:20]}...")
    print("PROMOTION ELIGIBLE — artifact copied")

    return {
        "status": "ELIGIBLE",
        "eligibility": eligibility,
        "snapshot": snapshot,
        "provenance": provenance,
        "registry_entry": registry_entry,
    }


if __name__ == "__main__":
    result = run_promotion_gate()
    out = os.path.join(DOCS, "phase20_gate_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
