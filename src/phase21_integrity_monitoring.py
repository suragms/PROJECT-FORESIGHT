"""Phase 21 — Production integrity baseline and ongoing hash monitoring."""

from __future__ import annotations

import json
import os

from src.phase21_common import (
    BASE, DOCS, P20_PROV, P20_REG, save_json, sha256, now_iso,
)

MODELS_FIN = os.path.join(BASE, "models", "final")
P20_MODEL = os.path.join(MODELS_FIN, "phase20", "phase20_synthetic_lightgbm.joblib")
BASELINE_PATH = os.path.join(DOCS, "phase21_production_integrity_baseline.json")


def record_integrity_baseline() -> dict:
    reg_orig = json.load(open(os.path.join(DOCS, "final_model_registry.json")))
    p20_reg = json.load(open(P20_REG)) if os.path.exists(P20_REG) else []
    prov = json.load(open(P20_PROV)) if os.path.exists(P20_PROV) else {}

    models = []
    all_match = True
    for e in reg_orig:
        mf = os.path.join(BASE, e["model_file"].replace("\\", os.sep))
        actual = sha256(mf)
        ok = actual == e["hash"]
        if not ok:
            all_match = False
        models.append({
            "model_id": e["model_id"], "path": e["model_file"],
            "expected_hash": e["hash"], "actual_hash": actual,
            "size_bytes": os.path.getsize(mf), "match": ok, "registry_status": e.get("status"),
        })

    p20_entry = p20_reg[0] if p20_reg else {}
    p20_hash = sha256(P20_MODEL) if os.path.exists(P20_MODEL) else None
    p20_ok = p20_hash == p20_entry.get("hash") if p20_hash else False
    if not p20_ok:
        all_match = False

    baseline = {
        "timestamp": now_iso(),
        "frozen_models_count": len(models),
        "all_frozen_match": all_match,
        "models": models,
        "phase20_production": {
            "model_id": p20_entry.get("model_id", "phase20_synthetic_lightgbm"),
            "path": "models/final/phase20/phase20_synthetic_lightgbm.joblib",
            "expected_hash": p20_entry.get("hash") or prov.get("promoted_sha256"),
            "actual_hash": p20_hash,
            "size_bytes": os.path.getsize(P20_MODEL) if os.path.exists(P20_MODEL) else None,
            "match": p20_ok,
            "registry_status": "production",
        },
        "provenance_copy_verified": prov.get("copy_verified"),
    }
    save_json(BASELINE_PATH, baseline)
    return baseline


def run_integrity_monitoring(baseline: dict | None = None) -> dict:
    if baseline is None:
        baseline = json.load(open(BASELINE_PATH)) if os.path.exists(BASELINE_PATH) else record_integrity_baseline()

    alerts = []
    frozen_ok = True
    for m in baseline.get("models", []):
        mf = os.path.join(BASE, m["path"].replace("\\", os.sep))
        if not os.path.exists(mf):
            frozen_ok = False
            alerts.append({"severity": "CRITICAL", "message": f"Missing model: {m['model_id']}"})
            continue
        actual = sha256(mf)
        if actual != m["expected_hash"]:
            frozen_ok = False
            alerts.append({
                "severity": "CRITICAL",
                "message": f"MODEL INTEGRITY ALERT: {m['model_id']} hash mismatch",
                "expected": m["expected_hash"][:16],
                "actual": actual[:16],
            })

    p20 = baseline.get("phase20_production", {})
    p20_path = os.path.join(BASE, p20.get("path", "").replace("\\", os.sep))
    p20_ok = True
    if os.path.exists(p20_path):
        actual = sha256(p20_path)
        if actual != p20.get("expected_hash"):
            p20_ok = False
            alerts.append({
                "severity": "CRITICAL",
                "message": "MODEL INTEGRITY ALERT: phase20_synthetic_lightgbm hash mismatch",
            })
    else:
        p20_ok = False
        alerts.append({"severity": "CRITICAL", "message": "Phase 20 production model missing"})

    status = "PASS" if frozen_ok and p20_ok else "FAIL"
    return {
        "timestamp": now_iso(),
        "frozen_12_unchanged": frozen_ok,
        "phase20_unchanged": p20_ok,
        "status": status,
        "alerts": alerts,
    }
