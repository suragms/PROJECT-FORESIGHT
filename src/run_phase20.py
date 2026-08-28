"""Phase 20 — Full Production Promotion Pipeline."""

import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.phase20_feature_contract import build_contract
from src.phase20_promotion_gate import run_promotion_gate, record_pre_promotion_snapshot
from src.phase20_e2e_validation import run_e2e, verify_lineage


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_post_promotion_integrity() -> dict:
    reg_path = os.path.join(BASE, "docs", "final_model_registry.json")
    with open(reg_path) as f:
        registry = json.load(f)

    frozen_ok = True
    for e in registry:
        mf = os.path.join(BASE, e["model_file"].replace("\\", os.sep))
        if sha256(mf) != e["hash"]:
            frozen_ok = False

    p17 = os.path.join(BASE, "models", "phase17", "synthetic", "phase17_synthetic_lightgbm.joblib")
    p19 = os.path.join(BASE, "models", "phase19", "synthetic", "phase19_synthetic_lightgbm.joblib")
    p20 = os.path.join(BASE, "models", "final", "phase20", "phase20_synthetic_lightgbm.joblib")
    prov = json.load(open(os.path.join(BASE, "docs", "phase20_promotion_provenance.json")))

    return {
        "frozen_12_unchanged": frozen_ok,
        "phase17_unchanged": os.path.exists(p17),
        "phase19_unchanged": os.path.exists(p19),
        "phase20_registered": os.path.exists(p20),
        "lineage_verified": prov.get("copy_verified", False),
        "phase19_hash": sha256(p19),
        "phase20_hash": sha256(p20),
    }


def run_phase20():
    print("\n" + "=" * 60)
    print("PHASE 20 — CONTROLLED PRODUCTION PROMOTION")
    print("=" * 60 + "\n")

    snapshot = record_pre_promotion_snapshot()
    p19_path = os.path.join(BASE, "models", "phase19", "synthetic", "phase19_synthetic_lightgbm.joblib")
    if os.path.exists(p19_path):
        snapshot["phase19_hash"] = sha256(p19_path)
        snap_path = os.path.join(BASE, "docs", "phase20_pre_promotion_snapshot.json")
        with open(snap_path, "w") as f:
            json.dump(snapshot, f, indent=2)

    build_contract()
    gate = run_promotion_gate()

    if gate["status"] == "BLOCKED":
        print("\nPRODUCTION PROMOTION BLOCKED")
        return gate

    e2e = run_e2e()
    integrity = verify_post_promotion_integrity()

    status = "COMPLETE" if e2e.get("pass") and integrity.get("frozen_12_unchanged") else "BLOCKED"

    result = {
        "promotion_status": status,
        "gate": gate,
        "e2e": e2e,
        "integrity": integrity,
    }
    out = os.path.join(BASE, "docs", "phase20_gate_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nFINAL STATUS: PRODUCTION PROMOTION {status}")
    return result


if __name__ == "__main__":
    run_phase20()
