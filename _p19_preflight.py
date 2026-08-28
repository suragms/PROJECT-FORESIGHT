"""Phase 19 pre-flight: verify frozen models and create directory structure."""
import hashlib
import json
import os
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(BASE, "docs")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    reg_path = os.path.join(DOCS, "final_model_registry.json")
    p18_path = os.path.join(DOCS, "phase18_production_hash_snapshot.json")
    with open(reg_path) as f:
        registry = json.load(f)
    p18 = json.load(open(p18_path)) if os.path.exists(p18_path) else None

    snapshot = {"timestamp": datetime.now(timezone.utc).isoformat(), "models": [], "all_match": True}
    for e in registry:
        mf = os.path.join(BASE, e["model_file"].replace("\\", os.sep))
        actual = sha256(mf)
        ok = actual == e["hash"]
        if not ok:
            snapshot["all_match"] = False
        snapshot["models"].append({
            "model_id": e["model_id"], "path": e["model_file"],
            "expected": e["hash"], "actual": actual, "match": ok,
        })
        print(f"{'PASS' if ok else 'FAIL'}: {e['model_id']}")

    out = os.path.join(DOCS, "phase19_production_hash_snapshot.json")
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=2)
    if not snapshot["all_match"]:
        raise RuntimeError("Production hash mismatch — STOP")

    dirs = [
        "data/phase19/features", "data/phase19/backtests", "data/phase19/forecasts",
        "data/phase19/risk", "data/phase19/diagnostics", "data/phase19/promotion",
        "models/phase19/synthetic", "models/phase19/experiments", "docs/phase19",
    ]
    for d in dirs:
        os.makedirs(os.path.join(BASE, d), exist_ok=True)
        print(f"Created: {d}")
    print(f"\nSnapshot: {out}")
    return snapshot


if __name__ == "__main__":
    main()
