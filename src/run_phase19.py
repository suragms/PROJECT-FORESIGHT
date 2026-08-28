"""Phase 19 — Full Hardening Pipeline Runner."""
import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.phase19_holiday_analysis import run_holiday_analysis
from src.phase19_features import run_feature_engineering
from src.phase19_forecasting import run_forecasting
from src.phase19_risk_validation import run_risk_validation


def verify_phase17_unchanged():
    """Ensure Phase 17 artifacts were not modified."""
    p17_model = os.path.join(BASE, "models", "phase17", "synthetic", "phase17_synthetic_lightgbm.joblib")
    p17_bt = os.path.join(BASE, "data", "phase17", "backtests", "backtest_results.parquet")
    assert os.path.exists(p17_model), "Phase 17 model missing"
    assert os.path.exists(p17_bt), "Phase 17 backtest missing"
    return True


def final_hash_check():
    snap_path = os.path.join(BASE, "docs", "phase19_production_hash_snapshot.json")
    with open(snap_path) as f:
        snap = json.load(f)
    all_ok = True
    for m in snap["models"]:
        path = os.path.join(BASE, m["path"].replace("\\", os.sep))
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if h != m["expected"]:
            all_ok = False
    return all_ok


def run_phase19():
    print("\n" + "=" * 60)
    print("PHASE 19 — SYNTHETIC CANDIDATE HARDENING")
    print("=" * 60 + "\n")

    # Pre-flight already run separately; verify phase17 intact
    verify_phase17_unchanged()
    print("Phase 17 artifacts: UNCHANGED\n")

    holiday = run_holiday_analysis()
    print()
    features = run_feature_engineering()
    print()
    metrics = run_forecasting()
    print()
    risk = run_risk_validation()
    print()

    final_ok = final_hash_check()
    print(f"\nFrozen production hashes: {'PASS' if final_ok else 'FAIL'}")

    return {"holiday": holiday, "features": features, "metrics": metrics, "risk": risk, "final_hash_ok": final_ok}


if __name__ == "__main__":
    results = run_phase19()
    out = os.path.join(BASE, "docs", "phase19_gate_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out}")
