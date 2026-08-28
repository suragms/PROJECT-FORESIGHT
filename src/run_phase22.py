"""Phase 22 — Final delivery pipeline."""

import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.phase22_final_audit import run_final_audit


def run_phase22():
    print("\n" + "=" * 60)
    print("PHASE 22 — FINAL DELIVERY AUDIT")
    print("=" * 60 + "\n")
    audit = run_final_audit()
    print(f"Audit: {audit['status']}")
    print(f"Delivery: {audit['delivery_status']}")
    for k, v in audit["checks"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return audit


if __name__ == "__main__":
    run_phase22()
