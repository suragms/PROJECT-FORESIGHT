"""
Phase 17 — Leakage Audit
==========================
Standalone leakage verification for Phase 17 features.
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P17_FEAT = os.path.join(BASE_DIR, "data", "phase17", "features")


def run_audit():
    audit_path = os.path.join(P17_FEAT, "leakage_audit.json")
    if not os.path.exists(audit_path):
        print("ERROR: leakage_audit.json not found. Run feature engineering first.")
        return False

    with open(audit_path) as f:
        audit = json.load(f)

    fails = [a for a in audit if a.get("leakage_status") == "FAIL"]
    reviews = [a for a in audit if a.get("leakage_status") == "REVIEW"]
    passes = [a for a in audit if a.get("leakage_status") == "PASS"]

    print(f"Leakage Audit: {len(passes)} PASS, {len(fails)} FAIL, {len(reviews)} REVIEW")

    if fails:
        print("\nFAILED features (must not be used):")
        for f_entry in fails:
            print(f"  {f_entry['feature']}: {f_entry['source']}")

    if reviews:
        print("\nREVIEW features:")
        for r in reviews:
            print(f"  {r['feature']}: {r['source']}")

    return len(fails) == 0


if __name__ == "__main__":
    ok = run_audit()
    sys.exit(0 if ok else 1)
