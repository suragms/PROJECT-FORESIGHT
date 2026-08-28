"""Phase 21 — Production Monitoring Pipeline."""

import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.phase21_monitoring import run_phase21_monitoring


def run_phase21():
    print("\n" + "=" * 60)
    print("PHASE 21 — PRODUCTION MONITORING, DRIFT & INTEGRITY")
    print("=" * 60 + "\n")
    summary = run_phase21_monitoring()
    return summary


if __name__ == "__main__":
    run_phase21()
