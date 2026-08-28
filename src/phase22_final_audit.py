"""Phase 22 — Final delivery audit and integrity snapshot."""

from __future__ import annotations

import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.phase21_common import DOCS, save_json, sha256, now_iso
from src.phase21_integrity_monitoring import record_integrity_baseline, run_integrity_monitoring

INTEGRITY_SNAPSHOT = os.path.join(DOCS, "phase22_integrity_snapshot.json")
AUDIT_REPORT = os.path.join(DOCS, "phase22_final_audit.json")

P20_MODEL = os.path.join(BASE, "models", "final", "phase20", "phase20_synthetic_lightgbm.joblib")
P17_MODEL = os.path.join(BASE, "models", "phase17", "synthetic", "phase17_synthetic_lightgbm.joblib")
P19_MODEL = os.path.join(BASE, "models", "phase19", "synthetic", "phase19_synthetic_lightgbm.joblib")

REQUIRED_DOCS = [
    "phase22_system_architecture.md",
    "phase22_api_documentation.md",
    "phase22_deployment_guide.md",
    "phase22_quick_start.md",
    "phase22_user_guide.md",
    "phase22_technical_documentation.md",
    "phase22_business_value.md",
    "phase22_model_card.md",
    "phase22_dataset_documentation.md",
    "PROJECT_FORESIGHT_FINAL_REPORT.md",
    "phase22_future_roadmap.md",
    "phase22_demo_script.md",
    "phase22_project_structure.md",
    "phase22_zidio_submission_checklist.md",
]

REQUIRED_FILES = [
    "dashboard/phase22_executive_dashboard.py",
    "dashboard/phase20_production.py",
    "dashboard/phase21_monitoring.py",
    "src/api/phase20_routes.py",
    "src/api/phase21_routes.py",
    "docs/phase20_feature_contract.json",
    "docs/phase20_production_registry.json",
    "data/phase21/monitoring/monitoring_summary.json",
    "tests/test_phase21_monitoring.py",
    "tests/test_phase22_final_delivery.py",
]


def _git_status() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=BASE, capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def record_integrity_snapshot() -> dict:
    baseline = record_integrity_baseline()
    integrity = run_integrity_monitoring(baseline)

    p17_exists = os.path.exists(P17_MODEL)
    p19_exists = os.path.exists(P19_MODEL)
    p17_hash = sha256(P17_MODEL) if p17_exists else None
    p19_hash = sha256(P19_MODEL) if p19_exists else None

    snapshot = {
        "timestamp": now_iso(),
        "git_status_short": _git_status(),
        "frozen_models_12_unchanged": integrity["frozen_12_unchanged"],
        "phase20_production_unchanged": integrity["phase20_unchanged"],
        "phase17_artifact_exists": p17_exists,
        "phase19_artifact_exists": p19_exists,
        "phase17_model_hash": p17_hash,
        "phase19_model_hash": p19_hash,
        "phase20_model_hash": baseline["phase20_production"]["actual_hash"],
        "integrity_status": integrity["status"],
        "baseline_reference": "docs/phase21_production_integrity_baseline.json",
    }
    save_json(INTEGRITY_SNAPSHOT, snapshot)
    return snapshot


def run_final_audit() -> dict:
    snapshot = record_integrity_snapshot()

    checks = {}
    checks["documentation"] = all(
        os.path.exists(os.path.join(DOCS, d)) for d in REQUIRED_DOCS
    )
    checks["production_model"] = os.path.exists(P20_MODEL)
    checks["phase20_contract"] = os.path.exists(os.path.join(DOCS, "phase20_feature_contract.json"))
    checks["phase21_monitoring"] = os.path.exists(
        os.path.join(BASE, "data", "phase21", "monitoring", "monitoring_summary.json")
    )
    checks["api_routes"] = all(
        os.path.exists(os.path.join(BASE, f.replace("/", os.sep))) for f in [
            "src/api/phase20_routes.py", "src/api/phase21_routes.py",
        ]
    )
    checks["dashboards"] = all(
        os.path.exists(os.path.join(BASE, f.replace("/", os.sep))) for f in [
            "dashboard/phase20_production.py",
            "dashboard/phase21_monitoring.py",
            "dashboard/phase22_executive_dashboard.py",
        ]
    )
    checks["tests"] = all(
        os.path.exists(os.path.join(BASE, f.replace("/", os.sep))) for f in [
            "tests/test_phase21_monitoring.py", "tests/test_phase22_final_delivery.py",
        ]
    )
    checks["frozen_models"] = snapshot["frozen_models_12_unchanged"]
    checks["phase20_hash"] = snapshot["phase20_production_unchanged"]
    checks["final_report"] = os.path.exists(os.path.join(DOCS, "PROJECT_FORESIGHT_FINAL_REPORT.md"))
    checks["executive_dashboard"] = os.path.exists(
        os.path.join(BASE, "dashboard", "phase22_executive_dashboard.py")
    )

    all_pass = all(checks.values())
    audit = {
        "timestamp": now_iso(),
        "checks": checks,
        "integrity_snapshot": snapshot,
        "status": "PASS" if all_pass else "FAIL",
        "delivery_status": "PROJECT DELIVERY READY" if all_pass else "PROJECT DELIVERY PARTIALLY READY",
    }
    save_json(AUDIT_REPORT, audit)
    return audit


if __name__ == "__main__":
    audit = run_final_audit()
    print(f"Phase 22 Audit: {audit['status']}")
    print(f"Delivery: {audit['delivery_status']}")
    for k, v in audit["checks"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    sys.exit(0 if audit["status"] == "PASS" else 1)
