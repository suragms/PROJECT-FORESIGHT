"""
Phase 14 validation suite — end-to-end production simulation.

Run: python src/validate_phase14.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import PROJECT_ROOT  # noqa: E402
from src.production.docker_check import run_docker_validation  # noqa: E402
from src.production.simulation import EXPECTED_SYN_HASH, EXPECTED_UCI_HASH, run_simulation  # noqa: E402
from src.validate_phase12 import run_validation as run_phase12  # noqa: E402
from src.validate_phase13 import run_validation as run_phase13  # noqa: E402

BOARD = [
    "Phase 12 Regression",
    "Phase 13 Regression",
    "API Health",
    "API Readiness",
    "Authentication",
    "Rate Limiting",
    "Single Forecast",
    "Batch Forecast",
    "Model Integrity",
    "Data Contract",
    "Inventory Risk",
    "Business Workflow",
    "Dashboard",
    "Failure Recovery",
    "Audit Logging",
    "Monitoring",
    "Reproducibility",
    "Performance",
    "Docker",
]


def _flag(passed: bool, status: str | None = None) -> str:
    if status in {"PARTIAL", "NOT IMPLEMENTED"}:
        return status
    return "PASS" if passed else "FAIL"


def _line(name: str, passed: bool, width: int = 26, status: str | None = None) -> str:
    return f"{name.ljust(width)}{_flag(passed, status)}"


def write_metadata(rows: list[dict], extra: dict) -> None:
    sim = extra.get("simulation") or {}
    evidence = sim.get("evidence") or {}
    docker = extra.get("docker") or {}
    by_name = {r["name"]: r for r in rows}
    passed = sum(1 for r in rows if r["passed"])
    total = len(rows)
    sf = evidence.get("single_forecast") or {}
    batch = evidence.get("batch") or {}
    runs = {r["n"]: r for r in batch.get("runs") or []}
    hashes = evidence.get("hashes") or {}
    perf = evidence.get("performance") or {}
    payload = {
        "phase": 14,
        "status": "COMPLETE" if passed == total else "INCOMPLETE",
        "implementation_class": "academic/reference",
        "production_deployment": False,
        "validation": {
            "command": "python src/validate_phase14.py",
            "passed": passed,
            "total": total,
            "summary": f"{passed}/{total} PASS",
            "note": "Counts are calculated at runtime, not hardcoded.",
        },
        "regression": {
            "phase12_validation": extra.get("phase12"),
            "phase13_validation": extra.get("phase13"),
        },
        "model_hash_status": (
            "UNCHANGED"
            if hashes.get("uci_h1") == EXPECTED_UCI_HASH
            and hashes.get("synthetic_h1") == EXPECTED_SYN_HASH
            and hashes.get("used_for_inference") == EXPECTED_UCI_HASH
            else "CHANGED"
        ),
        "final_models": {
            "uci_h1": {
                "model_id": "uci_h1_phase8_lightgbm",
                "sha256": hashes.get("uci_h1"),
            },
            "synthetic_h1": {
                "model_id": "synthetic_h1_hurdle_th050",
                "sha256": hashes.get("synthetic_h1"),
            },
            "used_for_simulation_inference": hashes.get("used_for_inference"),
        },
        "api_status": "PASS" if by_name.get("API Health", {}).get("passed") and by_name.get("API Readiness", {}).get("passed") else "FAIL",
        "security_status": "PASS" if by_name.get("Authentication", {}).get("passed") and by_name.get("Rate Limiting", {}).get("passed") else "FAIL",
        "inventory_risk_status": evidence.get("inventory", {}).get("join", "NOT AVAILABLE"),
        "business_validation_status": "PASS" if by_name.get("Business Workflow", {}).get("passed") else "FAIL",
        "dashboard_status": "PASS" if by_name.get("Dashboard", {}).get("passed") else "FAIL",
        "monitoring_status": "PASS" if by_name.get("Monitoring", {}).get("passed") else "FAIL",
        "single_forecast": sf,
        "batch": {
            "batch10_s": (runs.get(10) or {}).get("latency_s"),
            "batch10_rows_per_s": (runs.get(10) or {}).get("rows_per_s"),
            "batch10_response_bytes": (runs.get(10) or {}).get("response_bytes"),
            "batch100_s": (runs.get(100) or {}).get("latency_s"),
            "batch100_rows_per_s": (runs.get(100) or {}).get("rows_per_s"),
            "batch100_response_bytes": (runs.get(100) or {}).get("response_bytes"),
            "batch500_s": (runs.get(500) or {}).get("latency_s"),
            "batch500_rows_per_s": (runs.get(500) or {}).get("rows_per_s"),
            "batch500_response_bytes": (runs.get(500) or {}).get("response_bytes"),
            "error_rate": 0.0 if by_name.get("Batch Forecast", {}).get("passed") else None,
        },
        "performance": perf,
        "docker_status": docker.get("status"),
        "docker": {
            "static": (docker.get("static") or {}).get("status"),
            "runtime": (docker.get("runtime") or {}).get("status"),
            "detail": docker.get("detail"),
        },
        "deployment_status": "NOT EXECUTED",
        "known_limitations_doc": "docs/phase14_known_limitations.md",
        "evidence": "outputs/phase14/simulation.json",
    }
    path = PROJECT_ROOT / "docs" / "phase14_metadata.json"
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    if existing.get("pytest"):
        payload["pytest"] = existing["pytest"]
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_validation() -> tuple[list[dict], dict]:
    print("=" * 50)
    print("FORESIGHT PHASE 14 VALIDATION")
    print("=" * 50)
    print("\n-- nested Phase 12 --")
    p12, _ = run_phase12()
    print("\n-- nested Phase 13 --")
    p13, _ = run_phase13()
    print("\n-- Phase 14 simulation --")
    sim = run_simulation()
    print("\n-- Docker --")
    docker = run_docker_validation()
    by_name = {row["name"]: row["passed"] for row in sim["results"]}
    by_name["Phase 12 Regression"] = p12.failed == 0
    by_name["Phase 13 Regression"] = p13.failed == 0
    by_name["Docker"] = bool(docker.get("passed"))
    status_overrides = {"Docker": docker.get("status")}
    rows = []
    print()
    print("=" * 50)
    print("FORESIGHT PHASE 14 VALIDATION")
    print("=" * 50)
    print()
    for name in BOARD:
        passed = bool(by_name.get(name))
        rows.append({"name": name, "passed": passed, "status": status_overrides.get(name)})
        print(_line(name, passed, status=status_overrides.get(name)))
    extra_rows = [row for row in sim["results"] if row["name"] not in BOARD]
    print()
    for row in extra_rows:
        print(_line(row["name"], row["passed"]))
    passed = sum(1 for r in rows if r["passed"])
    total = len(rows)
    print()
    print(f"TOTAL: {passed}/{total} PASS")
    print("=" * 50)
    extra_payload = {
        "phase12": p12.summary(),
        "phase13": p13.summary(),
        "simulation": sim,
        "docker": docker,
    }
    write_metadata(rows, extra_payload)
    return rows, extra_payload


if __name__ == "__main__":
    try:
        rows, extra = run_validation()
        failed = sum(1 for r in rows if not r["passed"])
        sys.exit(0 if failed == 0 else 1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
