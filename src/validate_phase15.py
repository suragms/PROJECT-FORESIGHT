"""
Phase 15 validation suite — executive intelligence / BI / project completion.

Run: python src/validate_phase15.py

Nested Phase 12–14 regression is calculated at runtime. Results are never hardcoded.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.bi.exports import EXPORT_FILES, build_bi_exports, load_bi_tables, schema_issues  # noqa: E402
from src.bi.kpis import executive_kpi_row, inventory_kpis, load_risk  # noqa: E402
from src.bi.quality import freshness, quality_scorecard  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402
from src.forecasting.registry import load_registry, verify_hash  # noqa: E402
from src.production.simulation import EXPECTED_SYN_HASH, EXPECTED_UCI_HASH  # noqa: E402
from src.validate_phase14 import run_validation as run_phase14  # noqa: E402

BOARD = [
    "Phase 12 Regression",
    "Phase 13 Regression",
    "Phase 14 Regression",
    "KPI Layer",
    "BI Exports",
    "Risk Summary",
    "Recommendation Layer",
    "Dashboard Data",
    "Power BI Export Schema",
    "Data Quality",
    "Final Documentation",
    "Model Hash Integrity",
]

REQUIRED_DOCS = [
    "docs/phase15_metadata.json",
    "docs/phase15_business_intelligence_report.md",
    "docs/phase15_executive_summary.md",
    "docs/phase15_final_system_report.md",
    "docs/phase15_known_limitations.md",
    "docs/powerbi_data_model.md",
]


def _flag(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _line(name: str, passed: bool, width: int = 28) -> str:
    return f"{name.ljust(width)}{_flag(passed)}"


def _parse_summary(text: str) -> tuple[int, int]:
    token = str(text).split()[0]
    a, b = token.split("/")
    return int(a), int(b)


def write_metadata(rows: list[dict], extra: dict) -> None:
    passed = sum(1 for r in rows if r["passed"])
    total = len(rows)
    hashes = extra.get("hashes") or {}
    kpis = extra.get("kpis") or {}
    payload = {
        "phase": 15,
        "status": "COMPLETE" if passed == total else "INCOMPLETE",
        "implementation_class": "academic/reference",
        "production_deployment": False,
        "power_bi_deployment": False,
        "validation": {
            "command": "python src/validate_phase15.py",
            "passed": passed,
            "total": total,
            "summary": f"{passed}/{total} PASS",
            "note": "Counts are calculated at runtime, not hardcoded.",
        },
        "regression": {
            "phase12_validation": extra.get("phase12"),
            "phase13_validation": extra.get("phase13"),
            "phase14_validation": extra.get("phase14"),
        },
        "model_hash_status": (
            "UNCHANGED"
            if hashes.get("uci") == EXPECTED_UCI_HASH and hashes.get("synthetic") == EXPECTED_SYN_HASH
            else "CHANGED"
        ),
        "final_models": {
            "uci_h1": {"model_id": "uci_h1_phase8_lightgbm", "sha256": hashes.get("uci")},
            "synthetic_h1": {"model_id": "synthetic_h1_hurdle_th050", "sha256": hashes.get("synthetic")},
        },
        "kpi_layer": kpis,
        "bi_exports": extra.get("bi_files"),
        "schema_issues": extra.get("schema_issues"),
        "inventory_extract_rows": extra.get("inventory_n"),
        "recommendations_autonomous": False,
        "dashboard": "dashboard/executive_intelligence.py",
        "data_quality": extra.get("quality"),
        "freshness": extra.get("freshness"),
        "deployment_status": "NOT EXECUTED",
        "known_limitations_doc": "docs/phase15_known_limitations.md",
    }
    path = PROJECT_ROOT / "docs" / "phase15_metadata.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def check_kpi_layer() -> tuple[bool, dict]:
    row = executive_kpi_row()
    risk = load_risk()
    inv = inventory_kpis(risk)
    ok = (
        row.get("decision_support_only") is True
        and inv.get("n_rows", 0) > 0
        and row.get("forecast_mae") not in (None, "NOT AVAILABLE")
        and row.get("forecast_interval_coverage") == "NOT AVAILABLE"
        and "1000-row" in str(inv.get("extract_note", ""))
    )
    return ok, {"executive": row, "inventory": inv}


def check_docs() -> tuple[bool, list[str]]:
    missing = []
    for rel in REQUIRED_DOCS:
        if not (PROJECT_ROOT / rel).exists():
            missing.append(rel)
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    if "Phase 15" not in readme:
        missing.append("README.md Phase 15 section")
    limits = PROJECT_ROOT / "docs" / "phase15_known_limitations.md"
    if limits.exists():
        text = limits.read_text(encoding="utf-8")
        for needle in ("1000-row", "Power BI", "TLS", "identity", "secrets", "autoscaling", "retraining"):
            if needle.lower() not in text.lower() and needle not in text:
                missing.append(f"limitations missing: {needle}")
    return len(missing) == 0, missing


def run_validation() -> tuple[list[dict], dict]:
    print("=" * 50)
    print("FORESIGHT PHASE 15 VALIDATION")
    print("=" * 50)
    print("\n-- nested Phase 14 (includes Phase 12 and Phase 13) --")
    p14_rows, p14_extra = run_phase14()
    p12_text = p14_extra.get("phase12")
    p13_text = p14_extra.get("phase13")
    p12_pass, p12_total = _parse_summary(p12_text)
    p13_pass, p13_total = _parse_summary(p13_text)
    p14_pass = sum(1 for r in p14_rows if r["passed"])
    p14_total = len(p14_rows)

    print("\n-- Phase 15 BI layer --")
    bi_files = build_bi_exports()
    schema_probs = schema_issues()
    tables = load_bi_tables()
    recs = tables["recommendations.parquet"]
    demand = tables["product_demand.parquet"]
    health = tables["system_health.parquet"].iloc[0]
    kpis_ok, kpi_payload = check_kpi_layer()
    q = quality_scorecard()
    fresh = freshness()
    recs_ok = (
        not recs.empty
        and recs["autonomous_decision"].eq(False).all()
        and recs[["evidence", "reason", "recommended_review", "confidence_limitation"]].notna().all().all()
        and not recs["recommended_review"].astype(str).str.contains("purchase order", case=False).any()
    )
    dash_ok = (
        (PROJECT_ROOT / "dashboard" / "executive_intelligence.py").exists()
        and set(EXPORT_FILES).issubset(tables)
        and "LOW DEMAND" in demand["demand_rank_label"].astype(str).unique()
        and "BAD PRODUCT" not in demand["demand_rank_label"].astype(str).unique()
        and bool(health["live_data"]) is False
    )
    quality_ok = (
        q.get("no_invented_quality_score") is True
        and q.get("schema_validity") == "PASS"
        and q.get("missing_required_columns") == []
        and "not live" in str(fresh.get("label", ""))
    )
    recs_hash = {r["model_id"]: verify_hash(r) for r in load_registry()}
    hashes = {
        "uci": recs_hash.get("uci_h1_phase8_lightgbm"),
        "synthetic": recs_hash.get("synthetic_h1_hurdle_th050"),
    }
    hash_ok = hashes["uci"] == EXPECTED_UCI_HASH and hashes["synthetic"] == EXPECTED_SYN_HASH
    docs_ok, docs_missing = check_docs()
    inv = tables["inventory_risk.parquet"]

    flags = {
        "Phase 12 Regression": p12_pass == p12_total,
        "Phase 13 Regression": p13_pass == p13_total,
        "Phase 14 Regression": p14_pass == p14_total,
        "KPI Layer": kpis_ok,
        "BI Exports": all((PROJECT_ROOT / "outputs" / "bi" / name).exists() for name in EXPORT_FILES),
        "Risk Summary": len(inv) == 1000 and "extract_note" in inv.columns,
        "Recommendation Layer": recs_ok,
        "Dashboard Data": dash_ok,
        "Power BI Export Schema": schema_probs == [],
        "Data Quality": quality_ok,
        "Final Documentation": docs_ok,
        "Model Hash Integrity": hash_ok,
    }

    rows = []
    print()
    print("=" * 50)
    print("FORESIGHT PHASE 15 VALIDATION")
    print("=" * 50)
    print()
    for name in BOARD:
        passed = bool(flags.get(name))
        rows.append({"name": name, "passed": passed})
        print(_line(name, passed))
        if name == "Final Documentation" and docs_missing:
            print("    missing:", ", ".join(docs_missing))
        if name == "Power BI Export Schema" and schema_probs:
            print("    issues:", schema_probs)
    passed = sum(1 for r in rows if r["passed"])
    total = len(rows)
    print()
    print(f"TOTAL: {passed}/{total} PASS")
    print("=" * 50)
    extra = {
        "phase12": p12_text,
        "phase13": p13_text,
        "phase14": f"{p14_pass}/{p14_total} PASS",
        "hashes": hashes,
        "kpis": kpi_payload["inventory"],
        "bi_files": bi_files,
        "schema_issues": schema_probs,
        "inventory_n": int(len(inv)),
        "quality": q,
        "freshness": fresh,
        "docs_missing": docs_missing,
        "phase14_extra": {
            "phase12": p12_text,
            "phase13": p13_text,
            "docker": (p14_extra.get("docker") or {}).get("status"),
        },
    }
    write_metadata(rows, extra)
    return rows, extra


if __name__ == "__main__":
    try:
        rows, extra = run_validation()
        failed = sum(1 for r in rows if not r["passed"])
        sys.exit(0 if failed == 0 else 1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
