"""
Phase 13 validation suite.

Run: python src/validate_phase13.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (  # noqa: E402
    APP_VERSION,
    MODELS_FINAL_DIR,
    PHASE11_META_PATH,
    PROJECT_ROOT,
    REGISTRY_PATH,
)
from src.forecasting.registry import load_registry, verify_hash  # noqa: E402
from src.phase10_common import PHASE8_FREEZE_FILES, PHASE9_FREEZE_FILES  # noqa: E402
from src.phase10_common import hashes_unchanged, snapshot_hashes  # noqa: E402
from src.production.benchmark import run_benchmark  # noqa: E402
from src.production.business_validation import write_business_validation_report  # noqa: E402
from src.production.config_validation import validate_runtime_config  # noqa: E402


class ValidationResult:
    def __init__(self):
        self.results = []

    def check(self, name: str, passed: bool, detail: str = ""):
        self.results.append({
            "name": name, "status": "PASS" if passed else "FAIL", "detail": detail,
        })
        print(f"  [{'+' if passed else 'X'}] {name}" + (f" -- {detail}" if detail else ""))

    @property
    def total(self):
        return len(self.results)

    @property
    def passed(self):
        return sum(1 for r in self.results if r["status"] == "PASS")

    @property
    def failed(self):
        return sum(1 for r in self.results if r["status"] == "FAIL")

    def summary(self) -> str:
        return f"{self.passed}/{self.total} PASS"


def run_validation() -> tuple[ValidationResult, dict]:
    v = ValidationResult()
    print("=" * 60)
    print("PHASE 13 VALIDATION")
    print("=" * 60)

    print("\n[1] Structure")
    for rel in [
        "src/security/auth.py", "src/security/rate_limit.py", "src/security/audit.py",
        "src/production/readiness.py", "src/production/config_validation.py",
        "src/production/business_validation.py", ".env.example",
        "tests/test_security.py",
    ]:
        p = PROJECT_ROOT / rel
        v.check(f"exists {rel}", p.exists())

    print("\n[2] Imports")
    for mod in [
        "src.security.auth", "src.security.rate_limit", "src.security.audit",
        "src.production.readiness", "src.production.config_validation",
        "src.api.app", "src.api.metrics",
    ]:
        try:
            importlib.import_module(mod)
            v.check(f"import {mod}", True)
        except Exception as exc:
            v.check(f"import {mod}", False, type(exc).__name__)

    print("\n[3] Freeze / hashes")
    recs = load_registry()
    try:
        hashes = {r["model_id"]: verify_hash(r) for r in recs}
        v.check("all registry hashes match", True, f"n={len(hashes)}")
    except Exception as exc:
        hashes = {}
        v.check("all registry hashes match", False, str(exc))
    meta11 = json.loads(PHASE11_META_PATH.read_text(encoding="utf-8"))
    ok8, ch8 = hashes_unchanged(meta11["phase8_hashes"], snapshot_hashes(PHASE8_FREEZE_FILES))
    ok9, ch9 = hashes_unchanged(meta11["phase9_hashes"], snapshot_hashes(PHASE9_FREEZE_FILES))
    v.check("Phase 8 unchanged", ok8, str(ch8))
    v.check("Phase 9 unchanged", ok9, str(ch9))
    expected_uci = "331909f0fe191c0b9cb0418884b25eb59012f479f61f8b3e2ad51b729273e90d"
    expected_syn = "59a2b72024861d7f9c827596a52256af95facabfa796bdae5955374221cf1bf4"
    v.check("UCI h=1 hash unchanged", hashes.get("uci_h1_phase8_lightgbm") == expected_uci)
    v.check("SYNTHETIC h=1 hash unchanged", hashes.get("synthetic_h1_hurdle_th050") == expected_syn)
    v.check("models/final populated", MODELS_FINAL_DIR.exists() and any(MODELS_FINAL_DIR.glob("*.joblib")))
    v.check("registry path unchanged", REGISTRY_PATH.exists())

    print("\n[4] API security")
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    os.environ.pop("FORESIGHT_API_AUTH_ENABLED", None)
    os.environ.pop("FORESIGHT_API_API_KEY", None)
    os.environ["FORESIGHT_ENV"] = "development"
    os.environ["FORESIGHT_RATE_LIMIT_ENABLED"] = "false"
    client = TestClient(create_app())
    h = client.get("/health")
    v.check("GET /health public", h.status_code == 200 and h.json().get("status") == "ok")
    rdy = client.get("/ready")
    v.check("GET /ready", rdy.status_code == 200 and rdy.json().get("status") == "ready")
    v.check("security header X-Content-Type-Options", h.headers.get("x-content-type-options") == "nosniff")
    v.check("security header X-Frame-Options", h.headers.get("x-frame-options") == "DENY")

    os.environ["FORESIGHT_API_AUTH_ENABLED"] = "true"
    os.environ["FORESIGHT_API_API_KEY"] = "phase13-validate-key"
    authed = TestClient(create_app())
    denied = authed.get("/model")
    v.check("unauthenticated /model rejected", denied.status_code == 401)
    v.check("secret not leaked", "phase13-validate-key" not in denied.text)
    allowed = authed.get("/model", headers={"X-API-Key": "phase13-validate-key"})
    v.check("authenticated /model accepted", allowed.status_code == 200)
    v.check("health still public under auth", authed.get("/health").status_code == 200)

    os.environ["FORESIGHT_API_AUTH_ENABLED"] = "false"
    os.environ["FORESIGHT_RATE_LIMIT_ENABLED"] = "true"
    os.environ["FORESIGHT_RATE_LIMIT_REQUESTS"] = "2"
    os.environ["FORESIGHT_RATE_LIMIT_FORECAST_REQUESTS"] = "2"
    limited = TestClient(create_app())
    c1 = limited.get("/model").status_code
    c2 = limited.get("/model").status_code
    c3 = limited.get("/model").status_code
    v.check("rate limit 429", c1 == 200 and c2 == 200 and c3 == 429)
    os.environ["FORESIGHT_RATE_LIMIT_ENABLED"] = "false"

    print("\n[5] Configuration")
    os.environ["FORESIGHT_ENV"] = "development"
    v.check("development config valid", validate_runtime_config() == [])
    os.environ["FORESIGHT_ENV"] = "production"
    os.environ["FORESIGHT_API_AUTH_ENABLED"] = "false"
    os.environ.pop("FORESIGHT_API_API_KEY", None)
    prod_errors = validate_runtime_config()
    v.check("production fails without auth", any("AUTH" in e.upper() or "API_KEY" in e.upper() for e in prod_errors))
    os.environ["FORESIGHT_ENV"] = "development"
    os.environ["FORESIGHT_API_AUTH_ENABLED"] = "false"

    print("\n[6] Business validation")
    biz = write_business_validation_report()
    v.check("ten questions evidenced", biz.get("questions_with_evidence") == 10, str(biz.get("questions_with_evidence")))
    v.check("no automated replenishment claim", biz.get("automated_replenishment_implemented") is False)
    v.check("inventory risk artifact", biz.get("inventory_risk", {}).get("status") == "PASS")
    v.check("forecast evidence", biz.get("forecast_evidence", {}).get("status") == "PASS")

    print("\n[7] Performance")
    perf = run_benchmark()
    v.check("single forecast timed", perf["single_s"] >= 0)
    v.check("batch10 timed", perf["batch10_s"] >= 0)
    v.check("batch100 timed", perf["batch100_s"] >= 0)
    v.check("batch500 timed", perf["batch500_s"] >= 0)
    print(
        f"    load_s={perf['load_s']} single_s={perf['single_s']} "
        f"batch10_s={perf['batch10_s']} batch100_s={perf['batch100_s']} "
        f"batch500_s={perf['batch500_s']}"
    )

    print("\n[8] Version")
    v.check("app version 0.13.x", str(APP_VERSION).startswith("0.13"))

    print("\n" + "=" * 60)
    print(f"VALIDATION RESULT: {v.summary()}")
    if v.failed:
        for row in v.results:
            if row["status"] == "FAIL":
                print(f"  FAIL: {row['name']}: {row['detail']}")
    print("=" * 60)
    return v, {"performance": perf, "business": {
        "questions_with_evidence": biz.get("questions_with_evidence"),
        "status": biz.get("status"),
    }, "hashes": hashes}


if __name__ == "__main__":
    try:
        result, extra = run_validation()
        sys.exit(0 if result.failed == 0 else 1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
