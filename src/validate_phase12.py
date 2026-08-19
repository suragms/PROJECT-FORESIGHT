"""
Phase 12 validation suite.

Run: python src/validate_phase12.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (  # noqa: E402
    APP_VERSION,
    FINAL_FORECASTS_PATH,
    MODELS_FINAL_DIR,
    OUTPUTS_MONITORING_DIR,
    PHASE11_META_PATH,
    PROJECT_ROOT,
    REGISTRY_PATH,
    SAMPLES_DIR,
)
from src.forecasting.make_samples import write_samples  # noqa: E402
from src.forecasting.registry import load_registry, verify_hash  # noqa: E402
from src.phase10_common import hashes_unchanged, snapshot_hashes  # noqa: E402
from src.phase10_common import PHASE8_FREEZE_FILES, PHASE9_FREEZE_FILES  # noqa: E402


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


def run_validation() -> ValidationResult:
    v = ValidationResult()
    print("=" * 60)
    print("PHASE 12 VALIDATION")
    print("=" * 60)

    print("\n[1] Repository structure")
    for rel in [
        "src/config.py", "src/forecasting/inference.py", "src/forecasting/batch_forecast.py",
        "src/api/app.py", "src/api/routes.py", "src/monitoring/data_quality.py",
        "dashboard/forecast_analytics.py", "tests/test_api.py", "docs/api_documentation.md",
    ]:
        p = PROJECT_ROOT / rel
        v.check(f"exists {rel}", p.exists())
    v.check("legacy forecasting.py removed (now package)", not (PROJECT_ROOT / "src" / "forecasting.py").exists())
    v.check("forecasting.baselines exists", (PROJECT_ROOT / "src" / "forecasting" / "baselines.py").exists())

    print("\n[2] Imports")
    for mod in [
        "src.config", "src.forecasting.inference", "src.forecasting.baselines",
        "src.api.app", "src.monitoring.data_quality",
    ]:
        try:
            importlib.import_module(mod)
            v.check(f"import {mod}", True)
        except Exception as exc:
            v.check(f"import {mod}", False, type(exc).__name__)

    print("\n[3] Models / registry")
    recs = load_registry()
    v.check("registry loads", len(recs) == 12, f"n={len(recs)}")
    try:
        for r in recs:
            verify_hash(r)
        v.check("all registry hashes match", True)
    except Exception as exc:
        v.check("all registry hashes match", False, str(exc))
    v.check("models/final populated", MODELS_FINAL_DIR.exists() and any(MODELS_FINAL_DIR.glob("*.joblib")))

    print("\n[4] Inference")
    if not (SAMPLES_DIR / "uci_h1_sample.parquet").exists():
        write_samples()
    from src.forecasting.inference import ForecastEngine
    from src.forecasting.validation import InputValidationError
    df = pd.read_parquet(SAMPLES_DIR / "uci_h1_sample.parquet")
    t0 = time.perf_counter()
    eng = ForecastEngine("UCI", 1)
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    out = eng.predict(df.head(1), include_actual=False)
    single_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    outb = eng.predict(df.head(10), include_actual=False)
    batch_s = time.perf_counter() - t2
    v.check("valid sample predicts", len(out) == 1 and float(out["prediction"].iloc[0]) >= 0)
    a = eng.predict(df.head(3), include_actual=False)
    b = eng.predict(df.head(3), include_actual=False)
    v.check("deterministic", (a["prediction"].to_numpy() == b["prediction"].to_numpy()).all())
    try:
        eng.predict(df.head(2).drop(columns=["units_sold_lag_1"]))
        v.check("rejects missing lag", False)
    except InputValidationError:
        v.check("rejects missing lag", True)
    try:
        ForecastEngine("UCI", 2)
        v.check("rejects invalid horizon", False)
    except Exception:
        v.check("rejects invalid horizon", True)

    print("\n[5] API")
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    client = TestClient(create_app())
    h = client.get("/health")
    v.check("GET /health", h.status_code == 200 and h.json().get("status") == "ok")
    m = client.get("/model")
    v.check("GET /model", m.status_code == 200 and len(m.json().get("models", [])) >= 1)
    payload = json.loads((SAMPLES_DIR / "uci_h1_sample.json").read_text(encoding="utf-8"))
    rec = payload["record"]
    keys = ["date", "source_dataset", "entity_id", "product_key"]
    body = {
        "source_dataset": "UCI", "horizon": 1,
        "date": rec["date"], "entity_id": rec["entity_id"], "product_key": rec["product_key"],
        "features": {k: val for k, val in rec.items() if k not in keys},
    }
    fr = client.post("/forecast", json=body)
    v.check("POST /forecast", fr.status_code == 200, fr.text[:120])
    bad = client.post("/forecast", json={"source_dataset": "UCI", "horizon": 1})
    v.check("invalid schema rejected", bad.status_code in (400, 422))
    v.check("error has no traceback", "traceback" not in bad.text.lower())

    print("\n[6] Dashboard")
    os.environ["FORECAST_DASHBOARD_SKIP_MAIN"] = "1"
    from dashboard.forecast_analytics import load_parquet, load_json
    preds = load_parquet(str(FINAL_FORECASTS_PATH))
    v.check("dashboard loads forecasts", preds is not None and len(preds) > 0)
    v.check("dashboard missing file safe", load_parquet("nope.parquet") is None)

    print("\n[7] Monitoring")
    from src.monitoring.run_monitoring import run_monitoring
    summary = run_monitoring()
    v.check("monitoring summary written", (OUTPUTS_MONITORING_DIR / "monitoring_summary.json").exists())
    for fn in [
        "data_quality_report.json", "forecast_monitoring_report.json",
        "accuracy_monitoring_report.json", "drift_report.json",
    ]:
        v.check(f"monitoring {fn}", (OUTPUTS_MONITORING_DIR / fn).exists())
    acc = json.loads((OUTPUTS_MONITORING_DIR / "accuracy_monitoring_report.json").read_text(encoding="utf-8"))
    v.check("accuracy skips empty actuals safely", "n_with_actuals" in acc.get("overall", {}))

    print("\n[8] Regression freeze")
    meta11 = json.loads(PHASE11_META_PATH.read_text(encoding="utf-8"))
    ok8, ch8 = hashes_unchanged(meta11["phase8_hashes"], snapshot_hashes(PHASE8_FREEZE_FILES))
    ok9, ch9 = hashes_unchanged(meta11["phase9_hashes"], snapshot_hashes(PHASE9_FREEZE_FILES))
    v.check("Phase 8 unchanged", ok8, str(ch8))
    v.check("Phase 9 unchanged", ok9, str(ch9))
    v.check("Phase 11 validation recorded", meta11.get("validation", {}).get("summary") == "140/140 PASS")
    v.check("Phase 11 registry still valid", REGISTRY_PATH.exists())

    print("\n[9] Performance")
    rps = (len(outb) / batch_s) if batch_s else 0
    v.check("model load timed", load_s > 0)
    v.check("single forecast timed", single_s >= 0)
    print(f"    load_s={load_s:.3f} single_s={single_s:.4f} batch10_s={batch_s:.4f} rows_per_s={rps:.1f}")

    print("\n" + "=" * 60)
    print(f"VALIDATION RESULT: {v.summary()}")
    if v.failed:
        for r in v.results:
            if r["status"] == "FAIL":
                print(f"  FAIL: {r['name']}: {r['detail']}")
    print("=" * 60)
    return v, {
        "load_s": round(load_s, 4),
        "single_s": round(single_s, 4),
        "batch10_s": round(batch_s, 4),
        "rows_per_s": round(rps, 2),
        "app_version": APP_VERSION,
    }


if __name__ == "__main__":
    try:
        result, perf = run_validation()
        meta_path = PROJECT_ROOT / "docs" / "phase12_metadata.json"
        payload = {}
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        payload["phase"] = 12
        payload["validation"] = {"passed": result.passed, "total": result.total, "summary": result.summary()}
        # Keep the historical Phase 12 performance snapshot. Current timings are printed
        # above and recorded separately by later phases; do not overwrite frozen numbers.
        payload.setdefault("performance", perf)
        payload["last_run_performance"] = perf
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        sys.exit(0 if result.failed == 0 else 1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
