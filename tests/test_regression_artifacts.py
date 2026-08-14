"""Regression: Phase 8–11 artifacts must remain unchanged by Phase 12."""

from __future__ import annotations

import json

from src.config import PHASE10_META_PATH, PHASE11_META_PATH, PHASE8_FREEZE, PHASE9_FREEZE, REGISTRY_PATH
from src.forecasting.registry import load_registry, verify_hash
from src.phase10_common import hashes_unchanged, snapshot_hashes
from src.phase11_common import file_md5


def _md5_map(paths):
    out = {}
    for p in paths:
        out[str(p)] = {
            "exists": p.exists(),
            "md5": file_md5(str(p)) if p.exists() else None,
        }
    return out


def test_phase8_hashes_match_phase11_snapshot():
    meta = json.loads(PHASE11_META_PATH.read_text(encoding="utf-8"))
    current = snapshot_hashes([str(p) for p in PHASE8_FREEZE])
    ok, changed = hashes_unchanged(meta["phase8_hashes"], current)
    assert ok, changed


def test_phase9_hashes_match_phase11_snapshot():
    meta = json.loads(PHASE11_META_PATH.read_text(encoding="utf-8"))
    current = snapshot_hashes([str(p) for p in PHASE9_FREEZE])
    ok, changed = hashes_unchanged(meta["phase9_hashes"], current)
    assert ok, changed


def test_phase10_metadata_present():
    assert PHASE10_META_PATH.exists()
    meta = json.loads(PHASE10_META_PATH.read_text(encoding="utf-8"))
    assert meta.get("validation", {}).get("summary") == "87/87 PASS"


def test_phase11_registry_hashes():
    recs = load_registry()
    selected = [r for r in recs if r.get("status") == "selected"]
    assert len(selected) == 10
    for r in recs:
        verify_hash(r)


def test_phase11_report_exists():
    from src.config import PROJECT_ROOT
    assert (PROJECT_ROOT / "docs" / "final_forecasting_report.md").exists()
    meta = json.loads(PHASE11_META_PATH.read_text(encoding="utf-8"))
    assert meta.get("validation", {}).get("summary") == "140/140 PASS"
    assert meta.get("production_readiness") == "READY WITH MONITORING"
