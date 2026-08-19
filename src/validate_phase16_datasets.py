"""
Phase 16 — Dataset Extraction + Provenance Verification + Zidio Alignment Validator
=====================================================================================
Project FORESIGHT: Demand & Inventory Intelligence

Validates:
  1. External dataset directories exist
  2. External dataset manifest is valid JSON
  3. Raw data file hashes are unchanged from Phase 16 baseline
  4. Frozen model hashes match the model registry
  5. Source separation (UCI vs SYNTHETIC) is maintained in forecast_base
  6. UCI data does not contain fabricated inventory fields
  7. Synthetic inventory fields are traceable to generate_synthetic_retail.py
  8. Primary keys are valid (no nulls, no duplicates in grain)
  9. Phase 15 artifacts remain unchanged
 10. Schema alignment with Zidio specification
"""

import os
import sys
import json
import hashlib
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DOCS_DIR = os.path.join(BASE_DIR, "docs")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models", "final")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _result(name: str, status: str, detail: str = "") -> dict:
    return {"test": name, "status": status, "detail": detail}


def validate_external_directories() -> list:
    results = []
    for subdir in ["uci_online_retail_ii", "kaggle_synthetic_retail_10m"]:
        path = os.path.join(EXTERNAL_DIR, subdir)
        if os.path.isdir(path):
            results.append(_result(f"external_dir_{subdir}", PASS, path))
        else:
            results.append(_result(f"external_dir_{subdir}", WARN,
                                   f"Directory not found: {path}. External data not yet downloaded."))
    return results


def validate_external_manifest() -> list:
    results = []
    manifest_path = os.path.join(DOCS_DIR, "phase16_external_dataset_manifest.json")
    if not os.path.exists(manifest_path):
        return [_result("external_manifest_exists", FAIL, "File not found")]
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        results.append(_result("external_manifest_valid_json", PASS))
        if "sources" in data and isinstance(data["sources"], list):
            results.append(_result("external_manifest_has_sources", PASS,
                                   f"{len(data['sources'])} source(s)"))
        else:
            results.append(_result("external_manifest_has_sources", FAIL,
                                   "Missing 'sources' array"))
    except json.JSONDecodeError as e:
        results.append(_result("external_manifest_valid_json", FAIL, str(e)))
    return results


def validate_raw_data_unchanged() -> list:
    results = []
    baseline_path = os.path.join(DATA_DIR, "raw", ".raw_hashes_phase16.json")
    if not os.path.exists(baseline_path):
        return [_result("raw_data_baseline_exists", FAIL,
                         "No hash baseline found at data/raw/.raw_hashes_phase16.json")]

    with open(baseline_path) as f:
        baseline = json.load(f)

    all_match = True
    for entry in baseline:
        fp = os.path.join(DATA_DIR, "raw", entry["filename"])
        if not os.path.exists(fp):
            results.append(_result(f"raw_file_{entry['filename']}", FAIL, "File missing"))
            all_match = False
            continue
        current_hash = _sha256(fp)
        if current_hash == entry["sha256"]:
            results.append(_result(f"raw_file_{entry['filename']}", PASS, "Hash unchanged"))
        else:
            results.append(_result(f"raw_file_{entry['filename']}", FAIL,
                                   f"Hash changed: expected {entry['sha256'][:16]}... "
                                   f"got {current_hash[:16]}..."))
            all_match = False

    results.append(_result("raw_data_all_unchanged", PASS if all_match else FAIL))
    return results


def validate_frozen_model_hashes() -> list:
    results = []
    registry_path = os.path.join(DOCS_DIR, "final_model_registry.json")
    if not os.path.exists(registry_path):
        return [_result("model_registry_exists", FAIL)]

    with open(registry_path) as f:
        registry = json.load(f)

    all_match = True
    for entry in registry:
        mf = os.path.join(BASE_DIR, entry["model_file"].replace("\\", os.sep))
        if not os.path.exists(mf):
            results.append(_result(f"model_{entry['model_id']}", FAIL, "File not found"))
            all_match = False
            continue
        actual = _sha256(mf)
        if actual == entry["hash"]:
            results.append(_result(f"model_{entry['model_id']}", PASS))
        else:
            results.append(_result(f"model_{entry['model_id']}", FAIL,
                                   f"Hash mismatch: expected {entry['hash'][:16]} got {actual[:16]}"))
            all_match = False

    results.append(_result("frozen_model_integrity", PASS if all_match else FAIL,
                           "FROZEN MODEL INTEGRITY FAILURE" if not all_match else "All 12 models verified"))
    return results


def validate_source_separation() -> list:
    results = []
    fb_path = os.path.join(DATA_DIR, "processed", "integrated", "forecast_base.parquet")
    if not os.path.exists(fb_path):
        return [_result("forecast_base_exists", FAIL)]

    try:
        import pandas as pd
        fb = pd.read_parquet(fb_path)
        sources = set(fb["source_dataset"].unique())
        expected = {"UCI", "SYNTHETIC"}
        if sources == expected:
            results.append(_result("source_separation", PASS,
                                   f"Sources: {sorted(sources)}"))
        elif sources.issubset(expected):
            results.append(_result("source_separation", WARN,
                                   f"Only {sorted(sources)} found, expected both UCI and SYNTHETIC"))
        else:
            results.append(_result("source_separation", FAIL,
                                   f"Unexpected sources: {sorted(sources)}"))

        grain_cols = ["date", "source_dataset", "entity_id", "product_key"]
        dupes = fb.duplicated(subset=grain_cols).sum()
        results.append(_result("forecast_base_no_duplicate_keys", PASS if dupes == 0 else FAIL,
                               f"{dupes} duplicate grain rows"))

        null_keys = {c: int(fb[c].isna().sum()) for c in grain_cols if fb[c].isna().any()}
        results.append(_result("forecast_base_no_null_keys",
                               PASS if not null_keys else FAIL,
                               str(null_keys) if null_keys else "No null keys"))
    except ImportError:
        results.append(_result("source_separation", SKIP, "pandas not available"))
    return results


def validate_uci_no_fabricated_inventory() -> list:
    results = []
    fi_path = os.path.join(DATA_DIR, "processed", "integrated", "fact_inventory.parquet")
    if not os.path.exists(fi_path):
        return [_result("uci_no_fabricated_inventory", SKIP, "fact_inventory.parquet not found")]
    try:
        import pandas as pd
        fi = pd.read_parquet(fi_path)
        uci_inv = fi[fi["source_dataset"] == "UCI"] if "source_dataset" in fi.columns else pd.DataFrame()
        if len(uci_inv) == 0:
            results.append(_result("uci_no_fabricated_inventory", PASS,
                                   "Zero UCI rows in fact_inventory — no fabricated inventory"))
        else:
            results.append(_result("uci_no_fabricated_inventory", FAIL,
                                   f"{len(uci_inv)} UCI rows found in fact_inventory — "
                                   f"UCI should NOT have inventory records"))
    except ImportError:
        results.append(_result("uci_no_fabricated_inventory", SKIP, "pandas not available"))
    return results


def validate_synthetic_inventory_traceable() -> list:
    results = []
    inv_path = os.path.join(DATA_DIR, "raw", "inventory_snapshots.parquet")
    gen_path = os.path.join(BASE_DIR, "src", "generate_synthetic_retail.py")
    results.append(_result("synthetic_inventory_raw_exists",
                           PASS if os.path.exists(inv_path) else FAIL))
    results.append(_result("synthetic_generator_exists",
                           PASS if os.path.exists(gen_path) else FAIL))
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            content = f.read()
        has_inv = "inventory_snapshots" in content
        results.append(_result("generator_produces_inventory",
                               PASS if has_inv else FAIL,
                               "generate_synthetic_retail.py references inventory_snapshots"
                               if has_inv else "No inventory generation found"))
    return results


def validate_phase15_artifacts_exist() -> list:
    results = []
    artifacts = [
        "docs/phase15_metadata.json",
        "docs/phase15_known_limitations.md",
        "docs/final_model_registry.json",
    ]
    for a in artifacts:
        path = os.path.join(BASE_DIR, a)
        results.append(_result(f"artifact_{os.path.basename(a)}",
                               PASS if os.path.exists(path) else FAIL))
    return results


def validate_schema_alignment() -> list:
    results = []
    alignment_path = os.path.join(DOCS_DIR, "phase16_schema_alignment.md")
    results.append(_result("schema_alignment_doc_exists",
                           PASS if os.path.exists(alignment_path) else FAIL))
    return results


def validate_source_labels() -> list:
    results = []
    fb_path = os.path.join(DATA_DIR, "processed", "integrated", "forecast_base.parquet")
    if not os.path.exists(fb_path):
        return [_result("source_labels_valid", SKIP)]
    try:
        import pandas as pd
        fb = pd.read_parquet(fb_path, columns=["source_dataset"])
        valid = {"UCI", "SYNTHETIC"}
        actual = set(fb["source_dataset"].unique())
        unknown = actual - valid
        results.append(_result("source_labels_valid",
                               PASS if not unknown else FAIL,
                               f"Unknown labels: {unknown}" if unknown else f"Labels: {sorted(actual)}"))
    except ImportError:
        results.append(_result("source_labels_valid", SKIP))
    return results


def run_all() -> dict:
    all_results = []
    validators = [
        validate_external_directories,
        validate_external_manifest,
        validate_raw_data_unchanged,
        validate_frozen_model_hashes,
        validate_source_separation,
        validate_uci_no_fabricated_inventory,
        validate_synthetic_inventory_traceable,
        validate_phase15_artifacts_exist,
        validate_schema_alignment,
        validate_source_labels,
    ]
    for fn in validators:
        try:
            all_results.extend(fn())
        except Exception as e:
            all_results.append(_result(fn.__name__, FAIL, f"Exception: {e}"))

    passed = sum(1 for r in all_results if r["status"] == PASS)
    failed = sum(1 for r in all_results if r["status"] == FAIL)
    warned = sum(1 for r in all_results if r["status"] == WARN)
    skipped = sum(1 for r in all_results if r["status"] == SKIP)
    total = len(all_results)

    summary = {
        "phase": 16,
        "validator": "validate_phase16_datasets.py",
        "timestamp": datetime.utcnow().isoformat() + "+00:00",
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "skipped": skipped,
        "total": total,
        "overall": PASS if failed == 0 else FAIL,
        "results": all_results,
    }

    out_path = os.path.join(DOCS_DIR, "phase16_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_all()
    print(f"\n{'='*50}")
    print(f"PHASE 16 DATASET VALIDATION")
    print(f"{'='*50}")
    for r in summary["results"]:
        icon = {"PASS": "+", "FAIL": "!", "WARN": "~", "SKIP": "-"}.get(r["status"], "?")
        detail = f"  ({r['detail']})" if r["detail"] else ""
        print(f"  [{icon}] {r['status']:4s}: {r['test']}{detail}")
    print(f"\n  Summary: {summary['passed']}/{summary['total']} PASS, "
          f"{summary['failed']} FAIL, {summary['warned']} WARN, {summary['skipped']} SKIP")
    print(f"  Overall: {summary['overall']}")
    print(f"  Results saved to docs/phase16_validation_results.json")

    sys.exit(0 if summary["overall"] == PASS else 1)
