"""Load and resolve Phase 11 final models. Never accept filesystem paths from callers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import MODELS_FINAL_DIR, PROJECT_ROOT, REGISTRY_PATH, SUPPORTED_DATASETS, SUPPORTED_HORIZONS
from src.phase11_common import file_sha256

logger = logging.getLogger("forecast_service.registry")


class RegistryError(ValueError):
    """Invalid registry or unregistered model request."""


def load_registry(path: Path | None = None) -> list[dict[str, Any]]:
    p = Path(path or REGISTRY_PATH)
    if not p.exists():
        raise RegistryError(f"Model registry not found: {p}")
    with open(p, encoding="utf-8") as f:
        recs = json.load(f)
    if not isinstance(recs, list) or not recs:
        raise RegistryError("Registry must be a non-empty list")
    ids = [r.get("model_id") for r in recs]
    if len(ids) != len(set(ids)):
        raise RegistryError("Duplicate model_id in registry")
    return recs


def resolve_model_file(record: dict[str, Any]) -> Path:
    raw = record.get("model_file")
    if not raw or not isinstance(raw, str):
        raise RegistryError("Registry record missing model_file")
    # Reject absolute paths outside the repo and any parent traversal.
    raw = raw.replace("\\", "/")
    p = Path(raw)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (PROJECT_ROOT / p).resolve()
    models_root = MODELS_FINAL_DIR.resolve()
    try:
        resolved.relative_to(models_root)
    except ValueError as exc:
        raise RegistryError(
            f"Model file is not under models/final/: {resolved}"
        ) from exc
    if ".." in Path(raw).parts:
        raise RegistryError("Parent-directory traversal is not allowed in model_file")
    return resolved


def get_record(model_id: str, registry: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    recs = registry if registry is not None else load_registry()
    hits = [r for r in recs if r.get("model_id") == model_id]
    if not hits:
        raise RegistryError(f"Unregistered model_id: {model_id}")
    return hits[0]


def resolve_selected(dataset: str, horizon: int, registry: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if dataset not in SUPPORTED_DATASETS:
        raise RegistryError(f"Unsupported dataset {dataset}. Supported: {SUPPORTED_DATASETS}")
    if int(horizon) not in SUPPORTED_HORIZONS:
        raise RegistryError(f"Unsupported horizon {horizon}. Supported: {SUPPORTED_HORIZONS}")
    recs = registry if registry is not None else load_registry()
    hits = [
        r for r in recs
        if r.get("status") == "selected"
        and r.get("dataset") == dataset
        and int(r.get("horizon")) == int(horizon)
    ]
    if not hits:
        raise RegistryError(f"No selected model for dataset={dataset} horizon={horizon}")
    return hits[0]


def interval_companion(dataset: str, horizon: int, registry: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    recs = registry if registry is not None else load_registry()
    hits = [
        r for r in recs
        if r.get("status") == "interval_companion"
        and r.get("dataset") == dataset
        and int(r.get("horizon")) == int(horizon)
    ]
    return hits[0] if hits else None


def verify_hash(record: dict[str, Any]) -> str:
    path = resolve_model_file(record)
    if not path.exists():
        raise RegistryError(f"Model file missing: {path}")
    got = file_sha256(str(path))
    expected = record.get("hash")
    if not expected or got != expected:
        raise RegistryError(
            f"Hash mismatch for {record.get('model_id')}: expected {expected}, got {got}"
        )
    logger.info("hash_ok model_id=%s sha256=%s", record.get("model_id"), got)
    return got
