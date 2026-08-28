"""
Phase 16 — Dataset Alignment & Provenance Tests
=================================================
Tests verify data integrity, provenance, and frozen model safety.
No internet access required.
"""

import os
import sys
import json
import hashlib
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DOCS_DIR = os.path.join(BASE_DIR, "docs")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models", "final")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- External manifest ---

class TestExternalManifest:
    def test_manifest_exists(self):
        assert os.path.exists(os.path.join(DOCS_DIR, "phase16_external_dataset_manifest.json"))

    def test_manifest_valid_json(self):
        with open(os.path.join(DOCS_DIR, "phase16_external_dataset_manifest.json")) as f:
            data = json.load(f)
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) >= 2

    def test_source_ids_valid(self):
        with open(os.path.join(DOCS_DIR, "phase16_external_dataset_manifest.json")) as f:
            data = json.load(f)
        ids = {s["source_id"] for s in data["sources"]}
        assert "UCI_ONLINE_RETAIL_II" in ids
        assert "KAGGLE_SYNTHETIC_RETAIL_10M" in ids


# --- UCI must not contain fabricated inventory ---

class TestUCINoFabricatedInventory:
    @pytest.fixture
    def fact_inventory(self):
        path = os.path.join(DATA_DIR, "processed", "integrated", "fact_inventory.parquet")
        if not os.path.exists(path):
            pytest.skip("fact_inventory.parquet not found")
        import pandas as pd
        return pd.read_parquet(path)

    def test_no_uci_inventory_rows(self, fact_inventory):
        if "source_dataset" not in fact_inventory.columns:
            pytest.skip("No source_dataset column")
        uci_rows = fact_inventory[fact_inventory["source_dataset"] == "UCI"]
        assert len(uci_rows) == 0, (
            f"UCI should have zero inventory rows but found {len(uci_rows)}. "
            f"UCI Online Retail II does NOT provide on-hand inventory."
        )


# --- Synthetic inventory fields traceable ---

class TestSyntheticInventoryTraceable:
    def test_raw_inventory_exists(self):
        assert os.path.exists(os.path.join(DATA_DIR, "raw", "inventory_snapshots.parquet"))

    def test_generator_references_inventory(self):
        gen_path = os.path.join(BASE_DIR, "src", "generate_synthetic_retail.py")
        assert os.path.exists(gen_path)
        with open(gen_path) as f:
            content = f.read()
        assert "inventory_snapshots" in content


# --- Primary keys valid ---

class TestPrimaryKeys:
    @pytest.fixture
    def forecast_base(self):
        path = os.path.join(DATA_DIR, "processed", "integrated", "forecast_base.parquet")
        if not os.path.exists(path):
            pytest.skip("forecast_base.parquet not found")
        import pandas as pd
        return pd.read_parquet(path)

    def test_no_null_grain_keys(self, forecast_base):
        grain = ["date", "source_dataset", "entity_id", "product_key"]
        for col in grain:
            assert forecast_base[col].isna().sum() == 0, f"Null values in grain column {col}"

    def test_no_duplicate_grain(self, forecast_base):
        grain = ["date", "source_dataset", "entity_id", "product_key"]
        dupes = forecast_base.duplicated(subset=grain).sum()
        assert dupes == 0, f"{dupes} duplicate grain rows"


# --- Source datasets remain separated ---

class TestSourceSeparation:
    @pytest.fixture
    def forecast_base(self):
        path = os.path.join(DATA_DIR, "processed", "integrated", "forecast_base.parquet")
        if not os.path.exists(path):
            pytest.skip("forecast_base.parquet not found")
        import pandas as pd
        return pd.read_parquet(path)

    def test_only_valid_source_labels(self, forecast_base):
        valid = {"UCI", "SYNTHETIC"}
        actual = set(forecast_base["source_dataset"].unique())
        assert actual.issubset(valid), f"Unknown source labels: {actual - valid}"

    def test_both_sources_present(self, forecast_base):
        actual = set(forecast_base["source_dataset"].unique())
        assert "UCI" in actual and "SYNTHETIC" in actual


# --- Frozen model hashes unchanged ---

class TestFrozenModelHashes:
    @pytest.fixture
    def registry(self):
        path = os.path.join(DOCS_DIR, "final_model_registry.json")
        if not os.path.exists(path):
            pytest.skip("Model registry not found")
        with open(path) as f:
            return json.load(f)

    def test_all_model_hashes_match(self, registry):
        for entry in registry:
            mf = os.path.join(BASE_DIR, entry["model_file"].replace("\\", os.sep))
            assert os.path.exists(mf), f"Model file missing: {mf}"
            actual = _sha256(mf)
            assert actual == entry["hash"], (
                f"FROZEN MODEL INTEGRITY FAILURE: {entry['model_id']} "
                f"expected {entry['hash'][:20]} got {actual[:20]}"
            )

    def test_no_unexpected_model_files(self, registry):
        registered = {os.path.basename(e["model_file"]) for e in registry}
        actual = {
            f for f in os.listdir(MODELS_DIR)
            if os.path.isfile(os.path.join(MODELS_DIR, f)) and f.endswith(".joblib")
        } if os.path.isdir(MODELS_DIR) else set()
        unexpected = actual - registered
        assert len(unexpected) == 0, f"Unexpected model files: {unexpected}"


# --- Existing raw data unchanged ---

class TestRawDataUnchanged:
    @pytest.fixture
    def baseline(self):
        path = os.path.join(DATA_DIR, "raw", ".raw_hashes_phase16.json")
        if not os.path.exists(path):
            pytest.skip("Raw data hash baseline not found")
        with open(path) as f:
            return json.load(f)

    def test_all_raw_hashes_match(self, baseline):
        for entry in baseline:
            fp = os.path.join(DATA_DIR, "raw", entry["filename"])
            assert os.path.exists(fp), f"Raw file missing: {entry['filename']}"
            actual = _sha256(fp)
            assert actual == entry["sha256"], (
                f"Raw data modified: {entry['filename']} "
                f"expected {entry['sha256'][:20]} got {actual[:20]}"
            )


# --- Phase 15 artifacts unchanged ---

class TestPhase15Artifacts:
    def test_phase15_metadata_exists(self):
        assert os.path.exists(os.path.join(DOCS_DIR, "phase15_metadata.json"))

    def test_phase15_known_limitations_exists(self):
        assert os.path.exists(os.path.join(DOCS_DIR, "phase15_known_limitations.md"))

    def test_model_registry_exists(self):
        assert os.path.exists(os.path.join(DOCS_DIR, "final_model_registry.json"))

    def test_phase15_metadata_valid(self):
        with open(os.path.join(DOCS_DIR, "phase15_metadata.json")) as f:
            data = json.load(f)
        assert data["phase"] == 15
        assert "kpi_layer" in data


# --- Documentation KPIs match source artifacts ---

class TestDocumentationKPIs:
    def test_reorder_review_count_matches(self):
        with open(os.path.join(DOCS_DIR, "phase15_metadata.json")) as f:
            meta = json.load(f)
        reported = meta["kpi_layer"]["reorder_review_count"]
        risk_path = os.path.join(BASE_DIR, "outputs", "risk_scores", "inventory_risk_matrix.parquet")
        if not os.path.exists(risk_path):
            pytest.skip("Risk matrix not found")
        import pandas as pd
        rm = pd.read_parquet(risk_path)
        actual = int(rm["reorder_triggered"].sum())
        assert reported == actual, (
            f"Phase 15 metadata reports reorder_review_count={reported} "
            f"but risk matrix has {actual}"
        )

    def test_stockout_critical_high_matches(self):
        with open(os.path.join(DOCS_DIR, "phase15_metadata.json")) as f:
            meta = json.load(f)
        reported = meta["kpi_layer"]["stockout_critical_high"]
        risk_path = os.path.join(BASE_DIR, "outputs", "risk_scores", "inventory_risk_matrix.parquet")
        if not os.path.exists(risk_path):
            pytest.skip("Risk matrix not found")
        import pandas as pd
        rm = pd.read_parquet(risk_path)
        actual = int((rm["stockout_risk_level"] == "CRITICAL / HIGH").sum())
        assert reported == actual
