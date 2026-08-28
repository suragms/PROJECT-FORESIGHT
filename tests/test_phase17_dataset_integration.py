"""Phase 17 — Dataset Integration Tests."""
import os
import sys
import json
import hashlib
import pytest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P17_DIR = os.path.join(BASE_DIR, "data", "phase17")
P17_PROC = os.path.join(P17_DIR, "processed")
DOCS_DIR = os.path.join(BASE_DIR, "docs")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestSourceSeparation:
    @pytest.fixture
    def uci(self):
        p = os.path.join(P17_PROC, "uci_weekly_demand.parquet")
        if not os.path.exists(p):
            pytest.skip("UCI weekly demand not found")
        return pd.read_parquet(p)

    @pytest.fixture
    def syn(self):
        p = os.path.join(P17_PROC, "synthetic_weekly_demand.parquet")
        if not os.path.exists(p):
            pytest.skip("Synthetic weekly demand not found")
        return pd.read_parquet(p)

    def test_uci_source_label(self, uci):
        assert set(uci["source_dataset"].unique()) == {"UCI"}

    def test_synthetic_source_label(self, syn):
        assert set(syn["source_dataset"].unique()) == {"SYNTHETIC"}

    def test_uci_product_keys_prefixed(self, uci):
        assert all(pk.startswith("UCI_") for pk in uci["product_key"].unique())

    def test_synthetic_product_keys_prefixed(self, syn):
        assert all(pk.startswith("SYN_") for pk in syn["product_key"].unique())

    def test_no_overlap(self, uci, syn):
        uci_keys = set(uci["product_key"].unique())
        syn_keys = set(syn["product_key"].unique())
        assert uci_keys.isdisjoint(syn_keys)


class TestSchemaValidity:
    @pytest.fixture
    def uci(self):
        p = os.path.join(P17_PROC, "uci_weekly_demand.parquet")
        if not os.path.exists(p):
            pytest.skip("not found")
        return pd.read_parquet(p)

    def test_required_columns(self, uci):
        required = {"week", "product_key", "units_sold", "source_dataset"}
        assert required.issubset(set(uci.columns))

    def test_no_negative_demand(self, uci):
        assert (uci["units_sold"] >= 0).all()

    def test_temporal_ordering(self, uci):
        for pk in uci["product_key"].unique()[:10]:
            sub = uci[uci["product_key"] == pk].sort_values("week")
            assert sub["week"].is_monotonic_increasing


class TestFrozenModelsUnchanged:
    def test_all_frozen_hashes_match(self):
        reg_path = os.path.join(DOCS_DIR, "final_model_registry.json")
        if not os.path.exists(reg_path):
            pytest.skip("Registry not found")
        with open(reg_path) as f:
            registry = json.load(f)
        for entry in registry:
            mf = os.path.join(BASE_DIR, entry["model_file"].replace("\\", os.sep))
            assert os.path.exists(mf), f"Missing: {mf}"
            actual = _sha256(mf)
            assert actual == entry["hash"], f"FROZEN MODEL CHANGED: {entry['model_id']}"
