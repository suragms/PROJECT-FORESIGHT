"""Phase 15 BI tests. Does not retrain models or invoke Phase 14 Docker."""

from __future__ import annotations

import os

os.environ["EXECUTIVE_DASHBOARD_SKIP_MAIN"] = "1"

import pandas as pd

from src.bi.exports import EXPORT_FILES, REQUIRED_COLUMNS, build_bi_exports, load_bi_tables, schema_issues
from src.bi.filters import apply_filters
from src.bi.kpis import demand_kpis, executive_kpi_row, forecast_kpis, inventory_kpis, load_risk
from src.bi.quality import freshness, quality_scorecard
from src.bi.recommendations import build_recommendations, recommend_row
from src.bi.rules import classify_growth, demand_inventory_cell
from src.config import OUTPUTS_BI_DIR

EXPECTED_UCI_HASH = "331909f0fe191c0b9cb0418884b25eb59012f479f61f8b3e2ad51b729273e90d"
EXPECTED_SYN_HASH = "59a2b72024861d7f9c827596a52256af95facabfa796bdae5955374221cf1bf4"


def test_growth_rule_documented():
    assert classify_growth(None, 100, 100) == "Insufficient Evidence"
    assert classify_growth(0.20, 3, 3) == "Insufficient Evidence"
    assert classify_growth(0.02, 20, 20) == "Stable"
    assert classify_growth(0.20, 20, 20) == "Growing"
    assert classify_growth(-0.20, 20, 20) == "Declining"


def test_risk_matrix_cells():
    assert demand_inventory_cell(False, False) == "Normal"
    assert demand_inventory_cell(True, False) == "Stockout Review"
    assert demand_inventory_cell(False, True) == "Overstock Review"
    assert demand_inventory_cell(True, True) == "Critical Review"


def test_kpi_layer_uses_existing_data():
    risk = load_risk()
    inv = inventory_kpis(risk)
    dem = demand_kpis(risk)
    fc = forecast_kpis()
    row = executive_kpi_row()
    assert inv["n_rows"] == 1000
    assert inv["stockout_critical_high"] == 733
    assert inv["reorder_review_count"] > 0
    assert "1000-row" in str(inv["extract_note"])
    assert dem["growth_on_extract"] == "Insufficient Evidence"
    assert fc["mae"] != "NOT AVAILABLE"
    assert fc["interval_coverage"] == "NOT AVAILABLE"
    assert row["decision_support_only"] is True
    assert row["forecast_mae"] == fc["mae"]


def test_filters_do_not_invent_columns():
    df = pd.DataFrame({
        "source_dataset": ["UCI", "SYNTHETIC", "UCI"],
        "product_key": ["A", "B", "C"],
        "horizon": [1, 1, 7],
        "forecast_date": pd.to_datetime(["2011-10-01", "2024-01-01", "2011-11-01"]),
    })
    out = apply_filters(df, dataset="UCI", horizon=1)
    assert len(out) == 1
    assert out.iloc[0]["product_key"] == "A"
    unchanged = apply_filters(df)
    assert len(unchanged) == 3


def test_bi_exports_and_schema():
    build_bi_exports()
    issues = schema_issues()
    assert issues == []
    for name in EXPORT_FILES:
        path = OUTPUTS_BI_DIR / name
        assert path.exists()
        df = pd.read_parquet(path)
        for col in REQUIRED_COLUMNS[name]:
            assert col in df.columns
        assert not df.empty


def test_recommendation_mapping():
    row = pd.Series({
        "stockout_risk_level": "CRITICAL / HIGH",
        "overstock_risk_level": "OPTIMAL",
        "reorder_triggered": False,
        "ending_inventory": 0,
        "risk_matrix_cell": "Stockout Review",
        "sku_id": "SKU_X",
    })
    rec = recommend_row(row)
    assert rec["recommended_review"] == "Review replenishment"
    assert "evidence" in rec and "reason" in rec and "confidence_limitation" in rec
    assert "purchase order" not in rec["recommended_review"].lower()
    frame = pd.DataFrame([{
        "sku_id": "SKU_Y",
        "store_id": "STORE_001",
        "stockout_risk_level": "LOW / SAFE",
        "overstock_risk_level": "OPTIMAL",
        "reorder_triggered": False,
        "risk_matrix_cell": "Normal",
        "growth_class": "Stable",
        "relative_interval_width": 0.1,
    }])
    recs = build_recommendations(frame)
    assert recs["autonomous_decision"].eq(False).all()
    assert recs.iloc[0]["recommended_review"] == "No exceptional intervention indicated"


def test_risk_summary_extract_limitation():
    tables = load_bi_tables()
    inv = tables["inventory_risk.parquet"]
    assert len(inv) == 1000
    assert (inv["extract_note"] == "1000-row reference extract").all()
    assert "risk_matrix_cell" in inv.columns
    kpis = tables["executive_kpis.parquet"].iloc[0]
    assert int(kpis["inventory_n_rows"]) == 1000
    assert int(kpis["inventory_stockout_critical_high"]) == 733
    assert int(kpis["inventory_reorder_review_count"]) > 0


def test_data_freshness_is_snapshot():
    snap = freshness()
    assert "not live" in snap["label"]
    assert snap["data_as_of"] != "NOT AVAILABLE"
    assert snap["monitoring_snapshot"] != "NOT AVAILABLE"
    health = load_bi_tables()["system_health.parquet"].iloc[0]
    assert bool(health["live_data"]) is False
    assert health["uci_h1_hash"] == EXPECTED_UCI_HASH
    assert health["synthetic_h1_hash"] == EXPECTED_SYN_HASH


def test_dashboard_data_loading():
    tables = load_bi_tables()
    assert set(EXPORT_FILES).issubset(tables)
    recs = tables["recommendations.parquet"]
    for col in ("evidence", "reason", "recommended_review", "confidence_limitation"):
        assert col in recs.columns
        assert recs[col].notna().all()
    demand = tables["product_demand.parquet"]
    assert "BAD PRODUCT" not in demand["demand_rank_label"].astype(str).unique()
    assert "LOW DEMAND" in demand["demand_rank_label"].astype(str).unique()


def test_quality_scorecard_not_invented():
    q = quality_scorecard()
    assert q["no_invented_quality_score"] is True
    assert q["schema_validity"] in {"PASS", "FAIL"}
    assert q["n_rows"] == 957949
    assert q["duplicate_rate_pct"] == 0.0
    assert q["missing_required_columns"] == []
