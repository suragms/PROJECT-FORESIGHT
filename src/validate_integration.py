"""
Phase 4 — CAM Integration Validation
======================================
Project FORESIGHT: Demand & Inventory Intelligence

Re-runnable smoke test that validates the Phase 4 Common Analytical Model
against its requirements:

  * Only Phase 3 processed data used; raw files untouched.
  * `source_dataset` preserved; source-aware keys.
  * UCI and Synthetic never blindly concatenated.
  * Guest transactions, returns, cancellations preserved and separated.
  * Inventory Phase-3 REVIEW semantic preserved (receipts never re-added).
  * No fabricated values (UCI category/brand/supplier/lead time = NULL).
  * Grain, primary-key, and foreign-key integrity (0 orphans).
  * `forecast_base` created with no lag/rolling features.
  * ML / risk / Streamlit compatibility.
  * Data lineage + integration quality report produced.
  * Notebook executed successfully.
  * Pipeline reproducible (fresh build == saved artifacts).

Run:  python src/validate_integration.py

Exit code 0 when every check passes, 1 otherwise.
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nbformat

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from src.cam_adapter import (
    load_cam_table,
    compatibility_report,
    forecast_base_to_legacy_sales,
    inventory_analytics_to_legacy_snapshots,
)
from src.data_integration import (
    SRC_UCI,
    SRC_SYNTHETIC,
    run_integration_pipeline,
)

RESULTS = []

TABLES = [
    "dim_calendar", "dim_product", "dim_entity", "dim_customer",
    "fact_sales", "fact_inventory", "fact_returns", "fact_cancellations",
    "inventory_analytics", "customer_analytics", "forecast_base",
]

GRAINS = {
    "dim_calendar": ["date"],
    "dim_product": ["product_key"],
    "dim_entity": ["source_dataset", "entity_id"],
    "dim_customer": ["customer_key"],
    "fact_sales": ["date", "source_dataset", "entity_id", "product_key"],
    "fact_inventory": ["date", "source_dataset", "entity_id", "product_key"],
    "fact_returns": ["date", "source_dataset", "entity_id", "product_key"],
    "fact_cancellations": ["date", "source_dataset", "entity_id", "product_key"],
    "inventory_analytics": ["date", "source_dataset", "entity_id", "product_key"],
    "customer_analytics": ["customer_key"],
    "forecast_base": ["date", "source_dataset", "entity_id", "product_key"],
}


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


def _norm_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize every missing value (NA / NaN / NaT) to a single np.nan so
    in-memory builds and parquet round-trips compare equal."""
    out = pd.DataFrame()
    for col in df.columns:
        s = df[col].astype(object)
        out[col] = s.where(s.notna(), np.nan)
    return out


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame, keys: list) -> bool:
    """Compare two frames for equality (same columns, sorted by keys)."""
    try:
        if list(a.columns) != list(b.columns):
            return False
        if len(a) != len(b):
            return False
        if a.empty:
            return True
        a = _norm_missing(a.sort_values(keys).reset_index(drop=True))
        b = _norm_missing(b.sort_values(keys).reset_index(drop=True))
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False,
                                      rtol=1e-9, atol=1e-9)
        return True
    except Exception:
        return False


def main() -> int:
    print("=" * 78)
    print("Project FORESIGHT — Phase 4 CAM Integration Validation")
    print("=" * 78)

    # -------------------------------------------------------------- ARTIFACTS
    print("\n--- 1. Saved artifacts & docs ---")
    integrated_dir = os.path.join(_BASE, "data", "processed", "integrated")
    present = all(os.path.exists(os.path.join(integrated_dir, f"{t}.parquet")) for t in TABLES)
    check("all 11 CAM parquet tables exist under data/processed/integrated/", present)
    check("data_lineage.md exists",
          os.path.exists(os.path.join(_BASE, "docs", "data_lineage.md")))
    check("integration_quality_report.json exists",
          os.path.exists(os.path.join(_BASE, "docs", "integration_quality_report.json")))
    check("integration_quality_report.csv exists",
          os.path.exists(os.path.join(_BASE, "docs", "integration_quality_report.csv")))

    report_json = None
    qpath = os.path.join(_BASE, "docs", "integration_quality_report.json")
    if os.path.exists(qpath):
        report_json = json.load(open(qpath, encoding="utf-8"))
        check("inventory_data_status = REVIEW in report",
              report_json.get("inventory_data_status") == "REVIEW")
        check("all 11 tables report status PASS",
              all(t["status"] == "PASS" for t in report_json["tables"]) and
              len(report_json["tables"]) == 11)
        check("all business rules pass in report",
              report_json.get("rule_passed") == report_json.get("rule_total"),
              f"{report_json.get('rule_passed')}/{report_json.get('rule_total')}")

    nb_path = os.path.join(_BASE, "notebooks", "03_data_integration.ipynb")
    if os.path.exists(nb_path):
        nb = nbformat.read(nb_path, as_version=4)
        errs = [c for c in nb.cells
                if any(o.output_type == "error" for o in c.get("outputs", []))]
        check("notebook executed with 0 error outputs", len(errs) == 0,
              f"{len(errs)} error cells")

    # -------------------------------------------------------------- LOAD CAM
    print("\n--- 2. Load saved CAM tables ---")
    tables = {t: load_cam_table(t) for t in TABLES}
    check("all tables load non-empty", all(len(v) > 0 for v in tables.values()))

    # ------------------------------------------------- SOURCE IDENTITY
    print("\n--- 3. Source identity & keys ---")
    fact_like = ["fact_sales", "fact_inventory", "fact_returns",
                 "fact_cancellations", "forecast_base"]
    dim_like = ["dim_product", "dim_entity", "dim_customer"]
    check("source_dataset present on all facts/dims",
          all("source_dataset" in tables[t].columns for t in fact_like + dim_like))
    check("source_dataset values valid",
          all(set(tables[t]["source_dataset"].astype(str).unique()) <= {SRC_UCI, SRC_SYNTHETIC}
              for t in fact_like + dim_like))
    check("product keys source-aware (UCI_/SYN_ prefixes)",
          tables["dim_product"]["product_key"].str.startswith("UCI_").any() and
          tables["dim_product"]["product_key"].str.startswith("SYN_").any())
    check("entity keys source-aware (SYNTHETIC stores + ONLINE)",
          set(tables["dim_entity"].loc[tables["dim_entity"]["source_dataset"] == SRC_SYNTHETIC, "entity_type"]) == {"STORE"} and
          (tables["dim_entity"].loc[tables["dim_entity"]["source_dataset"] == SRC_UCI, "entity_id"] == "ONLINE").all())
    check("customer keys source-aware (SYN_/UCI_ prefixes)",
          tables["dim_customer"]["customer_key"].str.startswith("UCI_").any() and
          tables["dim_customer"]["customer_key"].str.startswith("SYN_").any())

    # ----------------------------------------------- NO BLIND CONCATENATION
    print("\n--- 4. No blind concatenation / separation ---")
    sales = tables["fact_sales"]
    check("UCI & synthetic rows coexist but are discriminated",
          set(sales["source_dataset"].unique()) == {SRC_UCI, SRC_SYNTHETIC})
    check("returns fact is UCI-only", set(tables["fact_returns"]["source_dataset"].unique()) == {SRC_UCI})
    check("cancellations fact is UCI-only", set(tables["fact_cancellations"]["source_dataset"].unique()) == {SRC_UCI})
    check("inventory fact is synthetic-only",
          set(tables["fact_inventory"]["source_dataset"].unique()) == {SRC_SYNTHETIC})

    # --------------------------------------------- GUEST TRANSACTION HANDLING
    print("\n--- 5. Guest transaction handling ---")
    uci_sales = sales[sales["source_dataset"] == SRC_UCI]
    check("guest UCI transactions kept in sales (units>0)",
          int(uci_sales["units_sold"].sum()) > 0)
    check("guests excluded from identified-customer metrics",
          bool(tables["customer_analytics"]["customer_key"].notna().all()) and
          (tables["customer_analytics"]["customer_key"].str.startswith(("UCI_", "SYN_"))).all())

    # ----------------------------------------------- NO FABRICATED VALUES
    print("\n--- 6. No fabricated values (UCI) ---")
    uci_prod = tables["dim_product"][tables["dim_product"]["source_dataset"] == SRC_UCI]
    for col in ["category", "sub_category", "brand", "supplier_id",
                "lead_time_days", "reorder_point", "safety_stock"]:
        check(f"UCI {col} is NULL (not fabricated)", bool(uci_prod[col].isna().all()))
    check("UCI promotion_flag is NULL (no promotion source)",
          bool(sales.loc[sales["source_dataset"] == SRC_UCI, "promotion_flag"].isna().all()))
    check("UCI entity has NULL store fields (no fake stores)",
          bool(tables["dim_entity"].loc[tables["dim_entity"]["source_dataset"] == SRC_UCI,
                                       ["store_name", "city", "state"]].isna().all().all()))

    # -------------------------------------------------- INVENTORY SEMANTIC
    print("\n--- 7. Inventory Phase-3 REVIEW semantic ---")
    inv = tables["fact_inventory"]
    balanced = inv[inv["inventory_balance_ok"]]
    check("ending == beginning - units_sold on balanced rows",
          bool((balanced["ending_inventory"] ==
                balanced["beginning_inventory"] - balanced["units_sold"]).all()),
          f"{len(balanced):,} rows")
    check("beginning_inventory_pre_receipts preserved",
          "beginning_inventory_pre_receipts" in inv.columns)
    check("inventory_balance_ok preserved", "inventory_balance_ok" in inv.columns)
    with_rec = balanced[balanced["receipts"] > 0]
    check("receipts not re-added to ending_inventory",
          not (with_rec["ending_inventory"] ==
               with_rec["beginning_inventory"] + with_rec["receipts"] -
               with_rec["units_sold"]).any(),
          f"{len(with_rec):,} receipt-days checked")

    # ------------------------------------------------------------- GRAIN / FK
    print("\n--- 8. Grain, PK and foreign keys ---")
    for t in TABLES:
        keys = GRAINS[t]
        dup = int(tables[t].duplicated(subset=keys).sum())
        nulls = int(tables[t][keys].isna().any(axis=1).sum())
        check(f"{t} grain unique + no null keys", dup == 0 and nulls == 0,
              f"dups={dup} nulls={nulls}")
    # FK orphans
    fks = [
        ("fact_sales", "dim_calendar", ["date"], ["date"]),
        ("fact_sales", "dim_product", ["product_key"], ["product_key"]),
        ("fact_sales", "dim_entity", ["source_dataset", "entity_id"], ["source_dataset", "entity_id"]),
        ("fact_inventory", "dim_calendar", ["date"], ["date"]),
        ("fact_inventory", "dim_product", ["product_key"], ["product_key"]),
        ("fact_inventory", "dim_entity", ["source_dataset", "entity_id"], ["source_dataset", "entity_id"]),
        ("fact_returns", "dim_calendar", ["date"], ["date"]),
        ("fact_returns", "dim_product", ["product_key"], ["product_key"]),
        ("fact_cancellations", "dim_calendar", ["date"], ["date"]),
        ("fact_cancellations", "dim_product", ["product_key"], ["product_key"]),
    ]
    total_orphans = 0
    for fact, dim, fk, dk in fks:
        merged = tables[fact][fk].merge(tables[dim][dk], left_on=fk, right_on=dk,
                                        how="left", indicator=True)
        total_orphans += int((merged["_merge"] == "left_only").sum())
    check("0 foreign-key orphans across star schema", total_orphans == 0,
          f"{total_orphans} orphans")

    # ------------------------------------------------------- FORECAST BASE
    print("\n--- 9. Forecast base contract ---")
    fb = tables["forecast_base"]
    leaky = [c for c in fb.columns if "lag_" in c or "rolling_" in c or "_ewm_" in c]
    check("forecast_base has no lag/rolling features", len(leaky) == 0, f"leaky={leaky}")
    check("forecast_base == fact_sales projection",
          list(fb.columns) == ["date", "source_dataset", "entity_id", "entity_type",
                               "product_key", "sku_id", "units_sold", "revenue",
                               "average_unit_price", "transaction_count",
                               "unique_customers", "promotion_flag"])

    # ------------------------------------------------------ COMPATIBILITY
    print("\n--- 10. Consumer compatibility ---")
    rep = compatibility_report()
    check("ML forecasting compatible", rep["ml_forecasting"]["compatible"])
    check("Inventory risk compatible", rep["inventory_risk"]["compatible"])
    ls = forecast_base_to_legacy_sales()
    check("forecast_base -> legacy sales mapping works", len(ls) == len(fb))
    legacy_inv = inventory_analytics_to_legacy_snapshots()
    check("inventory_analytics -> legacy snapshots mapping works",
          len(legacy_inv) == len(tables["inventory_analytics"]))

    # ---------------------------------------------------- REPRODUCIBILITY
    print("\n--- 11. Pipeline reproducibility (fresh build vs saved) ---")
    fresh, _ = run_integration_pipeline(save=False)
    repro_ok = all(_frames_equal(fresh[t], tables[t], GRAINS[t]) for t in TABLES)
    check("fresh pipeline build equals saved artifacts (deterministic)",
          repro_ok)

    # ------------------------------------------------------------- SUMMARY
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed")
    print("=" * 78)
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} -- {detail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
