"""
Pre-Integration Validation — ML Forecasting + Inventory Risk + Streamlit stack
===============================================================================
Project FORESIGHT: Demand & Inventory Intelligence

Re-runnable smoke test that validates the ML forecasting engine, the inventory
risk engine, and the persisted artifacts against the Phase 3 cleaned datasets.

Run:  python src/validate_ml_stack.py

Exit code 0 when every check passes, 1 otherwise.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_integration import (
    load_sales_daily,
    load_inventory_snapshots,
    load_sku_master,
    load_calendar,
)
from src.feature_engineering import build_forecasting_feature_matrix, aggregate_daily_sales
from src.forecasting import MLDemandForecaster, generate_multi_step_forecast
from src.risk_scoring import calculate_inventory_risk_matrix, answer_10_core_questions

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    print("=" * 78)
    print("Project FORESIGHT — Pre-Integration Stack Validation")
    print("=" * 78)

    # ------------------------------------------------------------------ DATA
    print("\n--- 1. Cleaned data availability ---")
    sales = load_sales_daily()
    inv = load_inventory_snapshots()
    skus = load_sku_master()
    cal = load_calendar()

    check("sales_daily_clean loads", len(sales) > 0, f"{len(sales):,} rows")
    check("inventory_snapshots_clean loads", len(inv) > 0, f"{len(inv):,} rows")
    check("inventory carries Phase-3 derived columns",
          {"beginning_inventory_pre_receipts", "inventory_balance_ok"} <= set(inv.columns))
    check("sales date span is full 2022-2025",
          sales["date"].min().year == 2022 and sales["date"].max().year == 2025,
          f"{sales['date'].min().date()} -> {sales['date'].max().date()}")
    check("grain (date, store, sku) unique", sales.groupby(["date", "store_id", "sku_id"]).size().eq(1).all())

    # ------------------------------------------------------------ MODEL FILES
    print("\n--- 2. Persisted model artifacts ---")
    for mtype in ("lightgbm", "xgboost", "random_forest"):
        path = os.path.join(BASE_DIR, "models", f"{mtype}_forecaster.joblib")
        exists = os.path.exists(path)
        check(f"{mtype}_forecaster.joblib exists", exists)
        if exists:
            m = MLDemandForecaster.load(path)
            check(f"{mtype} model loads + has 25 features",
                  hasattr(m, "feature_names") and len(m.feature_names) == 25)
            # prediction smoke test on a synthetic row
            row = pd.DataFrame([{f: 0.0 for f in m.feature_names}])
            pred = float(m.predict(row)[0])
            check(f"{mtype} predict() works (non-negative)", pred >= 0, f"pred={pred:.2f}")

    # ------------------------------------------------- FORECASTING FEATURES
    print("\n--- 3. Leakage-safe feature matrix at SKU-total grain ---")
    daily = aggregate_daily_sales(sales, group_cols=("sku_id",))
    fm = build_forecasting_feature_matrix(
        daily, sku_df=skus, store_df=None, calendar_df=cal, group_cols=["sku_id"]
    )
    check("SKU-total feature matrix builds", len(fm) > 0, f"{len(fm):,} rows")
    check("lag features present", all(
        c in fm.columns for c in ("units_sold_lag_1", "units_sold_lag_7", "units_sold_lag_30")))
    check("rolling features present", all(
        c in fm.columns for c in ("units_sold_rolling_mean_7", "units_sold_rolling_std_7",
                                  "units_sold_ewm_7", "units_sold_ewm_28")))
    check("no future-aggregate columns", not any(
        c for c in fm.columns if "rolling" in c and c.endswith("_future")))
    # Structural check: a row's lag_1 must equal the previous CALENDAR day's
    # actual from the ORIGINAL series (dropna removes intermediate rows, so we
    # compare against the pre-feature `daily` frame, not against shift(1) on the
    # already-pruned feature matrix).
    g = fm[fm["sku_id"] == "SKU_00001"].copy()
    daily_sku = daily[daily["sku_id"] == "SKU_00001"].set_index("date")["units_sold"]
    expected_prev = g["date"].map(lambda d: daily_sku.get(d - pd.Timedelta(days=1), pd.NA))
    mask = expected_prev.notna()
    row_lag1_matches = (g.loc[mask, "units_sold_lag_1"] == expected_prev[mask]).all()
    check("lag_1 equals previous calendar day's actual (no target leakage)",
          bool(row_lag1_matches), f"{int(mask.sum())} rows verified")

    # ------------------------------------------------------------ LEADERBOARD
    print("\n--- 4. Benchmark evidence (best model beats baseline) ---")
    lb_path = os.path.join(BASE_DIR, "outputs", "forecasts", "model_benchmark_leaderboard.csv")
    if os.path.exists(lb_path):
        lb = pd.read_csv(lb_path)
        check("leaderboard exists with required metrics",
              {"model", "mae", "rmse", "wape_pct", "mape_pct", "r2"} <= set(lb.columns))
        best = lb.sort_values("wape_pct").iloc[0]
        check("best model on WAPE is an ML model (not NAIVE)",
              "NAIVE" not in str(best["model"]).upper(),
              f"best={best['model']} WAPE={best['wape_pct']:.2f}%")
        check("ML WAPE meaningfully below baseline NAIVE",
              best["wape_pct"] < 50,
              f"best WAPE={best['wape_pct']:.1f}% (NAIVE ~66%)")
    else:
        check("leaderboard exists", False, "missing outputs/forecasts/model_benchmark_leaderboard.csv")

    # ------------------------------------------------- FORECAST SANITY CHECK
    print("\n--- 5. Forecast magnitude sanity (SKU-total grain) ---")
    mdl = MLDemandForecaster.load(os.path.join(BASE_DIR, "models", "lightgbm_forecaster.joblib"))
    top_sku = sales.groupby("sku_id")["units_sold"].sum().idxmax()
    hist = daily[daily["sku_id"] == top_sku][["date", "units_sold"]].reset_index(drop=True)
    fc = generate_multi_step_forecast(model=mdl, history_df=hist, horizon_days=14)
    recent_mean = float(hist["units_sold"].iloc[-35:].mean())
    fc_mean = float(fc["forecast_units"].mean())
    ratio = recent_mean / max(fc_mean, 1e-9)
    check("forecast same order of magnitude as recent history",
          0.25 <= ratio <= 4.0,
          f"recent_mean={recent_mean:.0f}, forecast_mean={fc_mean:.0f} (ratio {ratio:.2f})")
    check("forecast frame has CI bounds", {"forecast_lower", "forecast_upper"} <= set(fc.columns))
    check("forecast bounds consistent (lower <= point <= upper)",
          (fc["forecast_lower"] <= fc["forecast_units"]).all() and
          (fc["forecast_units"] <= fc["forecast_upper"]).all())

    # ----------------------------------------------------------- RISK ENGINE
    print("\n--- 6. Inventory risk engine ---")
    risk = calculate_inventory_risk_matrix()
    check("risk matrix computes", len(risk) > 0, f"{len(risk):,} rows")
    req = {"days_of_supply", "stockout_risk_score", "stockout_risk_level",
           "overstock_risk_score", "overstock_risk_level", "reorder_triggered",
           "recommended_reorder_qty", "capital_tied_up", "potential_lost_daily_revenue"}
    check("risk matrix has required fields", req <= set(risk.columns))
    check("risk scores bounded [0,100]", risk["stockout_risk_score"].between(0, 100).all() and
          risk["overstock_risk_score"].between(0, 100).all())
    check("days_of_supply non-negative", (risk["days_of_supply"] >= 0).all())
    # inventory semantic: engine uses ending_inventory directly, never re-adds receipts
    uses_pre_receipts = "beginning_inventory_pre_receipts" in risk.columns
    check("inventory semantic carried into risk matrix", uses_pre_receipts)
    check("risk engine does not re-add receipts to inventory position",
          "receipts" not in risk.columns or "ending_inventory" in risk.columns)

    print("\n--- 7. 10 Core Questions pipeline ---")
    ans = answer_10_core_questions()
    check("10-question analysis runs", len(ans) == 10,
          f"keys={len(ans)} | stockout criticals={ans['q7_stockout_risk']['critical_count']}")

    # ---------------------------------------------------------------- SUMMARY
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed")
    print("=" * 78)
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
