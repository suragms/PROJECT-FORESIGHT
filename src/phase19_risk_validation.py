"""
Phase 19 — Risk Engine Validation & Stress Tests
==================================================
Validates forecast-driven risk using Phase 19 hardened forecasts.
Creates controlled stress-test scenarios without modifying source data.
"""

import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P17_PROC = os.path.join(BASE, "data", "phase17", "processed")
P19_BT = os.path.join(BASE, "data", "phase19", "backtests")
P19_RISK = os.path.join(BASE, "data", "phase19", "risk")
DOCS = os.path.join(BASE, "docs")
HORIZON_WEEKS = 6  # supported horizon

os.makedirs(P19_RISK, exist_ok=True)


def compute_risk_row(row):
    """Apply Phase 17 risk logic to a single SKU row."""
    on_hand = row.get("on_hand_units", 0)
    on_order = row.get("on_order_units", 0)
    forecast_weekly = row.get("forecast_weekly_demand", 0.1)
    lead_time_weeks = row.get("lead_time_weeks", 2)
    safety_stock = row.get("safety_stock", 0)
    reorder_point = row.get("reorder_point", 0)
    base_price = row.get("base_price", np.nan)
    cost_price = row.get("cost_price", np.nan)

    lead_time_demand = forecast_weekly * lead_time_weeks
    inventory_position = on_hand + on_order
    weeks_of_supply = on_hand / max(forecast_weekly, 0.01)

    stockout_score = 0.0
    if on_hand <= 0:
        stockout_score = 100.0
    elif on_hand < safety_stock:
        stockout_score = 80.0
    elif weeks_of_supply < lead_time_weeks:
        stockout_score = 60.0
    elif inventory_position <= reorder_point:
        stockout_score = 40.0

    stockout_level = "CRITICAL" if stockout_score >= 70 else ("MEDIUM" if stockout_score >= 35 else "LOW")

    forward_demand = forecast_weekly * HORIZON_WEEKS
    excess = max(0, on_hand - forward_demand)
    overstock_score = 80.0 if weeks_of_supply > HORIZON_WEEKS * 2 else (50.0 if weeks_of_supply > HORIZON_WEEKS else 0.0)
    if forecast_weekly < 0.5:
        overstock_score = min(100.0, overstock_score + 20.0)
    overstock_level = "SEVERE" if overstock_score >= 65 else ("MODERATE" if overstock_score >= 30 else "OPTIMAL")

    if stockout_level == "CRITICAL":
        action = "REORDER NOW"
    elif overstock_level == "SEVERE":
        action = "MARKDOWN / CLEAR"
    elif stockout_level == "MEDIUM" or overstock_level == "MODERATE":
        action = "WATCH / VOLATILE"
    else:
        action = "HEALTHY"

    sales_at_risk = (forecast_weekly * lead_time_weeks * base_price
                     if stockout_level in ("CRITICAL", "MEDIUM") and pd.notna(base_price) else np.nan)
    locked_capital = excess * cost_price if pd.notna(cost_price) else np.nan

    return {
        **row,
        "lead_time_demand": lead_time_demand,
        "inventory_position": inventory_position,
        "weeks_of_supply": weeks_of_supply,
        "stockout_risk_score": stockout_score,
        "stockout_risk_level": stockout_level,
        "overstock_risk_score": overstock_score,
        "overstock_risk_level": overstock_level,
        "excess_units": int(excess),
        "action": action,
        "sales_at_risk": sales_at_risk,
        "locked_capital": locked_capital,
    }


def build_stress_scenarios():
    """Controlled synthetic scenarios — do not modify original dataset."""
    base = {
        "forecast_weekly_demand": 100.0,
        "lead_time_weeks": 2,
        "lead_time_days": 14,
        "safety_stock": 50,
        "reorder_point": 200,
        "base_price": 500.0,
        "cost_price": 300.0,
    }
    return [
        {"scenario": "severe_stockout", "on_hand_units": 0, "on_order_units": 0,
         "expected_stockout": "CRITICAL", "expected_action": "REORDER NOW"},
        {"scenario": "moderate_stockout", "on_hand_units": 100, "on_order_units": 50,
         "safety_stock": 50, "forecast_weekly_demand": 100.0,
         "expected_stockout": "MEDIUM", "expected_action": "WATCH / VOLATILE"},
        {"scenario": "healthy_inventory", "on_hand_units": 400, "on_order_units": 0,
         "forecast_weekly_demand": 100.0,
         "expected_stockout": "LOW", "expected_overstock": "OPTIMAL", "expected_action": "HEALTHY"},
        {"scenario": "moderate_overstock", "on_hand_units": 900, "on_order_units": 0,
         "forecast_weekly_demand": 100.0,
         "expected_overstock": "MODERATE", "expected_action": "WATCH / VOLATILE"},
        {"scenario": "severe_overstock", "on_hand_units": 5000, "on_order_units": 0,
         "forecast_weekly_demand": 20.0,
         "expected_overstock": "SEVERE", "expected_action": "MARKDOWN / CLEAR"},
        {"scenario": "high_volatility", "on_hand_units": 5, "on_order_units": 2000,
         "forecast_weekly_demand": 500.0, "safety_stock": 100,
         "expected_stockout": "CRITICAL", "expected_action": "REORDER NOW"},
    ]


def run_risk_validation():
    print("=" * 60)
    print("PHASE 19 — RISK VALIDATION")
    print("=" * 60)

    inv_path = os.path.join(P17_PROC, "synthetic_weekly_inventory.parquet")
    sku_path = os.path.join(P17_PROC, "sku_master.csv")
    bt_path = os.path.join(P19_BT, "backtest_results.parquet")

    inv = pd.read_parquet(inv_path)
    inv["week"] = pd.to_datetime(inv["week"])
    latest_inv = inv[inv["week"] == inv["week"].max()].copy()
    skus = pd.read_csv(sku_path)

    # Phase 19 forecast demand
    bt = pd.read_parquet(bt_path)
    sku_forecast = bt.groupby("product_key").agg(
        forecast_weekly_demand=("phase19_forecast", "mean"),
    ).reset_index()
    sku_forecast["sku_id"] = sku_forecast["product_key"].str.replace("SYN_", "", regex=False)

    inv_sku = latest_inv.groupby("sku_id").agg(
        on_hand_units=("ending_inventory", "sum"),
        on_order_units=("on_order_qty", "sum"),
    ).reset_index()

    sku_cols = ["sku_id", "lead_time_days", "reorder_point", "safety_stock", "cost_price", "base_price"]
    risk_df = inv_sku.merge(skus[[c for c in sku_cols if c in skus.columns]], on="sku_id", how="left")
    risk_df = risk_df.merge(sku_forecast[["sku_id", "forecast_weekly_demand"]], on="sku_id", how="left")
    risk_df["forecast_weekly_demand"] = risk_df["forecast_weekly_demand"].fillna(0.1)
    risk_df["lead_time_weeks"] = np.ceil(risk_df["lead_time_days"] / 7).astype(int)

    risk_rows = [compute_risk_row(row.to_dict()) for _, row in risk_df.iterrows()]
    risk_out = pd.DataFrame(risk_rows)
    risk_path = os.path.join(P19_RISK, "forecast_driven_risk.parquet")
    risk_out.to_parquet(risk_path, index=False)

    # Stress tests
    stress_results = []
    for scenario in build_stress_scenarios():
        row = {**scenario, **{k: v for k, v in scenario.items() if k not in ("scenario", "expected_stockout",
              "expected_overstock", "expected_action")}}
        result = compute_risk_row({**base_defaults_from_scenario(scenario), **row})
        passed = True
        if "expected_stockout" in scenario:
            passed = passed and (result["stockout_risk_level"] == scenario["expected_stockout"])
        if "expected_overstock" in scenario:
            passed = passed and (result["overstock_risk_level"] == scenario["expected_overstock"])
        if "expected_action" in scenario:
            passed = passed and (result["action"] == scenario["expected_action"])
        stress_results.append({
            "scenario": scenario["scenario"],
            "stockout_level": result["stockout_risk_level"],
            "overstock_level": result["overstock_risk_level"],
            "action": result["action"],
            "expected_action": scenario.get("expected_action"),
            "pass": passed,
            "explanation": f"on_hand={scenario.get('on_hand_units')}, forecast={scenario.get('forecast_weekly_demand', 100)}, "
                           f"lead_time_demand={result['lead_time_demand']:.0f}",
        })

    stress_path = os.path.join(P19_RISK, "stress_test_results.json")
    with open(stress_path, "w") as f:
        json.dump(stress_results, f, indent=2)

    # Decision grid validation
    reorder_ok = set(risk_out[risk_out["action"] == "REORDER NOW"]["stockout_risk_level"]) == {"CRITICAL"}
    healthy_ok = (
        risk_out[risk_out["action"] == "HEALTHY"]["stockout_risk_level"].isin(["LOW"]).all() and
        risk_out[risk_out["action"] == "HEALTHY"]["overstock_risk_level"].isin(["OPTIMAL"]).all()
        if (risk_out["action"] == "HEALTHY").any() else True
    )
    wos_ok = np.allclose(
        risk_out["weeks_of_supply"].fillna(0),
        risk_out["on_hand_units"] / np.maximum(risk_out["forecast_weekly_demand"], 0.01),
        rtol=0.01,
    )

    summary = {
        "demand_source": "PHASE19_FORECAST",
        "total_skus": len(risk_out),
        "reorder_now": int((risk_out["action"] == "REORDER NOW").sum()),
        "markdown_clear": int((risk_out["action"] == "MARKDOWN / CLEAR").sum()),
        "watch_volatile": int((risk_out["action"] == "WATCH / VOLATILE").sum()),
        "healthy": int((risk_out["action"] == "HEALTHY").sum()),
        "decision_grid_reorder_pass": reorder_ok,
        "decision_grid_healthy_pass": healthy_ok,
        "wos_consistency_pass": wos_ok,
        "stress_tests_pass": all(s["pass"] for s in stress_results),
        "stress_test_details": stress_results,
        "total_sales_at_risk": round(float(risk_out["sales_at_risk"].sum()), 2) if risk_out["sales_at_risk"].notna().any() else None,
        "total_locked_capital": round(float(risk_out["locked_capital"].sum()), 2) if risk_out["locked_capital"].notna().any() else None,
        "uci_risk_status": "NOT_AVAILABLE",
    }

    summary_path = os.path.join(P19_RISK, "risk_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Sample critical SKUs for documentation
    critical = risk_out[risk_out["stockout_risk_level"] == "CRITICAL"].head(5)
    sample = critical[["sku_id", "forecast_weekly_demand", "lead_time_demand", "on_hand_units",
                       "on_order_units", "stockout_risk_level", "action", "sales_at_risk"]].to_dict(orient="records")

    # Risk validation markdown
    md = os.path.join(DOCS, "phase19_risk_validation.md")
    lines = [
        "# Phase 19 — Risk Validation\n",
        f"**Demand source:** {summary['demand_source']}\n",
        f"**Supported horizon:** {HORIZON_WEEKS} weeks\n",
        "\n## Decision Grid Validation\n",
        f"- REORDER NOW -> CRITICAL stockout: {'PASS' if reorder_ok else 'FAIL'}\n",
        f"- HEALTHY -> LOW/OPTIMAL: {'PASS' if healthy_ok else 'FAIL'}\n",
        f"- WoS consistency: {'PASS' if wos_ok else 'FAIL'}\n",
        "\n## Stress Tests\n",
        "| Scenario | Stockout | Overstock | Action | Expected | Pass |",
        "|----------|----------|-----------|--------|----------|------|",
    ]
    for s in stress_results:
        lines.append(
            f"| {s['scenario']} | {s['stockout_level']} | {s['overstock_level']} | "
            f"{s['action']} | {s['expected_action']} | {s['pass']} |"
        )
    lines += [
        "\n## Sample Critical SKUs\n",
        "```json\n" + json.dumps(sample, indent=2, default=str) + "\n```\n",
        f"\n## Rupee Impact\n",
        f"- Sales at risk: {summary['total_sales_at_risk']}\n",
        f"- Locked capital: {summary['total_locked_capital']}\n",
    ]
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Risk matrix: {len(risk_out)} SKUs")
    print(f"Stress tests: {sum(s['pass'] for s in stress_results)}/{len(stress_results)} PASS")
    return summary


def base_defaults_from_scenario(scenario):
    return {
        "forecast_weekly_demand": scenario.get("forecast_weekly_demand", 100.0),
        "lead_time_weeks": 2,
        "safety_stock": scenario.get("safety_stock", 50),
        "reorder_point": 200,
        "base_price": 500.0,
        "cost_price": 300.0,
    }


if __name__ == "__main__":
    run_risk_validation()
