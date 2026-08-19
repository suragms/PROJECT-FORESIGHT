"""Decision-support recommendations from existing risk and forecast fields."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.bi.rules import UNCERTAINTY_REL_WIDTH, demand_inventory_cell


def recommend_row(row: pd.Series) -> dict[str, Any]:
    stockout = str(row.get("stockout_risk_level", "NOT AVAILABLE"))
    overstock = str(row.get("overstock_risk_level", "NOT AVAILABLE"))
    reorder = row.get("reorder_triggered")
    cell = row.get("risk_matrix_cell", "NOT AVAILABLE")
    rel_width = row.get("relative_interval_width")

    if stockout == "CRITICAL / HIGH":
        return {
            "priority": 1,
            "recommended_review": "Review replenishment",
            "reason": "Existing stockout_risk_level is CRITICAL / HIGH.",
            "evidence": f"stockout_risk_level={stockout}; ending_inventory={row.get('ending_inventory', 'NOT AVAILABLE')}",
            "confidence_limitation": "1000-row reference extract; not an automatic purchase order.",
        }
    if overstock in {"SEVERE OVERSTOCK", "MODERATE OVERSTOCK"}:
        return {
            "priority": 2,
            "recommended_review": "Review inventory exposure",
            "reason": f"Existing overstock_risk_level is {overstock}.",
            "evidence": f"overstock_risk_level={overstock}; days_of_supply={row.get('days_of_supply', 'NOT AVAILABLE')}",
            "confidence_limitation": "Reference scoring only; no inventory write-back.",
        }
    if pd.notna(reorder) and bool(reorder):
        return {
            "priority": 2,
            "recommended_review": "Review replenishment",
            "reason": "Existing reorder_triggered flag is true.",
            "evidence": f"reorder_triggered={reorder}; reorder_point={row.get('reorder_point', 'NOT AVAILABLE')}",
            "confidence_limitation": "Flag is analytical; no PO is sent.",
        }
    if rel_width is not None and pd.notna(rel_width) and float(rel_width) >= UNCERTAINTY_REL_WIDTH:
        return {
            "priority": 3,
            "recommended_review": "Review forecast uncertainty",
            "reason": "Relative P10/P90 width meets the documented uncertainty threshold.",
            "evidence": f"relative_interval_width={float(rel_width):.4f}; threshold={UNCERTAINTY_REL_WIDTH}",
            "confidence_limitation": "Interval companions are not guaranteed coverage bands.",
        }
    if str(row.get("growth_class")) == "Growing":
        return {
            "priority": 3,
            "recommended_review": "Monitor demand growth",
            "reason": "Documented growth rule classified TEST actuals as Growing.",
            "evidence": f"growth_class=Growing; growth_rate={row.get('growth_rate', 'NOT AVAILABLE')}",
            "confidence_limitation": "Held-out TEST split, not live sales velocity.",
        }
    if cell == "Critical Review":
        return {
            "priority": 2,
            "recommended_review": "Review replenishment",
            "reason": "Demand-high and inventory-high median split (documented 2x2).",
            "evidence": f"risk_matrix_cell={cell}",
            "confidence_limitation": "Median split on the extract, not a new risk model.",
        }
    return {
        "priority": 4,
        "recommended_review": "No exceptional intervention indicated",
        "reason": "No critical stockout, overstock, reorder, or uncertainty flag on this row.",
        "evidence": f"stockout={stockout}; overstock={overstock}; cell={cell}",
        "confidence_limitation": "Absence of a flag is not proof of operational safety.",
    }


def build_recommendations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        rec = recommend_row(row)
        rec.update({
            "sku_id": row.get("sku_id", "NOT AVAILABLE"),
            "store_id": row.get("store_id", "NOT AVAILABLE"),
            "product_key": row.get("product_key", row.get("sku_id", "NOT AVAILABLE")),
            "autonomous_decision": False,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def risk_matrix_cell(demand_high: bool, inventory_high: bool) -> str:
    return demand_inventory_cell(demand_high, inventory_high)
