"""Documented BI classification rules. Does not retrain models."""

from __future__ import annotations

# Growth uses held-out TEST actuals split at the median forecast_date.
# Thresholds are explicit and conservative.
GROWTH_MIN_OBS = 10
GROWTH_STABLE_ABS = 0.05  # |rate| < 5% => Stable

GROWTH_LABELS = ("Growing", "Stable", "Declining", "Insufficient Evidence")

UNCERTAINTY_REL_WIDTH = 0.50  # (P90-P10)/|pred| above this => review uncertainty

DEMAND_HIGH_QUANTILE = 0.50  # median split on avg_daily_demand within the extract
INVENTORY_HIGH_QUANTILE = 0.50  # median split on ending_inventory within the extract
# Strict '>' so a zero-inflated median (ending_inventory median is 0 on the extract)
# does not classify every row as inventory-high.


def classify_growth(rate: float | None, n_hist: int, n_recent: int) -> str:
    if rate is None or n_hist < GROWTH_MIN_OBS or n_recent < GROWTH_MIN_OBS:
        return "Insufficient Evidence"
    if abs(rate) < GROWTH_STABLE_ABS:
        return "Stable"
    if rate >= GROWTH_STABLE_ABS:
        return "Growing"
    return "Declining"


def demand_inventory_cell(demand_high: bool, inventory_high: bool) -> str:
    if (not demand_high) and (not inventory_high):
        return "Normal"
    if demand_high and (not inventory_high):
        return "Stockout Review"
    if (not demand_high) and inventory_high:
        return "Overstock Review"
    return "Critical Review"
