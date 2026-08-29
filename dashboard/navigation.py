"""Phase 23 — Navigation configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavItem:
    key: str
    label: str
    icon: str
    group: str


NAV_GROUPS: list[tuple[str, list[NavItem]]] = [
    ("OVERVIEW", [
        NavItem("home", "Home", "🏠", "OVERVIEW"),
        NavItem("executive", "Executive Dashboard", "📊", "OVERVIEW"),
    ]),
    ("FORECASTING", [
        NavItem("forecasting", "Demand Forecasting", "📈", "FORECASTING"),
        NavItem("forecast_explorer", "Forecast Explorer", "🔮", "FORECASTING"),
        NavItem("horizon_analysis", "Horizon Analysis", "📅", "FORECASTING"),
    ]),
    ("INVENTORY INTELLIGENCE", [
        NavItem("inventory_overview", "Inventory Overview", "📦", "INVENTORY INTELLIGENCE"),
        NavItem("stockout_risk", "Stockout Risk", "⚠️", "INVENTORY INTELLIGENCE"),
        NavItem("overstock_risk", "Overstock Risk", "📦", "INVENTORY INTELLIGENCE"),
        NavItem("recommendations", "Recommendations", "💡", "INVENTORY INTELLIGENCE"),
    ]),
    ("ANALYTICS", [
        NavItem("business_analytics", "Business Analytics", "📊", "ANALYTICS"),
        NavItem("demand_trends", "Demand Trends", "📈", "ANALYTICS"),
        NavItem("sku_analysis", "SKU Analysis", "📦", "ANALYTICS"),
        NavItem("seasonality", "Seasonality Analysis", "📅", "ANALYTICS"),
        NavItem("performance_metrics", "Performance Metrics", "📊", "ANALYTICS"),
    ]),
    ("MACHINE LEARNING", [
        NavItem("model_overview", "Model Overview", "🤖", "MACHINE LEARNING"),
        NavItem("feature_contract", "Feature Contract", "🧠", "MACHINE LEARNING"),
        NavItem("model_performance", "Model Performance", "📊", "MACHINE LEARNING"),
        NavItem("model_explainability", "Model Explainability", "🔍", "MACHINE LEARNING"),
    ]),
    ("MONITORING", [
        NavItem("system_health", "System Health", "🟢", "MONITORING"),
        NavItem("data_quality", "Data Quality", "📊", "MONITORING"),
        NavItem("data_drift", "Data Drift", "📉", "MONITORING"),
        NavItem("prediction_drift", "Prediction Drift", "🔮", "MONITORING"),
        NavItem("alerts", "Alerts", "⚠️", "MONITORING"),
        NavItem("integrity", "Model Integrity", "🔐", "MONITORING"),
    ]),
    ("SYSTEM", [
        NavItem("model_information", "Model Information", "📋", "SYSTEM"),
        NavItem("documentation", "Documentation", "📄", "SYSTEM"),
        NavItem("validation_status", "Validation Status", "🧪", "SYSTEM"),
        NavItem("about", "About Project", "ℹ️", "SYSTEM"),
    ]),
]


def all_nav_items() -> list[NavItem]:
    items: list[NavItem] = []
    for _, group_items in NAV_GROUPS:
        items.extend(group_items)
    return items


def nav_label_map() -> dict[str, str]:
    return {item.key: f"{item.icon} {item.label}" for item in all_nav_items()}
