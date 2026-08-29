"""Phase 23.5 — Professional sidebar navigation configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavItem:
    key: str
    label: str
    icon: str
    group: str


# Only real unified-app routes (see app.py). No invented pages.
NAV_GROUPS: list[tuple[str, list[NavItem]]] = [
    ("OVERVIEW", [
        NavItem("home", "Home", "🏠", "OVERVIEW"),
        NavItem("executive", "Executive Dashboard", "📊", "OVERVIEW"),
    ]),
    ("ANALYTICS", [
        NavItem("business_analytics", "Sales Analytics", "📈", "ANALYTICS"),
        NavItem("demand_trends", "Demand Trends", "📉", "ANALYTICS"),
        NavItem("sku_analysis", "Product Performance", "📦", "ANALYTICS"),
        NavItem("seasonality", "Seasonality", "📅", "ANALYTICS"),
        NavItem("performance_metrics", "Performance Metrics", "📊", "ANALYTICS"),
    ]),
    ("INVENTORY & RISK", [
        NavItem("inventory_overview", "Inventory Dashboard", "📦", "INVENTORY & RISK"),
        NavItem("stockout_risk", "Stockout Risk", "⚠️", "INVENTORY & RISK"),
        NavItem("overstock_risk", "Overstock Risk", "📚", "INVENTORY & RISK"),
        NavItem("recommendations", "Recommendations", "💡", "INVENTORY & RISK"),
    ]),
    ("FORECASTING", [
        NavItem("forecasting", "Demand Forecasting", "🔮", "FORECASTING"),
        NavItem("forecast_explorer", "Forecast Explorer", "📈", "FORECASTING"),
        NavItem("horizon_analysis", "Horizon Analysis", "🗓️", "FORECASTING"),
    ]),
    ("MACHINE LEARNING", [
        NavItem("model_overview", "ML Performance", "🤖", "MACHINE LEARNING"),
        NavItem("feature_contract", "Feature Contract", "🧠", "MACHINE LEARNING"),
        NavItem("model_performance", "Model Metrics", "📊", "MACHINE LEARNING"),
        NavItem("model_explainability", "Explainability", "🔍", "MACHINE LEARNING"),
    ]),
    ("PRODUCTION", [
        NavItem("system_health", "System Health", "🟢", "PRODUCTION"),
        NavItem("data_quality", "Data Quality", "🧪", "PRODUCTION"),
        NavItem("data_drift", "Data Drift", "📉", "PRODUCTION"),
        NavItem("prediction_drift", "Prediction Drift", "🔮", "PRODUCTION"),
        NavItem("alerts", "Alerts", "⚠️", "PRODUCTION"),
        NavItem("integrity", "Model Integrity", "❤️", "PRODUCTION"),
    ]),
    ("SYSTEM", [
        NavItem("model_information", "Model Information", "ℹ️", "SYSTEM"),
        NavItem("documentation", "Documentation", "📄", "SYSTEM"),
        NavItem("validation_status", "Validation Status", "🧪", "SYSTEM"),
        NavItem("about", "About", "⚙️", "SYSTEM"),
    ]),
]


def all_nav_items() -> list[NavItem]:
    items: list[NavItem] = []
    for _, group_items in NAV_GROUPS:
        items.extend(group_items)
    return items


def nav_label_map() -> dict[str, str]:
    return {item.key: f"{item.icon}  {item.label}" for item in all_nav_items()}


ADMIN_ONLY_PAGES = frozenset()  # All authenticated users may access every page


def filtered_nav_groups(role: str) -> list[tuple[str, list[NavItem]]]:
    return NAV_GROUPS
