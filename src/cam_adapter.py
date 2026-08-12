"""
Phase 4 — CAM Compatibility Adapter
=====================================
Project FORESIGHT: Demand & Inventory Intelligence

Thin, read-only bridge between the **Common Analytical Model** (the
source-aware star schema in ``data/processed/integrated/``) and the existing
legacy consumer schemas:

* **ML Forecasting Engine** — ``src/forecasting.py`` + ``src/feature_engineering.py``
  consume a legacy *sales* frame (``total_revenue`` / ``avg_unit_price``
  naming) and a legacy *SKU master*.
* **Inventory Risk Engine** — ``src/risk_scoring.py`` consumes legacy
  *inventory snapshot* columns (``store_id`` / ``sku_id``).
* **Streamlit app** — ``dashboard/app.py`` consumes the Phase 3 processed
  files directly.

Nothing here retrains models, recomputes risk scores, or modifies CAM data.
The adapter only renames / projects CAM columns to the exact legacy shapes, so
Phase 6+ can feed ``forecast_base`` into the existing pipeline without touching
the engine code. The engines themselves are NOT rebuilt.
"""

import os
import sys
import functools
import pandas as pd

# Allow running as a script (`python src/cam_adapter.py`) as well as importing.
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from src.data_integration import (
    INTEGRATED_DIR,
    SRC_UCI,
    SRC_SYNTHETIC,
)

# Legacy column names the ML feature pipeline expects on the sales frame.
LEGACY_SALES_COLUMNS = [
    "date", "sku_id", "units_sold", "total_revenue", "avg_unit_price",
    "transaction_count", "unique_customers", "promotion_flag",
]


@functools.lru_cache(maxsize=32)
def load_cam_table(name: str, integrated_dir: str = INTEGRATED_DIR) -> pd.DataFrame:
    """Load a CAM table from ``data/processed/integrated/`` (parquet)."""
    path = os.path.join(integrated_dir, f"{name}.parquet")
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_forecast_base(integrated_dir: str = INTEGRATED_DIR) -> pd.DataFrame:
    """Load ``forecast_base.parquet`` (the standardized forecasting input)."""
    return load_cam_table("forecast_base", integrated_dir)


# ---------------------------------------------------------------------------
# ML Forecasting compatibility
# ---------------------------------------------------------------------------

def forecast_base_to_legacy_sales(forecast_base: pd.DataFrame = None,
                                  source_dataset: str = None) -> pd.DataFrame:
    """Map ``forecast_base`` onto the legacy *sales* frame shape.

    Returns a frame with the exact column names the existing feature pipeline
    expects (``aggregate_daily_sales`` / ``build_forecasting_feature_matrix``):
    ``total_revenue`` and ``avg_unit_price`` replace the CAM names ``revenue``
    and ``average_unit_price``. Optionally filter to one source system.
    """
    fb = forecast_base if forecast_base is not None else load_forecast_base()
    if source_dataset is not None:
        fb = fb[fb["source_dataset"] == source_dataset]
    out = pd.DataFrame({
        "date": fb["date"],
        "sku_id": fb["sku_id"],
        "units_sold": fb["units_sold"],
        "total_revenue": fb["revenue"],
        "avg_unit_price": fb["average_unit_price"],
        "transaction_count": fb["transaction_count"],
        "unique_customers": fb["unique_customers"],
        "promotion_flag": fb["promotion_flag"],
    })
    # Keep source context (harmless to the engine; useful for filtering).
    out["source_dataset"] = fb["source_dataset"].values
    out["entity_id"] = fb["entity_id"].values
    return out.reset_index(drop=True)


def product_dim_to_legacy_sku_master(dim_product: pd.DataFrame = None,
                                     source_dataset: str = None) -> pd.DataFrame:
    """Map ``dim_product`` onto the legacy *SKU master* shape the ML pipeline
    merges on ``sku_id`` (``sku_name`` <- ``product_name``)."""
    prod = dim_product if dim_product is not None else load_cam_table("dim_product")
    if source_dataset is not None:
        prod = prod[prod["source_dataset"] == source_dataset]
    out = prod.rename(columns={"product_name": "sku_name"})[
        ["sku_id", "sku_name", "category", "sub_category", "brand",
         "cost_price", "base_price", "weight_kg", "supplier_id",
         "lead_time_days", "reorder_point", "safety_stock"]
    ].reset_index(drop=True)
    return out


def check_ml_compatibility() -> dict:
    """Verify ``forecast_base`` -> legacy sales / SKU master mapping coverage."""
    fb = load_forecast_base()
    sales = forecast_base_to_legacy_sales(fb)
    missing = [c for c in LEGACY_SALES_COLUMNS if c not in sales.columns]
    prod = load_cam_table("dim_product")
    sku_missing = [c for c in ["sku_id", "sku_name", "category", "sub_category",
                               "brand", "cost_price", "base_price", "weight_kg",
                               "supplier_id", "lead_time_days", "reorder_point",
                               "safety_stock"] if c not in prod.rename(columns={"product_name": "sku_name"}).columns]
    return {
        "consumer": "ML Forecasting Engine (feature pipeline)",
        "source": "forecast_base + dim_product",
        "required_columns": LEGACY_SALES_COLUMNS,
        "missing_required_columns": missing,
        "missing_sku_master_columns": sku_missing,
        "synthetic_rows": int((fb["source_dataset"] == SRC_SYNTHETIC).sum()),
        "uci_rows": int((fb["source_dataset"] == SRC_UCI).sum()),
        "compatible": len(missing) == 0 and len(sku_missing) == 0,
    }


# ---------------------------------------------------------------------------
# Inventory Risk compatibility
# ---------------------------------------------------------------------------

def inventory_analytics_to_legacy_snapshots(inventory_analytics: pd.DataFrame = None,
                                            source_dataset: str = SRC_SYNTHETIC) -> pd.DataFrame:
    """Map ``inventory_analytics`` onto the legacy *inventory snapshot* shape
    the risk engine consumes (``store_id`` <- ``entity_id``, ``sku_id`` kept).
    Risk engines only ever consume the Synthetic source (UCI has no inventory).
    """
    inv = inventory_analytics if inventory_analytics is not None else load_cam_table("inventory_analytics")
    if source_dataset is not None:
        inv = inv[inv["source_dataset"] == source_dataset]
    out = pd.DataFrame({
        "date": inv["date"],
        "store_id": inv["entity_id"],
        "sku_id": inv["sku_id"],
        "ending_inventory": inv["ending_inventory"],
        "on_order_qty": inv["on_order_qty"],
        "stockout_flag": inv["stockout_flag"],
        "reorder_point": inv["reorder_point"],
        "safety_stock": inv["safety_stock"],
        "lead_time_days": inv["lead_time_days"],
        "category": inv["category"],
        "sub_category": inv["sub_category"],
        "brand": inv["brand"],
    })
    return out.reset_index(drop=True)


def check_risk_compatibility() -> dict:
    """Verify ``inventory_analytics`` -> legacy snapshot mapping coverage."""
    inv = load_cam_table("inventory_analytics")
    mapped = inventory_analytics_to_legacy_snapshots(inv)
    risk_cols = ["date", "store_id", "sku_id", "ending_inventory",
                 "on_order_qty", "stockout_flag", "reorder_point",
                 "safety_stock", "lead_time_days"]
    missing = [c for c in risk_cols if c not in mapped.columns]
    return {
        "consumer": "Inventory Risk Engine",
        "source": "inventory_analytics",
        "required_columns": risk_cols,
        "missing_required_columns": missing,
        "rows": int(len(mapped)),
        "compatible": len(missing) == 0,
    }


# ---------------------------------------------------------------------------
# Streamlit compatibility
# ---------------------------------------------------------------------------

def check_app_compatibility() -> dict:
    """Document what the existing Streamlit app currently requires.

    Phase 4 does NOT redesign the app. This records the contract the CAM must
    satisfy (or bridge via adapters) so a later phase can switch the app's data
    layer to CAM outputs.
    """
    return {
        "consumer": "Streamlit app (dashboard/app.py)",
        "policy": "App is unchanged in Phase 4; this is an inventory of its inputs.",
        "required_data_files": [
            "data/processed/sales_daily_clean.parquet",
            "data/processed/inventory_snapshots_clean.parquet",
            "data/processed/sku_master_clean.csv",
            "data/processed/store_master_clean.csv",
            "data/processed/customer_master_clean.csv",
            "data/processed/calendar_clean.csv",
            "data/processed/online_retail_sales.parquet",
        ],
        "required_columns": {
            "sales": ["date", "store_id", "sku_id", "units_sold", "total_revenue",
                      "avg_unit_price", "transaction_count", "unique_customers", "promotion_flag"],
            "inventory": ["date", "store_id", "sku_id", "ending_inventory", "on_order_qty",
                          "stockout_flag", "beginning_inventory_pre_receipts", "inventory_balance_ok"],
        },
        "required_model_files": [
            "models/lightgbm_forecaster.joblib",
            "models/xgboost_forecaster.joblib",
            "models/random_forest_forecaster.joblib",
        ],
        "required_risk_outputs": [
            "outputs/risk_scores/inventory_risk_matrix.parquet",
        ],
        "required_adapters": [
            "src/cam_adapter.forecast_base_to_legacy_sales (ML grain bridge)",
            "src/cam_adapter.inventory_analytics_to_legacy_snapshots (risk bridge)",
            "src/cam_adapter.product_dim_to_legacy_sku_master (product bridge)",
        ],
        "note": (
            "A later phase may re-point the app at CAM outputs; the app's "
            "selectors (active stores / tracked SKUs) and its SKU-total "
            "training grain must move to CAM together to avoid grain drift."
        ),
    }


def compatibility_report() -> dict:
    """Combined compatibility report across ML, risk, and app consumers."""
    ml = check_ml_compatibility()
    risk = check_risk_compatibility()
    app = check_app_compatibility()
    return {
        "phase": "Phase 4 — CAM compatibility layer",
        "ml_forecasting": ml,
        "inventory_risk": risk,
        "streamlit": app,
        "all_compatible": bool(ml["compatible"] and risk["compatible"]),
    }


if __name__ == "__main__":
    import json
    rep = compatibility_report()
    print(json.dumps(rep, indent=2, default=str))
