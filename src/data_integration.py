"""
Phase 4 — Data Integration & Common Analytical Model (CAM)
============================================================
Project FORESIGHT: Demand & Inventory Intelligence

Provides unified data access, dimension merging, and business aggregation
across multi-store sales, inventory snapshots, product dimensions, calendar,
and UCI Online Retail II transactions.

The module has two layers:

1. **Clean-data loaders** (cached) — the runtime access layer used by the
   ML forecasting engine, the inventory risk engine, and the Streamlit app.
   These are backward-compatible and must not change behaviour.

2. **Common Analytical Model (CAM) builders** — the Phase 4 standardized,
   source-aware star-schema layer written to ``data/processed/integrated/``.
   Every dimension and fact preserves ``source_dataset`` (``UCI`` /
   ``SYNTHETIC``) so the two source systems are never silently mixed.
"""

import os
import functools
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
INTEGRATED_DIR = os.path.join(PROCESSED_DIR, "integrated")

# Source-system tags (mandatory discriminator on every fact/dimension).
SRC_UCI = "UCI"
SRC_SYNTHETIC = "SYNTHETIC"

# UCI is a single online channel entity — never invent physical stores.
UCI_ENTITY_ID = "ONLINE"
UCI_ENTITY_TYPE = "CHANNEL"

# Month -> season mapping used by the Phase 3 cleaned calendar (and applied to
# UCI-derived calendar dates for the Northern-hemisphere calendar maths).
SEASON_MAP = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall", 12: "Winter",
}


# ===========================================================================
# Layer 1 — Clean-data loaders (backward-compatible, used by app & engines)
# ===========================================================================

@functools.lru_cache(maxsize=1)
def load_store_master(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned store master dataset."""
    path = os.path.join(processed_dir, "store_master_clean.csv")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "store_master.csv")
    df = pd.read_csv(path)
    df["store_id"] = df["store_id"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_sku_master(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned SKU master dataset."""
    path = os.path.join(processed_dir, "sku_master_clean.csv")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "sku_master.csv")
    df = pd.read_csv(path)
    df["sku_id"] = df["sku_id"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_customer_master(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned customer master dataset."""
    path = os.path.join(processed_dir, "customer_master_clean.csv")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "customer_master.csv")
    df = pd.read_csv(path)
    df["customer_id"] = df["customer_id"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_calendar(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned calendar dataset (synthetic 2022-2025)."""
    path = os.path.join(processed_dir, "calendar_clean.csv")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "calendar.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@functools.lru_cache(maxsize=1)
def load_sales_daily(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned sales daily parquet dataset (synthetic)."""
    path = os.path.join(processed_dir, "sales_daily_clean.parquet")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "sales_daily.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["store_id"] = df["store_id"].astype(str)
    df["sku_id"] = df["sku_id"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_inventory_snapshots(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned inventory snapshots parquet dataset (synthetic)."""
    path = os.path.join(processed_dir, "inventory_snapshots_clean.parquet")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "inventory_snapshots.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["store_id"] = df["store_id"].astype(str)
    df["sku_id"] = df["sku_id"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_online_retail(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load cleaned Online Retail II parquet dataset (sales split)."""
    parquet_path = os.path.join(processed_dir, "online_retail_sales.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        csv_path = os.path.join(processed_dir, "online_retail_sales.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, low_memory=False)
        else:
            clean_csv = os.path.join(processed_dir, "online_retail_clean.csv")
            df = pd.read_csv(clean_csv, low_memory=False)
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@functools.lru_cache(maxsize=1)
def load_uci_sales(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load the cleaned UCI *sales* split (normal demand only)."""
    path = os.path.join(processed_dir, "online_retail_sales.parquet")
    if not os.path.exists(path):
        path = os.path.join(processed_dir, "online_retail_sales.csv")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)
    df["invoice_date_only"] = pd.to_datetime(df["invoice_date_only"])
    return df


@functools.lru_cache(maxsize=1)
def load_uci_returns(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load the cleaned UCI returns split (kept separate from demand)."""
    path = os.path.join(processed_dir, "online_retail_returns.csv")
    df = pd.read_csv(path)
    df["invoice_date_only"] = pd.to_datetime(df["invoice_date_only"])
    return df


@functools.lru_cache(maxsize=1)
def load_uci_cancellations(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load the cleaned UCI cancellations split (kept separate from demand)."""
    path = os.path.join(processed_dir, "online_retail_cancellations.csv")
    df = pd.read_csv(path)
    df["invoice_date_only"] = pd.to_datetime(df["invoice_date_only"])
    return df


# ===========================================================================
# Layer 2 — Common Analytical Model (CAM) builders
# ===========================================================================

def _as_string_series(s: pd.Series) -> pd.Series:
    """Cast to pandas nullable ``string`` dtype so ``pd.NA`` round-trips."""
    return s.astype("string")


def _modal_by_key(df: pd.DataFrame, key_col: str, val_col: str) -> pd.Series:
    """Vectorized most-common ``val_col`` value per ``key_col`` (ties: first).

    Avoids per-group Python lambdas on million-row frames.
    """
    sub = df.dropna(subset=[val_col])
    counts = sub.value_counts(subset=[key_col, val_col], sort=False)
    counts = counts.reset_index(name="n").sort_values("n", ascending=False)
    top = counts.drop_duplicates(subset=[key_col], keep="first")
    return top.set_index(key_col)[val_col]


def load_clean_data() -> dict:
    """Load every Phase 3 processed dataset used by the CAM.

    Returns a dict keyed by logical name. Raw files are never read.
    """
    return {
        "store_master": load_store_master(),
        "sku_master": load_sku_master(),
        "customer_master": load_customer_master(),
        "calendar": load_calendar(),
        "sales_daily": load_sales_daily(),
        "inventory_snapshots": load_inventory_snapshots(),
        "uci_sales": load_uci_sales(),
        "uci_returns": load_uci_returns(),
        "uci_cancellations": load_uci_cancellations(),
    }


def create_calendar_dimension() -> pd.DataFrame:
    """Build the shared ``dim_calendar``.

    Starts from the Phase 3 cleaned synthetic calendar (2022-2025) and extends
    it over the UCI transaction date range (2009-2011). UCI dates get pure
    calendar-math attributes (year/month/quarter/day/day-of-week/week-of-year
    and month-derived season). ``is_holiday`` is set to 0 and ``holiday_name``
    to NULL for UCI dates — the synthetic US holiday calendar is *not* applied
    to the UK online source (no fabricated holidays).

    ``date_source`` records provenance of each date row.
    """
    cal = load_calendar().copy()
    keep = ["date", "year", "month", "quarter", "day", "day_of_week",
            "day_name", "is_weekend", "is_holiday", "holiday_name",
            "season", "week_of_year"]
    cal = cal[keep]
    cal["date"] = pd.to_datetime(cal["date"])
    cal["date_source"] = SRC_SYNTHETIC

    # UCI contiguous date span (no hardcoding — derive from the data).
    uci_dates = load_uci_sales()["invoice_date_only"]
    dmin, dmax = uci_dates.min(), uci_dates.max()
    uci_range = pd.date_range(dmin, dmax, freq="D")

    uci_cal = pd.DataFrame({"date": uci_range})
    uci_cal["year"] = uci_cal["date"].dt.year
    uci_cal["month"] = uci_cal["date"].dt.month
    uci_cal["quarter"] = uci_cal["date"].dt.quarter
    uci_cal["day"] = uci_cal["date"].dt.day
    uci_cal["day_of_week"] = uci_cal["date"].dt.dayofweek
    uci_cal["day_name"] = uci_cal["date"].dt.day_name()
    uci_cal["is_weekend"] = uci_cal["date"].dt.dayofweek.isin([5, 6]).astype(int)
    uci_cal["week_of_year"] = uci_cal["date"].dt.isocalendar().week.astype(int)
    uci_cal["season"] = uci_cal["month"].map(SEASON_MAP)
    uci_cal["is_holiday"] = 0
    uci_cal["holiday_name"] = pd.NA
    uci_cal["date_source"] = SRC_UCI + "_DERIVED"

    dim = pd.concat([cal, uci_cal], ignore_index=True)
    dim = dim.drop_duplicates(subset=["date"], keep="first")
    dim = dim.sort_values("date").reset_index(drop=True)
    dim["day_name"] = _as_string_series(dim["day_name"])
    dim["holiday_name"] = _as_string_series(dim["holiday_name"])
    return dim


def create_product_dimension() -> pd.DataFrame:
    """Build the source-aware ``dim_product`` with collision-safe keys.

    Keys: ``SYN_<sku_id>`` for the synthetic retail system and ``UCI_<StockCode>``
    for the online source. Synthetic rows carry the full SKU master attributes;
    UCI rows only populate fields that are actually available (product_key,
    source_dataset, sku_id, product_name) — category/brand/supplier/lead
    time/reorder point/safety stock are NULL (never fabricated).
    """
    skus = load_sku_master()
    syn = pd.DataFrame({
        "product_key": "SYN_" + skus["sku_id"],
        "source_dataset": SRC_SYNTHETIC,
        "sku_id": skus["sku_id"],
        "product_name": skus["sku_name"],
        "category": skus["category"],
        "sub_category": skus["sub_category"],
        "brand": skus["brand"],
        "cost_price": skus["cost_price"],
        "base_price": skus["base_price"],
        "weight_kg": skus["weight_kg"],
        "supplier_id": skus["supplier_id"],
        "lead_time_days": skus["lead_time_days"],
        "reorder_point": skus["reorder_point"],
        "safety_stock": skus["safety_stock"],
    })

    # UCI product universe = StockCodes present in sales + returns + cancellations
    # (so every fact's product FK resolves). Description = modal per StockCode.
    parts = [
        load_uci_sales()[["StockCode", "Description"]],
        load_uci_returns()[["StockCode", "Description"]],
        load_uci_cancellations()[["StockCode", "Description"]],
    ]
    lines = pd.concat(parts, ignore_index=True)
    modal_desc = _modal_by_key(lines, "StockCode", "Description")
    uci = pd.DataFrame({
        "product_key": "UCI_" + modal_desc.index.astype(str),
        "source_dataset": SRC_UCI,
        "sku_id": modal_desc.index.astype(str),
        "product_name": modal_desc.values,
    })
    for col in ["category", "sub_category", "brand", "supplier_id",
                "lead_time_days", "reorder_point", "safety_stock"]:
        uci[col] = pd.NA
    for col in ["cost_price", "base_price", "weight_kg"]:
        uci[col] = np.nan

    dim = pd.concat([syn, uci], ignore_index=True).reset_index(drop=True)
    for col in ["product_key", "source_dataset", "sku_id", "product_name",
                "category", "sub_category", "brand", "supplier_id"]:
        dim[col] = _as_string_series(dim[col])
    for col in ["lead_time_days", "reorder_point", "safety_stock"]:
        dim[col] = dim[col].astype("Int64")
    for col in ["cost_price", "base_price", "weight_kg"]:
        dim[col] = dim[col].astype("float64")
    return dim


def create_entity_dimension() -> pd.DataFrame:
    """Build the source-aware ``dim_entity``.

    Synthetic: one row per store (``entity_id = store_id``, ``entity_type =
    STORE``) with the physical store attributes. UCI: a single channel row
    (``entity_id = ONLINE``, ``entity_type = CHANNEL``) with NULL physical-store
    fields — no fake stores are invented for the online source.
    """
    stores = load_store_master()
    syn = pd.DataFrame({
        "entity_id": stores["store_id"],
        "entity_type": "STORE",
        "source_dataset": SRC_SYNTHETIC,
        "store_name": stores["store_name"],
        "city": stores["city"],
        "state": stores["state"],
        "region": stores["region"],
        "store_type": stores["store_type"],
        "store_size_sqft": stores["store_size_sqft"],
        "opening_date": stores["opening_date"],
    })
    uci = pd.DataFrame({
        "entity_id": [UCI_ENTITY_ID],
        "entity_type": [UCI_ENTITY_TYPE],
        "source_dataset": [SRC_UCI],
        "store_name": [pd.NA],
        "city": [pd.NA],
        "state": [pd.NA],
        "region": [pd.NA],
        "store_type": [pd.NA],
        "store_size_sqft": [pd.NA],
        "opening_date": [pd.NA],
    })
    dim = pd.concat([syn, uci], ignore_index=True).reset_index(drop=True)
    for col in ["entity_id", "entity_type", "source_dataset", "store_name",
                "city", "state", "region", "store_type", "opening_date"]:
        dim[col] = _as_string_series(dim[col])
    dim["store_size_sqft"] = dim["store_size_sqft"].fillna(np.nan).astype("float64")
    return dim


def create_customer_dimension() -> pd.DataFrame:
    """Build the source-aware ``dim_customer`` with collision-safe keys.

    Keys: ``SYN_<customer_id>`` and ``UCI_<CustomerID>``. Guest UCI transactions
    have no customer identity, so they are *not* dimension rows (their
    ``customer_key`` would be NULL); they are flagged via ``is_guest_transaction``
    on the sales fact and excluded from identified-customer analytics.
    """
    cust = load_customer_master()
    syn = pd.DataFrame({
        "customer_key": "SYN_" + cust["customer_id"],
        "source_dataset": SRC_SYNTHETIC,
        "customer_id": cust["customer_id"],
        "customer_name": cust["customer_name"],
        "customer_segment": cust["customer_segment"],
        "loyalty_member": cust["loyalty_member"],
        "signup_date": cust["signup_date"],
        "country": pd.NA,
        "is_guest_transaction": False,
    })

    sales = load_uci_sales()
    identified = sales[sales["Customer ID"].notna()].copy()
    identified["cid"] = identified["Customer ID"].astype(int).astype(str)
    country = _modal_by_key(identified, "cid", "Country")
    uci = pd.DataFrame({
        "customer_key": "UCI_" + country.index,
        "source_dataset": SRC_UCI,
        "customer_id": country.index.astype(str),
        "customer_name": pd.NA,
        "customer_segment": pd.NA,
        "loyalty_member": pd.NA,
        "signup_date": pd.NA,
        "country": country.values,
        "is_guest_transaction": False,
    })
    dim = pd.concat([syn, uci], ignore_index=True).reset_index(drop=True)
    for col in ["customer_key", "source_dataset", "customer_id", "customer_name",
                "customer_segment", "signup_date", "country"]:
        dim[col] = _as_string_series(dim[col])
    dim["loyalty_member"] = dim["loyalty_member"].astype("Int64")
    dim["is_guest_transaction"] = dim["is_guest_transaction"].astype(bool)
    return dim


def create_sales_fact() -> pd.DataFrame:
    """Build ``fact_sales`` at grain ``date + source_dataset + entity_id + product_key``.

    UCI rows are aggregated from transaction level to DATE + SKU with
    ``entity_id = ONLINE``; guest transactions are kept in the sales (their
    revenue/units count) but never counted as identified ``unique_customers``.
    Synthetic rows map ``store_id -> entity_id`` and ``sku_id -> product_key``
    (``total_revenue -> revenue``, ``avg_unit_price -> average_unit_price``).
    Returns and cancellations are never included in demand sales.
    """
    # ---- UCI -------------------------------------------------------------
    sales = load_uci_sales()
    sales["date"] = sales["invoice_date_only"]
    sales["revenue"] = sales["Quantity"] * sales["Price"]
    g = sales.groupby(["date", "StockCode"], as_index=False).agg(
        units_sold=("Quantity", "sum"),
        revenue=("revenue", "sum"),
        transaction_count=("Invoice", "nunique"),
    )
    # Distinct identified customers per date+SKU (guests excluded, C-level).
    uc = (
        sales[sales["Customer ID"].notna()]
        .groupby(["date", "StockCode"])["Customer ID"].nunique()
        .rename("unique_customers").reset_index()
    )
    g = g.merge(uc, on=["date", "StockCode"], how="left")
    g["unique_customers"] = g["unique_customers"].fillna(0).astype("int64")
    g["average_unit_price"] = g["revenue"] / g["units_sold"].replace(0, np.nan)
    uci_fact = pd.DataFrame({
        "date": g["date"],
        "source_dataset": SRC_UCI,
        "entity_id": UCI_ENTITY_ID,
        "entity_type": UCI_ENTITY_TYPE,
        "product_key": "UCI_" + g["StockCode"].astype(str),
        "sku_id": g["StockCode"].astype(str),
        "units_sold": g["units_sold"],
        "revenue": g["revenue"],
        "average_unit_price": g["average_unit_price"],
        "transaction_count": g["transaction_count"],
        "unique_customers": g["unique_customers"],
    })
    uci_fact["promotion_flag"] = pd.NA  # no promotion data for the online source

    # ---- Synthetic --------------------------------------------------------
    syn = load_sales_daily()
    syn_fact = pd.DataFrame({
        "date": syn["date"],
        "source_dataset": SRC_SYNTHETIC,
        "entity_id": syn["store_id"],
        "entity_type": "STORE",
        "product_key": "SYN_" + syn["sku_id"],
        "sku_id": syn["sku_id"],
        "units_sold": syn["units_sold"],
        "revenue": syn["total_revenue"],
        "average_unit_price": syn["avg_unit_price"],
        "transaction_count": syn["transaction_count"],
        "unique_customers": syn["unique_customers"],
        "promotion_flag": syn["promotion_flag"],
    })

    fact = pd.concat([uci_fact, syn_fact], ignore_index=True).reset_index(drop=True)
    fact["date"] = pd.to_datetime(fact["date"])
    for col in ["source_dataset", "entity_id", "entity_type", "product_key", "sku_id"]:
        fact[col] = _as_string_series(fact[col])
    fact["promotion_flag"] = fact["promotion_flag"].astype("Int64")
    return fact


def create_inventory_fact() -> pd.DataFrame:
    """Build ``fact_inventory`` (Synthetic only — UCI has no native inventory).

    Preserves the Phase 3 REVIEW semantic: ``beginning_inventory`` already
    includes the day's receipts, so ``ending_inventory = beginning_inventory -
    units_sold``. ``beginning_inventory_pre_receipts`` and
    ``inventory_balance_ok`` are carried through unchanged; no inventory value
    is overwritten. UCI receives NO fake inventory records.
    """
    inv = load_inventory_snapshots()
    fact = pd.DataFrame({
        "date": inv["date"],
        "source_dataset": SRC_SYNTHETIC,
        "entity_id": inv["store_id"],
        "product_key": "SYN_" + inv["sku_id"],
        "sku_id": inv["sku_id"],
        "beginning_inventory": inv["beginning_inventory"],
        "beginning_inventory_pre_receipts": inv["beginning_inventory_pre_receipts"],
        "receipts": inv["receipts"],
        "units_sold": inv["units_sold"],
        "ending_inventory": inv["ending_inventory"],
        "stockout_flag": inv["stockout_flag"],
        "on_order_qty": inv["on_order_qty"],
        "inventory_balance_ok": inv["inventory_balance_ok"],
    }).reset_index(drop=True)
    fact["date"] = pd.to_datetime(fact["date"])
    for col in ["source_dataset", "entity_id", "product_key", "sku_id"]:
        fact[col] = _as_string_series(fact[col])
    return fact


def _aggregate_uci_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a UCI split (returns / cancellations) to DATE + SKU."""
    df = df.copy()
    df["date"] = df["invoice_date_only"]
    df["revenue"] = df["Quantity"] * df["Price"]
    g = df.groupby(["date", "StockCode"], as_index=False).agg(
        quantity=("Quantity", lambda s: s.abs().sum()),
        transactions=("Invoice", "nunique"),
        revenue_impact=("revenue", "sum"),
    )
    return g


def create_returns_fact() -> pd.DataFrame:
    """Build ``fact_returns`` from the UCI returns split.

    Kept strictly separate from demand sales. Note: all Phase 3 UCI returns are
    guest, UK, price-zero lines — revenue impact is therefore 0.0 (documented,
    not silently dropped).
    """
    ret = load_uci_returns()
    g = _aggregate_uci_lines(ret)
    fact = pd.DataFrame({
        "date": g["date"],
        "source_dataset": SRC_UCI,
        "entity_id": UCI_ENTITY_ID,
        "entity_type": UCI_ENTITY_TYPE,
        "product_key": "UCI_" + g["StockCode"].astype(str),
        "sku_id": g["StockCode"].astype(str),
        "quantity_returned": g["quantity"],
        "return_transactions": g["transactions"],
        "revenue_impact": g["revenue_impact"],
    }).reset_index(drop=True)
    fact["date"] = pd.to_datetime(fact["date"])
    for col in ["source_dataset", "entity_id", "entity_type", "product_key", "sku_id"]:
        fact[col] = _as_string_series(fact[col])
    return fact


def create_cancellations_fact() -> pd.DataFrame:
    """Build ``fact_cancellations`` from the UCI cancellations split.

    The documented anomalous cancellation line (Invoice ``C496350``, StockCode
    ``M`` "Manual", quantity ``+1``, price ``373.57``) is preserved — it appears
    as the ``UCI_M`` product on 2010-02-01 with a positive revenue impact of
    373.57, and is never silently removed.
    """
    canc = load_uci_cancellations()
    g = _aggregate_uci_lines(canc)
    fact = pd.DataFrame({
        "date": g["date"],
        "source_dataset": SRC_UCI,
        "entity_id": UCI_ENTITY_ID,
        "entity_type": UCI_ENTITY_TYPE,
        "product_key": "UCI_" + g["StockCode"].astype(str),
        "sku_id": g["StockCode"].astype(str),
        "cancelled_quantity": g["quantity"],
        "cancellation_transactions": g["transactions"],
        "revenue_impact": g["revenue_impact"],
    }).reset_index(drop=True)
    fact["date"] = pd.to_datetime(fact["date"])
    for col in ["source_dataset", "entity_id", "entity_type", "product_key", "sku_id"]:
        fact[col] = _as_string_series(fact[col])
    return fact


def create_customer_analytics() -> pd.DataFrame:
    """Build ``customer_analytics`` — identified-customer metrics only.

    UCI: computed from transaction lines where a Customer ID exists. Guest
    transactions are excluded from identified-customer metrics. Synthetic: the
    processed datasets carry no customer-grain transactions, so transaction
    metrics are NULL (never fabricated); segment/loyalty/signup come from the
    customer master.
    """
    # ---- UCI (identified customers only) ---------------------------------
    sales = load_uci_sales()
    ids = sales[sales["Customer ID"].notna()].copy()
    ids["cid"] = ids["Customer ID"].astype(int).astype(str)
    ids["date"] = ids["invoice_date_only"]
    ids["revenue"] = ids["Quantity"] * ids["Price"]
    country = _modal_by_key(ids, "cid", "Country")
    agg = ids.groupby("cid").agg(
        transaction_count=("Invoice", "nunique"),
        total_units=("Quantity", "sum"),
        total_revenue=("revenue", "sum"),
        first_purchase_date=("date", "min"),
        last_purchase_date=("date", "max"),
    )
    uci = pd.DataFrame({
        "customer_key": "UCI_" + agg.index,
        "source_dataset": SRC_UCI,
        "customer_id": agg.index.astype(str),
        "customer_segment": pd.NA,
        "loyalty_member": pd.NA,
        "signup_date": pd.NA,
        "country": country.reindex(agg.index).values,
        "transaction_count": agg["transaction_count"].values,
        "total_units": agg["total_units"].values,
        "total_revenue": agg["total_revenue"].values,
        "first_purchase_date": agg["first_purchase_date"].values,
        "last_purchase_date": agg["last_purchase_date"].values,
    })

    # ---- Synthetic (master attributes; no customer-grain transactions) ----
    cust = load_customer_master()
    syn = pd.DataFrame({
        "customer_key": "SYN_" + cust["customer_id"],
        "source_dataset": SRC_SYNTHETIC,
        "customer_id": cust["customer_id"],
        "customer_segment": cust["customer_segment"],
        "loyalty_member": cust["loyalty_member"],
        "signup_date": cust["signup_date"],
        "country": pd.NA,
        "transaction_count": pd.NA,
        "total_units": pd.NA,
        "total_revenue": pd.NA,
        "first_purchase_date": pd.NA,
        "last_purchase_date": pd.NA,
    })

    out = pd.concat([uci, syn], ignore_index=True).reset_index(drop=True)
    for col in ["customer_key", "source_dataset", "customer_id", "customer_segment",
                "signup_date", "country"]:
        out[col] = _as_string_series(out[col])
    out["loyalty_member"] = out["loyalty_member"].astype("Int64")
    out["transaction_count"] = out["transaction_count"].astype("Int64")
    out["total_units"] = out["total_units"].astype("Int64")
    for col in ["first_purchase_date", "last_purchase_date"]:
        out[col] = pd.to_datetime(out[col])
    return out


def create_inventory_analytics(fact_inventory: pd.DataFrame = None,
                               dim_product: pd.DataFrame = None) -> pd.DataFrame:
    """Build ``inventory_analytics`` — inventory joined to product attributes.

    This table *prepares* the data for Phase 10 (inventory risk scoring); it
    does NOT compute risk scores. Grain is unchanged from ``fact_inventory``.
    """
    inv = fact_inventory if fact_inventory is not None else create_inventory_fact()
    prod = dim_product if dim_product is not None else create_product_dimension()
    prod_keep = prod[["product_key", "category", "sub_category", "brand",
                      "lead_time_days", "reorder_point", "safety_stock"]]
    out = inv.merge(prod_keep, on="product_key", how="left")
    cols = ["date", "source_dataset", "entity_id", "product_key", "sku_id",
            "category", "sub_category", "brand", "ending_inventory",
            "on_order_qty", "stockout_flag", "lead_time_days",
            "reorder_point", "safety_stock"]
    return out[cols].reset_index(drop=True)


def create_forecast_base(fact_sales: pd.DataFrame = None) -> pd.DataFrame:
    """Build ``forecast_base`` — the standardized forecasting input.

    A projection of ``fact_sales`` with the exact downstream column contract.
    Deliberately contains NO lag / rolling features — those belong to
    Phase 6 (Feature Engineering).
    """
    fact = fact_sales if fact_sales is not None else create_sales_fact()
    cols = ["date", "source_dataset", "entity_id", "entity_type", "product_key",
            "sku_id", "units_sold", "revenue", "average_unit_price",
            "transaction_count", "unique_customers", "promotion_flag"]
    return fact[cols].reset_index(drop=True)


# ===========================================================================
# Validation
# ===========================================================================

def validate_grain(df: pd.DataFrame, key_cols: list, table_name: str) -> dict:
    """Return grain diagnostics: row count, duplicate keys, null keys."""
    row_count = int(len(df))
    if df.empty or not all(c in df.columns for c in key_cols):
        return {
            "table": table_name,
            "grain": "+".join(key_cols),
            "row_count": row_count,
            "duplicate_key_count": 0,
            "null_key_count": 0,
        }
    dup = int(df.duplicated(subset=key_cols).sum())
    nulls = int(df[key_cols].isna().any(axis=1).sum())
    return {
        "table": table_name,
        "grain": "+".join(key_cols),
        "row_count": row_count,
        "duplicate_key_count": dup,
        "null_key_count": nulls,
    }


def validate_foreign_keys(fact: pd.DataFrame, dim: pd.DataFrame,
                          fact_keys: list, dim_keys: list,
                          fact_name: str, dim_name: str) -> dict:
    """Count fact rows whose key has no matching dimension row (orphans)."""
    if fact.empty or dim.empty:
        return {"fact": fact_name, "dim": dim_name, "orphan_count": 0}
    if not all(c in fact.columns for c in fact_keys) or not all(c in dim.columns for c in dim_keys):
        return {"fact": fact_name, "dim": dim_name, "orphan_count": -1}  # schema mismatch -> flag
    merged = fact[fact_keys].merge(
        dim[dim_keys], left_on=fact_keys, right_on=dim_keys, how="left", indicator=True
    )
    orphans = int((merged["_merge"] == "left_only").sum())
    return {"fact": fact_name, "dim": dim_name, "orphan_count": orphans}


def validate_business_rules(tables: dict) -> list:
    """Validate the documented business rules of the CAM.

    Returns a list of ``(rule_name, passed: bool, detail)`` tuples.
    """
    rules = []

    inv = tables["fact_inventory"]
    balanced = inv[inv["inventory_balance_ok"]]
    eq_ok = (balanced["ending_inventory"] ==
             balanced["beginning_inventory"] - balanced["units_sold"]).all()
    rules.append((
        "inventory_equation_ending_eq_beginning_minus_units_sold",
        bool(eq_ok),
        f"{len(balanced):,} balanced rows verified",
    ))
    # The engine must never re-add receipts: on balanced rows ending must also
    # differ from (beginning + receipts - units_sold) wherever receipts > 0.
    with_rec = balanced[balanced["receipts"] > 0]
    no_rea_add = not (with_rec["ending_inventory"] ==
                      with_rec["beginning_inventory"] + with_rec["receipts"] -
                      with_rec["units_sold"]).any()
    rules.append((
        "receipts_not_rea_added_to_ending",
        bool(no_rea_add),
        f"{len(with_rec):,} receipt-days checked",
    ))

    sales = tables["fact_sales"]
    rules.append(("sales_units_nonnegative", bool((sales["units_sold"] >= 0).all()),
                  f"{int((sales['units_sold'] < 0).sum())} negatives"))
    rules.append(("source_dataset_enum_valid",
                  set(sales["source_dataset"].unique()) <= {SRC_UCI, SRC_SYNTHETIC},
                  str(sorted(sales["source_dataset"].unique()))))
    uci_promo = sales.loc[sales["source_dataset"] == SRC_UCI, "promotion_flag"]
    rules.append(("uci_promotion_flag_null_not_invented",
                  bool(uci_promo.isna().all()),
                  "UCI has no promotion source -> NULL"))

    ret = tables["fact_returns"]
    canc = tables["fact_cancellations"]
    rules.append(("returns_separated_from_sales",
                  bool(ret["source_dataset"].eq(SRC_UCI).all()) and
                  not ret.empty,
                  f"{len(ret):,} return rows"))
    rules.append(("cancellations_separated_from_sales",
                  bool(canc["source_dataset"].eq(SRC_UCI).all()) and
                  not canc.empty,
                  f"{len(canc):,} cancellation rows"))

    # Preserve the documented anomalous cancellation line. It is preserved at
    # the LINE level in the source split (Invoice C496350 / StockCode M /
    # quantity +1 / price 373.57) and at the FACT level as the UCI_M group on
    # 2010-02-01 (the +373.57 contribution is aggregated with any other M
    # cancellations that day, so the group total is not 373.57 alone).
    src = load_uci_cancellations()
    src_anom = src[(src["Invoice"].astype(str) == "C496350") &
                   (src["StockCode"].astype(str) == "M")]
    fact_anom = canc[(canc["date"] == pd.Timestamp("2010-02-01")) &
                     (canc["product_key"] == "UCI_M")]
    src_ok = (len(src_anom) == 1 and int(src_anom["Quantity"].iloc[0]) == 1 and
              abs(float(src_anom["Price"].iloc[0]) - 373.57) < 1e-6)
    rules.append(("cancellation_anomaly_preserved",
                  bool(src_ok and len(fact_anom) == 1),
                  f"source line C496350 preserved: {src_ok}; fact group UCI_M/2010-02-01 "
                  f"revenue_impact={float(fact_anom['revenue_impact'].iloc[0]) if len(fact_anom) else None}"))

    inv_only_syn = set(tables["fact_inventory"]["source_dataset"].unique()) == {SRC_SYNTHETIC}
    rules.append(("inventory_synthetic_only_no_fake_uci_inventory",
                  bool(inv_only_syn), str(sorted(tables["fact_inventory"]["source_dataset"].unique()))))

    cust_analytics = tables["customer_analytics"]
    rules.append(("customer_analytics_no_null_customer_key",
                  bool(cust_analytics["customer_key"].notna().all()),
                  f"{int(cust_analytics['customer_key'].isna().sum())} null keys"))

    forecast_base = tables["forecast_base"]
    leaky = [c for c in forecast_base.columns if "lag_" in c or "rolling_" in c or "_ewm_" in c]
    rules.append(("forecast_base_has_no_lag_rolling_features",
                  len(leaky) == 0, f"leaky cols found: {leaky}"))

    return rules


def generate_integration_report(tables: dict, docs_dir: str = DOCS_DIR) -> dict:
    """Generate the Phase 4 integration quality report (JSON + CSV)."""
    os.makedirs(docs_dir, exist_ok=True)

    grains = {
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
    pks = {k: v for k, v in grains.items()}

    # Foreign-key relationships to validate across the star schema.
    fk_rels = [
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
        ("forecast_base", "dim_calendar", ["date"], ["date"]),
        ("forecast_base", "dim_product", ["product_key"], ["product_key"]),
        ("forecast_base", "dim_entity", ["source_dataset", "entity_id"], ["source_dataset", "entity_id"]),
    ]

    rows = []
    for name, df in tables.items():
        g = validate_grain(df, grains.get(name, ["date"]), name)
        fk_viol = sum(
            validate_foreign_keys(df, tables[dim], fk, dk, name, dim)["orphan_count"]
            for fact_name, dim, fk, dk in fk_rels if fact_name == name
        )
        status = "PASS"
        if g["duplicate_key_count"] or g["null_key_count"] or fk_viol:
            status = "REVIEW"
        rows.append({
            "table_name": name,
            "source_dataset": ",".join(sorted(df["source_dataset"].astype(str).unique()))
                             if "source_dataset" in df.columns else "N/A",
            "row_count": g["row_count"],
            "column_count": int(len(df.columns)),
            "primary_key": "+".join(pks[name]),
            "duplicate_keys": g["duplicate_key_count"],
            "null_keys": g["null_key_count"],
            "foreign_key_violations": fk_viol,
            "grain": g["grain"],
            "status": status,
        })

    business_rules = [
        {"rule": r[0], "passed": bool(r[1]), "detail": r[2]}
        for r in validate_business_rules(tables)
    ]

    report = {
        "project": "FORESIGHT — Demand & Inventory Intelligence",
        "phase": "Phase 4 — Data Integration & Common Analytical Model (CAM)",
        "generated_at": str(pd.Timestamp.now()),
        "inventory_data_status": "REVIEW",
        "inventory_data_status_notes": (
            "Phase 3 semantic preserved: beginning_inventory already includes "
            "the day's receipts; ending_inventory = beginning_inventory - "
            "units_sold. Receipts are never re-added."
        ),
        "tables": rows,
        "business_rules": business_rules,
        "rule_passed": int(sum(r["passed"] for r in business_rules)),
        "rule_total": len(business_rules),
    }

    json_path = os.path.join(docs_dir, "integration_quality_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    csv_path = os.path.join(docs_dir, "integration_quality_report.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    report["_files"] = {"json": json_path, "csv": csv_path}
    return report


# ===========================================================================
# Persistence
# ===========================================================================

def save_integrated_data(tables: dict, out_dir: str = INTEGRATED_DIR) -> dict:
    """Persist every CAM table as Parquet under ``data/processed/integrated/``."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for name, df in tables.items():
        path = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        written[name] = path
    return written


# ===========================================================================
# Orchestration
# ===========================================================================

def run_integration_pipeline(save: bool = True,
                             out_dir: str = INTEGRATED_DIR,
                             docs_dir: str = DOCS_DIR) -> tuple:
    """Run the full Phase 4 integration pipeline.

    Returns ``(tables, report)`` where ``tables`` is the dict of CAM
    DataFrames and ``report`` is the integration quality report.
    """
    tables = {
        "dim_calendar": create_calendar_dimension(),
        "dim_product": create_product_dimension(),
        "dim_entity": create_entity_dimension(),
        "dim_customer": create_customer_dimension(),
        "fact_sales": create_sales_fact(),
        "fact_inventory": create_inventory_fact(),
        "fact_returns": create_returns_fact(),
        "fact_cancellations": create_cancellations_fact(),
        "customer_analytics": create_customer_analytics(),
        "inventory_analytics": create_inventory_analytics(),
        "forecast_base": create_forecast_base(),
    }
    if save:
        save_integrated_data(tables, out_dir=out_dir)
    report = generate_integration_report(tables, docs_dir=docs_dir)
    return tables, report


# ===========================================================================
# Backward-compatible analytics helpers (consumed by the Streamlit app)
# ===========================================================================

def build_integrated_cam(
    sample_store_count: int = None,
    sample_sku_count: int = None,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Build the Common Analytical Model (CAM) joining Sales, Inventory,
    SKU Master, Store Master, and Calendar.
    """
    sales = load_sales_daily()
    inventory = load_inventory_snapshots()
    skus = load_sku_master()
    stores = load_store_master()
    calendar = load_calendar()

    # Optional filtering
    if start_date:
        sales = sales[sales["date"] >= pd.to_datetime(start_date)]
        inventory = inventory[inventory["date"] >= pd.to_datetime(start_date)]
    if end_date:
        sales = sales[sales["date"] <= pd.to_datetime(end_date)]
        inventory = inventory[inventory["date"] <= pd.to_datetime(end_date)]
    if sample_store_count:
        selected_stores = stores["store_id"].head(sample_store_count).tolist()
        sales = sales[sales["store_id"].isin(selected_stores)]
        inventory = inventory[inventory["store_id"].isin(selected_stores)]
    if sample_sku_count:
        selected_skus = skus["sku_id"].head(sample_sku_count).tolist()
        sales = sales[sales["sku_id"].isin(selected_skus)]
        inventory = inventory[inventory["sku_id"].isin(selected_skus)]

    # Merge sales and inventory
    merged = pd.merge(
        sales,
        inventory,
        on=["date", "store_id", "sku_id"],
        how="inner",
        suffixes=("", "_inv"),
    )

    # Merge SKU Master
    sku_cols = [
        "sku_id", "sku_name", "category", "sub_category", "brand",
        "cost_price", "base_price", "lead_time_days", "reorder_point", "safety_stock"
    ]
    merged = pd.merge(merged, skus[sku_cols], on="sku_id", how="left")

    # Merge Store Master
    store_cols = ["store_id", "store_name", "city", "state", "region", "store_type"]
    merged = pd.merge(merged, stores[store_cols], on="store_id", how="left")

    # Merge Calendar
    cal_cols = ["date", "year", "month", "quarter", "day_of_week", "day_name", "is_weekend", "is_holiday", "season"]
    merged = pd.merge(merged, calendar[cal_cols], on="date", how="left")

    # Calculate derived financial & inventory metrics
    merged["cogs"] = merged["units_sold"] * merged["cost_price"]
    merged["gross_profit"] = merged["total_revenue"] - merged["cogs"]
    merged["margin_pct"] = np.where(
        merged["total_revenue"] > 0,
        merged["gross_profit"] / merged["total_revenue"],
        0.0
    )
    merged["ending_inventory_value"] = merged["ending_inventory"] * merged["cost_price"]
    merged["is_stockout"] = (merged["stockout_flag"] == 1) | (merged["ending_inventory"] == 0)

    return merged


def get_executive_kpis(cam_df: pd.DataFrame = None) -> dict:
    """Compute high-level executive KPIs across sales, inventory, and service levels."""
    if cam_df is None:
        sales = load_sales_daily()
        inventory = load_inventory_snapshots()
        skus = load_sku_master()
        stores = load_store_master()

        total_revenue = float(sales["total_revenue"].sum())
        total_units = int(sales["units_sold"].sum())
        total_transactions = int(sales["transaction_count"].sum())

        # Latest inventory snapshot
        latest_date = inventory["date"].max()
        latest_inv = inventory[inventory["date"] == latest_date].copy()
        latest_inv = pd.merge(latest_inv, skus[["sku_id", "cost_price", "base_price", "safety_stock", "reorder_point"]], on="sku_id", how="left")

        total_inventory_units = int(latest_inv["ending_inventory"].sum())
        total_inventory_value = float((latest_inv["ending_inventory"] * latest_inv["cost_price"]).sum())
        stockout_incidents = int(latest_inv["stockout_flag"].sum())
        stockout_rate = float(latest_inv["stockout_flag"].mean() * 100)

        # Reorder breaches
        reorder_triggered = int((latest_inv["ending_inventory"] <= latest_inv["reorder_point"]).sum())
        safety_breaches = int((latest_inv["ending_inventory"] < latest_inv["safety_stock"]).sum())

        return {
            "total_revenue": total_revenue,
            "total_units_sold": total_units,
            "total_transactions": total_transactions,
            "total_inventory_units": total_inventory_units,
            "total_inventory_value": total_inventory_value,
            "current_stockout_count": stockout_incidents,
            "current_stockout_rate_pct": stockout_rate,
            "reorder_triggered_count": reorder_triggered,
            "safety_stock_breaches": safety_breaches,
            "total_stores": len(stores),
            "total_skus": len(skus),
            "active_skus": int(sales["sku_id"].nunique()),
            "latest_date": str(latest_date)[:10],
        }
    else:
        total_revenue = float(cam_df["total_revenue"].sum())
        total_units = int(cam_df["units_sold"].sum())
        total_profit = float(cam_df["gross_profit"].sum()) if "gross_profit" in cam_df.columns else 0.0
        avg_margin = float(cam_df["margin_pct"].mean() * 100) if "margin_pct" in cam_df.columns else 0.0

        latest_date = cam_df["date"].max()
        latest_subset = cam_df[cam_df["date"] == latest_date]
        total_inventory_value = float(latest_subset["ending_inventory_value"].sum())
        stockout_rate = float(latest_subset["is_stockout"].mean() * 100)

        return {
            "total_revenue": total_revenue,
            "total_units_sold": total_units,
            "total_gross_profit": total_profit,
            "avg_margin_pct": avg_margin,
            "total_inventory_value": total_inventory_value,
            "current_stockout_rate_pct": stockout_rate,
            "latest_date": str(latest_date)[:10],
        }


def get_top_bottom_skus(top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve top performing and bottom performing SKUs by revenue and volume."""
    sales = load_sales_daily()
    skus = load_sku_master()

    agg = sales.groupby("sku_id").agg(
        total_units=("units_sold", "sum"),
        total_revenue=("total_revenue", "sum"),
        avg_price=("avg_unit_price", "mean"),
        active_days=("date", "nunique")
    ).reset_index()

    merged = pd.merge(agg, skus[["sku_id", "sku_name", "category", "brand", "cost_price", "base_price"]], on="sku_id", how="left")
    merged["margin_pct"] = ((merged["base_price"] - merged["cost_price"]) / merged["base_price"]) * 100

    top_df = merged.sort_values(by="total_revenue", ascending=False).head(top_n).reset_index(drop=True)
    bottom_df = merged.sort_values(by="total_revenue", ascending=True).head(top_n).reset_index(drop=True)
    return top_df, bottom_df


if __name__ == "__main__":
    tables, report = run_integration_pipeline()
    print("=" * 78)
    print("Phase 4 — CAM integration complete")
    print(f"  Tables written: {len(report['tables'])}")
    for row in report["tables"]:
        print(f"  {row['table_name']:<22} rows={row['row_count']:>12,} "
              f"dups={row['duplicate_keys']} nulls={row['null_keys']} "
              f"fk_viol={row['foreign_key_violations']} status={row['status']}")
    passed = report["rule_passed"]
    print(f"  Business rules: {passed}/{report['rule_total']} passed")
    for r in report["business_rules"]:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"    [{tag}] {r['rule']}  -- {r['detail']}")
