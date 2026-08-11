"""
Synthetic Retail Dataset Generator (Multi-Store Relational Model)
Simulates high-fidelity retail operations across:
- 30 Stores
- 5,000 SKUs (Categories, Cost, Base Price, Supplier Lead Time, Safety Stock, Reorder Point)
- 10,000 Customers (Segments, Loyalty status)
- 4 Years (2022 - 2025 Calendar with Seasonality & Holidays)
- Daily Aggregated Sales (`sales_daily.csv` & `sales_daily.parquet`)
- Inventory Snapshots (`inventory_snapshots.csv` & `inventory_snapshots.parquet`)
"""

import os
import time
import numpy as np
import pandas as pd

def generate_store_master(num_stores=30, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    cities = [
        ("New York", "NY", "East"), ("Los Angeles", "CA", "West"), ("Chicago", "IL", "Midwest"),
        ("Houston", "TX", "South"), ("Phoenix", "AZ", "West"), ("Philadelphia", "PA", "East"),
        ("San Antonio", "TX", "South"), ("San Diego", "CA", "West"), ("Dallas", "TX", "South"),
        ("San Jose", "CA", "West"), ("Austin", "TX", "South"), ("Jacksonville", "FL", "South"),
        ("San Francisco", "CA", "West"), ("Columbus", "OH", "Midwest"), ("Indianapolis", "IN", "Midwest"),
        ("Seattle", "WA", "West"), ("Denver", "CO", "West"), ("Boston", "MA", "East"),
        ("El Paso", "TX", "South"), ("Nashville", "TN", "South"), ("Detroit", "MI", "Midwest"),
        ("Portland", "OR", "West"), ("Memphis", "TN", "South"), ("Oklahoma City", "OK", "South"),
        ("Las Vegas", "NV", "West"), ("Louisville", "KY", "South"), ("Baltimore", "MD", "East"),
        ("Milwaukee", "WI", "Midwest"), ("Albuquerque", "NM", "West"), ("Tucson", "AZ", "West")
    ]
    
    store_types = ["Superstore", "Flagship", "Express", "Standard", "Outlet"]
    store_type_weights = [0.2, 0.15, 0.25, 0.3, 0.1]
    
    records = []
    for i in range(num_stores):
        store_id = f"STORE_{i+1:03d}"
        city_info = cities[i % len(cities)]
        stype = np.random.choice(store_types, p=store_type_weights)
        size = int(np.random.choice([15000, 25000, 45000, 60000, 85000]))
        opening_year = np.random.choice([2015, 2016, 2017, 2018, 2019, 2020, 2021])
        opening_date = f"{opening_year}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
        
        records.append({
            "store_id": store_id,
            "store_name": f"{city_info[0]} {stype} Retail",
            "city": city_info[0],
            "state": city_info[1],
            "region": city_info[2],
            "store_type": stype,
            "store_size_sqft": size,
            "opening_date": opening_date
        })
        
    df = pd.DataFrame(records)
    path = os.path.join(output_dir, "store_master.csv")
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} stores to {path}")
    return df

def generate_sku_master(num_skus=5000, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    categories = {
        "Electronics": {
            "subcategories": ["Audio", "Accessories", "Smart Home", "Wearables", "Computers"],
            "price_range": (15.0, 450.0),
            "margin_range": (0.25, 0.45),
            "lead_time_range": (7, 25)
        },
        "Apparel": {
            "subcategories": ["Men's Wear", "Women's Wear", "Footwear", "Sportswear", "Accessories"],
            "price_range": (12.0, 180.0),
            "margin_range": (0.40, 0.65),
            "lead_time_range": (10, 30)
        },
        "Groceries & Essentials": {
            "subcategories": ["Beverages", "Snacks", "Packaged Foods", "Personal Care", "Household Supplies"],
            "price_range": (2.5, 45.0),
            "margin_range": (0.15, 0.35),
            "lead_time_range": (3, 14)
        },
        "Home & Kitchen": {
            "subcategories": ["Cookware", "Bedding", "Storage & Org", "Home Decor", "Small Appliances"],
            "price_range": (10.0, 250.0),
            "margin_range": (0.35, 0.55),
            "lead_time_range": (7, 28)
        },
        "Health & Beauty": {
            "subcategories": ["Skincare", "Haircare", "Vitamins & Supplements", "Cosmetics", "Oral Care"],
            "price_range": (6.0, 120.0),
            "margin_range": (0.45, 0.70),
            "lead_time_range": (5, 21)
        },
        "Sports & Outdoors": {
            "subcategories": ["Fitness Gear", "Outdoor Recreation", "Camping", "Cycling", "Team Sports"],
            "price_range": (15.0, 350.0),
            "margin_range": (0.30, 0.50),
            "lead_time_range": (10, 30)
        }
    }
    
    cat_names = list(categories.keys())
    cat_probs = [0.18, 0.22, 0.25, 0.15, 0.12, 0.08]
    brands = [f"Brand_{chr(65+i)}{chr(65+j)}" for i in range(10) for j in range(5)]
    suppliers = [f"SUPP_{i+1:03d}" for i in range(50)]
    
    records = []
    for i in range(num_skus):
        sku_id = f"SKU_{i+1:05d}"
        cat = np.random.choice(cat_names, p=cat_probs)
        cat_info = categories[cat]
        subcat = np.random.choice(cat_info["subcategories"])
        brand = np.random.choice(brands)
        supp = np.random.choice(suppliers)
        
        base_price = round(float(np.random.uniform(cat_info["price_range"][0], cat_info["price_range"][1])), 2)
        margin = float(np.random.uniform(cat_info["margin_range"][0], cat_info["margin_range"][1]))
        cost_price = round(base_price * (1.0 - margin), 2)
        weight_kg = round(float(np.random.exponential(scale=1.5) + 0.1), 2)
        lead_time = int(np.random.randint(cat_info["lead_time_range"][0], cat_info["lead_time_range"][1] + 1))
        
        expected_daily_sales = float(np.random.pareto(a=2.5) * 5.0 + 1.0)
        safety_stock = int(np.ceil(1.65 * np.sqrt(lead_time) * (expected_daily_sales * 0.4)))
        reorder_point = int(np.ceil(expected_daily_sales * lead_time + safety_stock))
        
        records.append({
            "sku_id": sku_id,
            "sku_name": f"{brand} {subcat} Item {i+1}",
            "category": cat,
            "sub_category": subcat,
            "brand": brand,
            "cost_price": cost_price,
            "base_price": base_price,
            "weight_kg": min(weight_kg, 25.0),
            "supplier_id": supp,
            "lead_time_days": lead_time,
            "reorder_point": reorder_point,
            "safety_stock": safety_stock
        })
        
    df = pd.DataFrame(records)
    path = os.path.join(output_dir, "sku_master.csv")
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} SKUs to {path}")
    return df

def generate_customer_master(num_customers=10000, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    segments = ["Consumer", "Corporate", "Small Business", "VIP Loyalty"]
    seg_weights = [0.60, 0.20, 0.12, 0.08]
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
                   "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
                   "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                  "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson"]
    
    records = []
    for i in range(num_customers):
        cust_id = f"CUST_{i+1:06d}"
        fn = np.random.choice(first_names)
        ln = np.random.choice(last_names)
        seg = np.random.choice(segments, p=seg_weights)
        loyalty = 1 if (seg == "VIP Loyalty" or np.random.rand() < 0.35) else 0
        signup_year = np.random.choice([2019, 2020, 2021, 2022, 2023])
        signup_date = f"{signup_year}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
        
        records.append({
            "customer_id": cust_id,
            "customer_name": f"{fn} {ln}",
            "customer_segment": seg,
            "loyalty_member": loyalty,
            "signup_date": signup_date
        })
        
    df = pd.DataFrame(records)
    path = os.path.join(output_dir, "customer_master.csv")
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} customers to {path}")
    return df

def generate_calendar(start_date="2022-01-01", end_date="2025-12-31", output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    records = []
    seasons = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
               6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall", 12: "Winter"}
    
    us_holidays = {
        "01-01": "New Year's Day", "07-04": "Independence Day", "10-31": "Halloween",
        "11-25": "Thanksgiving Window", "11-26": "Black Friday", "11-29": "Cyber Monday",
        "12-24": "Christmas Eve", "12-25": "Christmas Day", "12-31": "New Year's Eve"
    }
    
    for d in dates:
        mmdd = d.strftime("%m-%d")
        is_holiday = 1 if mmdd in us_holidays else 0
        hol_name = us_holidays.get(mmdd, "Regular Day")
        is_weekend = 1 if d.dayofweek in [5, 6] else 0
        
        records.append({
            "date": d.strftime("%Y-%m-%d"),
            "year": int(d.year),
            "month": int(d.month),
            "quarter": int(d.quarter),
            "day": int(d.day),
            "day_of_week": int(d.dayofweek),
            "day_name": d.strftime("%A"),
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "holiday_name": hol_name,
            "season": seasons[d.month],
            "week_of_year": int(d.isocalendar().week)
        })
        
    df = pd.DataFrame(records)
    path = os.path.join(output_dir, "calendar.csv")
    df.to_csv(path, index=False)
    print(f"Saved calendar with {len(df)} days to {path}")
    return df

def generate_daily_sales_and_inventory_fast(output_dir="data/raw", num_stores=10, num_skus=100):
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    store_df = pd.read_csv(os.path.join(output_dir, "store_master.csv"))
    sku_df = pd.read_csv(os.path.join(output_dir, "sku_master.csv"))
    calendar_df = pd.read_csv(os.path.join(output_dir, "calendar.csv"))
    
    stores = store_df["store_id"].values[:num_stores]
    skus = sku_df["sku_id"].values[:num_skus]
    
    # Pre-extract calendar arrays
    date_strs = calendar_df["date"].values
    months = calendar_df["month"].values
    dows = calendar_df["day_of_week"].values
    weekends = calendar_df["is_weekend"].values
    holidays = calendar_df["is_holiday"].values
    num_days = len(date_strs)
    
    # Calendar factors
    month_factors = 1.0 + 0.35 * np.sin(2 * np.pi * (months - 3) / 12)
    month_factors[(months == 11) | (months == 12)] *= 1.30
    dow_factors = np.where(weekends == 1, 1.4, 0.95)
    hol_factors = np.where(holidays == 1, 1.6, 1.0)
    base_cal_factors = month_factors * dow_factors * hol_factors
    
    # Sku properties
    sku_price_map = dict(zip(sku_df["sku_id"], sku_df["base_price"]))
    sku_lead_map = dict(zip(sku_df["sku_id"], sku_df["lead_time_days"]))
    sku_rop_map = dict(zip(sku_df["sku_id"], sku_df["reorder_point"]))
    
    all_dates = []
    all_stores = []
    all_skus = []
    all_units_sold = []
    all_revenue = []
    all_prices = []
    all_tx_counts = []
    all_cust_counts = []
    all_promos = []
    
    all_beg_inv = []
    all_receipts = []
    all_end_inv = []
    all_stockout = []
    all_on_order = []
    
    total_series = len(stores) * len(skus)
    print(f"Vectorizing simulation across {total_series} series ({num_days} days each = {total_series * num_days:,} records)...")
    
    for store_id in stores:
        for sku_id in skus:
            base_price = sku_price_map[sku_id]
            lead_time = sku_lead_map[sku_id]
            rop = sku_rop_map[sku_id]
            base_demand = np.random.uniform(8, 35)
            
            # Promotions array
            promo_flags = (np.random.rand(num_days) < 0.10).astype(int)
            promo_discounts = promo_flags * np.random.choice([0.10, 0.15, 0.20, 0.25], size=num_days)
            promo_multipliers = np.where(promo_flags == 1, 1.0 + promo_discounts * 2.2, 1.0)
            prices = np.round(base_price * (1.0 - promo_discounts), 2)
            
            expected_demand = base_demand * base_cal_factors * promo_multipliers
            actual_demand = np.random.poisson(expected_demand)
            
            # Fast stateful simulation for inventory
            current_inv = int(rop * 1.5)
            on_order = 0
            order_arrival_day = -1
            
            beg_inv_arr = np.zeros(num_days, dtype=np.int32)
            receipts_arr = np.zeros(num_days, dtype=np.int32)
            sold_arr = np.zeros(num_days, dtype=np.int32)
            end_inv_arr = np.zeros(num_days, dtype=np.int32)
            stockout_arr = np.zeros(num_days, dtype=np.int32)
            on_order_arr = np.zeros(num_days, dtype=np.int32)
            
            for d_idx in range(num_days):
                if d_idx == order_arrival_day:
                    receipts = on_order
                    current_inv += receipts
                    on_order = 0
                    order_arrival_day = -1
                else:
                    receipts = 0
                    
                beg = current_inv
                dem = actual_demand[d_idx]
                sold = min(dem, current_inv)
                stkout = 1 if dem > current_inv else 0
                current_inv = max(0, current_inv - sold)
                end = current_inv
                
                if current_inv <= rop and on_order == 0:
                    on_order = int(rop * 1.8)
                    order_arrival_day = d_idx + lead_time
                    
                beg_inv_arr[d_idx] = beg
                receipts_arr[d_idx] = receipts
                sold_arr[d_idx] = sold
                end_inv_arr[d_idx] = end
                stockout_arr[d_idx] = stkout
                on_order_arr[d_idx] = on_order
                
            rev_arr = np.round(sold_arr * prices, 2)
            tx_arr = np.where(sold_arr > 0, np.maximum(1, np.ceil(sold_arr / 1.8)).astype(np.int32), 0)
            cust_arr = np.maximum(1, (tx_arr * 0.9).astype(np.int32))
            cust_arr[tx_arr == 0] = 0
            
            all_dates.append(date_strs)
            all_stores.append(np.full(num_days, store_id))
            all_skus.append(np.full(num_days, sku_id))
            all_units_sold.append(sold_arr)
            all_revenue.append(rev_arr)
            all_prices.append(prices)
            all_tx_counts.append(tx_arr)
            all_cust_counts.append(cust_arr)
            all_promos.append(promo_flags)
            
            all_beg_inv.append(beg_inv_arr)
            all_receipts.append(receipts_arr)
            all_end_inv.append(end_inv_arr)
            all_stockout.append(stockout_arr)
            all_on_order.append(on_order_arr)
            
    df_sales = pd.DataFrame({
        "date": np.concatenate(all_dates),
        "store_id": np.concatenate(all_stores),
        "sku_id": np.concatenate(all_skus),
        "units_sold": np.concatenate(all_units_sold),
        "total_revenue": np.concatenate(all_revenue),
        "avg_unit_price": np.concatenate(all_prices),
        "transaction_count": np.concatenate(all_tx_counts),
        "unique_customers": np.concatenate(all_cust_counts),
        "promotion_flag": np.concatenate(all_promos)
    })
    
    sales_path = os.path.join(output_dir, "sales_daily.csv")
    sales_parquet = os.path.join(output_dir, "sales_daily.parquet")
    df_sales.to_csv(sales_path, index=False)
    df_sales.to_parquet(sales_parquet, index=False)
    print(f"Saved {len(df_sales):,} daily sales records to {sales_path} and {sales_parquet}")
    
    df_inv = pd.DataFrame({
        "date": np.concatenate(all_dates),
        "store_id": np.concatenate(all_stores),
        "sku_id": np.concatenate(all_skus),
        "beginning_inventory": np.concatenate(all_beg_inv),
        "receipts": np.concatenate(all_receipts),
        "units_sold": np.concatenate(all_units_sold),
        "ending_inventory": np.concatenate(all_end_inv),
        "stockout_flag": np.concatenate(all_stockout),
        "on_order_qty": np.concatenate(all_on_order)
    })
    
    inv_path = os.path.join(output_dir, "inventory_snapshots.csv")
    inv_parquet = os.path.join(output_dir, "inventory_snapshots.parquet")
    df_inv.to_csv(inv_path, index=False)
    df_inv.to_parquet(inv_parquet, index=False)
    print(f"Saved {len(df_inv):,} inventory snapshots to {inv_path} and {inv_parquet}")
    print(f"Completed simulation in {time.time() - t0:.2f} seconds.")
    return df_sales, df_inv

if __name__ == "__main__":
    generate_store_master()
    generate_sku_master()
    generate_customer_master()
    generate_calendar()
    generate_daily_sales_and_inventory_fast()
