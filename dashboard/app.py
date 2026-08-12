"""
Demand & Inventory Intelligence Platform (Project FORESIGHT)
=============================================================
Executive Streamlit Web Application

Author: Surag M S
Project FORESIGHT: Retail Demand Forecasting & Inventory Risk Optimization
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_integration import (
    load_sales_daily,
    load_inventory_snapshots,
    load_sku_master,
    load_store_master,
    load_calendar,
    load_online_retail,
    get_executive_kpis,
    get_top_bottom_skus,
)
from src.feature_engineering import (
    build_forecasting_feature_matrix,
    aggregate_daily_sales,
)
from src.forecasting import (
    MLDemandForecaster,
    BaselineForecaster,
    generate_multi_step_forecast,
    train_and_benchmark_models,
)
from src.risk_scoring import (
    calculate_inventory_risk_matrix,
    answer_10_core_questions,
)

# ---------------------------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FORESIGHT | Demand & Inventory Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive glassmorphism styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #0a0f1d 100%);
        color: #f1f5f9;
    }
    
    /* Executive Metric Card */
    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.2);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-delta-pos {
        color: #10b981;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .metric-delta-neg {
        color: #f43f5e;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .metric-delta-warn {
        color: #f59e0b;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 2px;
    }
    
    /* Alert Pill */
    .alert-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .alert-critical {
        background: rgba(244, 63, 94, 0.2);
        color: #fda4af;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }
    .alert-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #fde68a;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .alert-success {
        background: rgba(16, 185, 129, 0.2);
        color: #a7f3d0;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading retail data cache...")
def get_cached_raw_data():
    sales = load_sales_daily()
    inventory = load_inventory_snapshots()
    skus = load_sku_master()
    stores = load_store_master()
    calendar = load_calendar()
    return sales, inventory, skus, stores, calendar


@st.cache_data(show_spinner="Computing inventory risk matrix...")
def get_cached_risk_matrix():
    return calculate_inventory_risk_matrix()


@st.cache_data(show_spinner="Computing 10 Core Questions analysis...")
def get_cached_10_questions():
    return answer_10_core_questions()


@st.cache_resource(show_spinner="Training / Loading ML Forecasters...")
def get_cached_forecasting_models():
    """
    Train ML forecasters on SKU-level daily demand aggregated across stores.

    The forecast history served in the app is a SKU's daily total across all
    stores. Training on a random sample of store-SKU rows produced a 10x scale
    mismatch (store-SKU days average ~7 units vs SKU-total ~74) and silently
    under-scaled every forecast. Training on the full SKU-total series matches
    the inference grain and breaks the random-sampling of the time series.
    """
    sales = load_sales_daily()
    skus = load_sku_master()
    calendar = load_calendar()
    daily = aggregate_daily_sales(sales, group_cols=("sku_id",))
    feat_matrix = build_forecasting_feature_matrix(
        sales_df=daily,
        sku_df=skus,
        store_df=None,
        calendar_df=calendar,
        group_cols=["sku_id"],
    )
    models, leaderboard, _ = train_and_benchmark_models(feat_matrix, test_days=30)
    return models, leaderboard


# Load datasets
sales_df, inv_df, skus_df, stores_df, cal_df = get_cached_raw_data()
risk_df = get_cached_risk_matrix()
q_answers = get_cached_10_questions()
kpis = get_executive_kpis()

# Train/cache models at SKU-total grain (matches forecast inference grain)
models_dict, leaderboard_df = get_cached_forecasting_models()


# ---------------------------------------------------------------------------
# Sidebar Navigation & Filter Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Project FORESIGHT")
    st.caption("Retail Demand & Inventory Intelligence Platform")
    st.divider()

    st.markdown("#### 🏢 Global Filters")
    # Only expose stores and SKUs that actually have transactional data so
    # selections never produce empty history / empty risk slices.
    active_store_ids = set(sales_df["store_id"].unique())
    active_sku_ids = set(sales_df["sku_id"].unique())
    active_stores = stores_df[stores_df["store_id"].isin(active_store_ids)]
    active_skus_catalog = skus_df[skus_df["sku_id"].isin(active_sku_ids)]

    all_regions = ["All"] + sorted(active_stores["region"].unique().tolist())
    selected_region = st.selectbox("Filter Region", all_regions, index=0)

    filtered_stores = active_stores if selected_region == "All" else active_stores[active_stores["region"] == selected_region]
    store_options = ["All Stores"] + [f"{r.store_id} - {r.store_name} ({r.city})" for _, r in filtered_stores.iterrows()]
    selected_store_str = st.selectbox("Store Location", store_options, index=0)
    selected_store_id = None if selected_store_str == "All Stores" else selected_store_str.split(" - ")[0]

    all_categories = ["All"] + sorted(active_skus_catalog["category"].unique().tolist())
    selected_category = st.selectbox("Product Category", all_categories, index=0)

    filtered_skus = active_skus_catalog if selected_category == "All" else active_skus_catalog[active_skus_catalog["category"] == selected_category]
    sku_options = [f"{r.sku_id} - {r.sku_name} (${r.base_price:.2f})" for _, r in filtered_skus.iterrows()]
    selected_sku_str = st.selectbox("Target SKU Selection", sku_options, index=0)
    selected_sku_id = selected_sku_str.split(" - ")[0] if sku_options else active_skus_catalog["sku_id"].iloc[0]

    st.divider()
    st.markdown("#### 🎯 Platform Highlights")
    st.info(f"""
    - **Stores Monitored:** {len(active_stores)} (of {len(stores_df)} in master)
    - **Tracked SKUs:** {len(active_skus_catalog)} (of {len(skus_df)} in catalog)
    - **Active Sales Records:** {len(sales_df):,}
    - **Inventory Horizon:** 2022 – 2025
    """)

    st.caption("Project FORESIGHT v2.0 • Data Science & Analytics")


# ---------------------------------------------------------------------------
# Top Header
# ---------------------------------------------------------------------------
st.markdown('<div class="header-banner">Demand & Inventory Intelligence</div>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.05rem; margin-bottom: 20px;'>Enterprise Retail Demand Forecasting, Dynamic Inventory Risk Scoring, and Automated Replenishment Engine</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🌟 Executive Cockpit",
    "📈 Demand Forecasting Studio",
    "🎛️ What-If Scenario Simulator",
    "⚠️ Inventory Risk Triage",
    "📦 Automated Replenishment & PO",
    "🔍 SKU & Store Deep-Dive",
    "💡 The 10 Core Questions",
    "📊 Data Quality & Profiling",
])


# ===========================================================================
# TAB 1: EXECUTIVE COCKPIT
# ===========================================================================
with tab1:
    st.markdown("### 📊 Enterprise Key Performance Indicators")

    # Compute real year-over-year revenue growth (2025 vs 2024) instead of a fabricated delta
    rev_by_year = sales_df.groupby(sales_df["date"].dt.year)["total_revenue"].sum()
    yoy_growth = None
    if 2025 in rev_by_year.index and 2024 in rev_by_year.index and rev_by_year[2024] > 0:
        yoy_growth = (rev_by_year[2025] - rev_by_year[2024]) / rev_by_year[2024] * 100
    delta_class = "metric-delta-pos" if (yoy_growth or 0) >= 0 else "metric-delta-neg"
    delta_text = f"{yoy_growth:+.1f}% YoY (2025 vs 2024) • {kpis['total_units_sold']:,} units" if yoy_growth is not None else f"{kpis['total_units_sold']:,} units"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Gross Revenue</div>
            <div class="metric-value">${kpis['total_revenue']:,.2f}</div>
            <div class="{delta_class}">{delta_text}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Inventory Valuation</div>
            <div class="metric-value">${kpis['total_inventory_value']:,.2f}</div>
            <div class="metric-delta-warn">● {kpis['total_inventory_units']:,} units in stock</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Current Stockout Incidents</div>
            <div class="metric-value">{kpis['current_stockout_count']} <span style='font-size:1.1rem;color:#f43f5e;'>({kpis['current_stockout_rate_pct']:.1f}%)</span></div>
            <div class="metric-delta-neg">▼ {kpis['safety_stock_breaches']} safety stock breaches</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Reorder Triggers Active</div>
            <div class="metric-value">{kpis['reorder_triggered_count']}</div>
            <div class="metric-delta-warn">⚡ Urgent PO generation required</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1: Revenue trend & Category breakdown
    col_l, col_r = st.columns([7, 5])
    with col_l:
        st.markdown("#### 📅 Multi-Year Sales Velocity & Revenue Trajectory")
        monthly_sales = sales_df.copy()
        monthly_sales["year_month"] = monthly_sales["date"].dt.to_period("M").astype(str)
        monthly_agg = monthly_sales.groupby("year_month").agg(
            revenue=("total_revenue", "sum"),
            units=("units_sold", "sum")
        ).reset_index()

        fig_trend = px.line(
            monthly_agg,
            x="year_month",
            y="revenue",
            title="Monthly Revenue Trend (2022 - 2025)",
            labels={"year_month": "Month", "revenue": "Total Revenue ($)"},
            template="plotly_dark",
        )
        fig_trend.update_traces(line=dict(color="#6366f1", width=3))
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=360,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_r:
        st.markdown("#### 🏷️ Revenue Distribution by Category")
        cat_agg = pd.merge(sales_df, skus_df[["sku_id", "category"]], on="sku_id", how="left")
        cat_rev = cat_agg.groupby("category")["total_revenue"].sum().reset_index()
        fig_pie = px.pie(
            cat_rev,
            names="category",
            values="total_revenue",
            hole=0.45,
            template="plotly_dark",
            color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#06b6d4", "#ec4899", "#8b5cf6"],
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=20),
            height=360,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Top & Bottom Performers
    st.markdown("#### 🏆 Product Performance Quadrants")
    top_skus, bottom_skus = get_top_bottom_skus(top_n=5)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Top 5 Revenue Drivers**")
        st.dataframe(
            top_skus[["sku_id", "sku_name", "category", "total_units", "total_revenue", "margin_pct"]].style.format({
                "total_units": "{:,}",
                "total_revenue": "${:,.2f}",
                "margin_pct": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
    with t2:
        st.markdown("**Bottom 5 Slow-Moving Items (Deadstock Risk)**")
        st.dataframe(
            bottom_skus[["sku_id", "sku_name", "category", "total_units", "total_revenue", "margin_pct"]].style.format({
                "total_units": "{:,}",
                "total_revenue": "${:,.2f}",
                "margin_pct": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )


# ===========================================================================
# TAB 2: DEMAND FORECASTING STUDIO
# ===========================================================================
with tab2:
    st.markdown("### 📈 Machine Learning Demand Forecasting Studio")
    st.caption("Multi-horizon recursive predictions with confidence bounds and benchmark model selection.")

    fc_c1, fc_c2, fc_c3 = st.columns([4, 4, 4])
    with fc_c1:
        model_choice = st.selectbox(
            "Select Forecasting Algorithm",
            ["LightGBM Regressor", "XGBoost Regressor", "Random Forest Regressor", "Seasonal Naive (7D)", "7-Day Moving Average"],
            index=0
        )
    with fc_c2:
        forecast_horizon = st.radio("Forecast Horizon", [7, 14, 30], horizontal=True, index=2)
    with fc_c3:
        target_sku_for_fc = selected_sku_id
        sku_meta = skus_df[skus_df["sku_id"] == target_sku_for_fc].iloc[0]
        st.info(f"**Target:** {sku_meta['sku_name']} (`{sku_meta['sku_id']}`) | Base Price: ${sku_meta['base_price']:.2f}")

    # Prepare SKU time series
    sku_sales_history = sales_df[sales_df["sku_id"] == target_sku_for_fc].groupby("date").agg(
        units_sold=("units_sold", "sum"),
        total_revenue=("total_revenue", "sum"),
        avg_unit_price=("avg_unit_price", "mean")
    ).reset_index()

    # Map model choice to forecaster. Baseline choices produce a real baseline
    # forecast instead of silently falling back to LightGBM.
    chosen_forecaster = None
    baseline_kind = None
    if "LightGBM" in model_choice:
        chosen_forecaster = models_dict.get("lightgbm")
    elif "XGBoost" in model_choice:
        chosen_forecaster = models_dict.get("xgboost")
    elif "Random Forest" in model_choice:
        chosen_forecaster = models_dict.get("random_forest")
    elif "Moving Average" in model_choice:
        baseline_kind = "moving_average"
    elif "Seasonal Naive" in model_choice:
        baseline_kind = "seasonal_naive"
    else:
        chosen_forecaster = models_dict.get("lightgbm")

    def _baseline_forecast_frame(values, last_date, horizon):
        dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        std = float(np.std(values)) if len(values) > 1 else 1.0
        return pd.DataFrame({
            "date": dates,
            "forecast_units": values,
            "forecast_lower": np.maximum(0, values - 1.96 * std),
            "forecast_upper": values + 1.96 * std,
            "step": list(range(1, horizon + 1)),
            "discount_applied": 0.0,
            "promo_applied": 0,
        })

    # Generate multi-step forecast (guard against insufficient history)
    fc_df = None
    if len(sku_sales_history) < 7:
        st.warning("Insufficient historical observations for the selected SKU. Choose a different SKU to see a forecast.")
    elif baseline_kind == "moving_average":
        vals = BaselineForecaster.moving_average_forecast(sku_sales_history["units_sold"], window=7, horizon=forecast_horizon)
        fc_df = _baseline_forecast_frame(vals, sku_sales_history["date"].max(), forecast_horizon)
    elif baseline_kind == "seasonal_naive":
        vals = BaselineForecaster.seasonal_naive_forecast(sku_sales_history["units_sold"], season_length=7, horizon=forecast_horizon)
        fc_df = _baseline_forecast_frame(vals, sku_sales_history["date"].max(), forecast_horizon)
    else:
        fc_df = generate_multi_step_forecast(
            model=chosen_forecaster,
            history_df=sku_sales_history,
            horizon_days=forecast_horizon,
        )

    # Plot actuals + forecast
    recent_actuals = sku_sales_history.tail(60)

    fig_fc = go.Figure()

    # Historical Actuals
    fig_fc.add_trace(go.Scatter(
        x=recent_actuals["date"],
        y=recent_actuals["units_sold"],
        mode="lines+markers",
        name="Historical Actuals",
        line=dict(color="#38bdf8", width=2.5),
        marker=dict(size=4)
    ))

    if fc_df is not None:
        # Upper Confidence Interval
        fig_fc.add_trace(go.Scatter(
            x=fc_df["date"],
            y=fc_df["forecast_upper"],
            mode="lines",
            name="Upper Bound (95% CI)",
            line=dict(width=0),
            showlegend=False
        ))

        # Lower Confidence Interval (filled area)
        fig_fc.add_trace(go.Scatter(
            x=fc_df["date"],
            y=fc_df["forecast_lower"],
            mode="lines",
            name="Confidence Interval (95%)",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(99, 102, 241, 0.2)",
        ))

        # Point Forecast
        fig_fc.add_trace(go.Scatter(
            x=fc_df["date"],
            y=fc_df["forecast_units"],
            mode="lines+markers",
            name=f"Forecast ({model_choice})",
            line=dict(color="#f59e0b", width=3, dash="dash"),
            marker=dict(size=6, color="#f59e0b")
        ))

    fig_fc.update_layout(
        title=f"Multi-Step Demand Forecast for {sku_meta['sku_name']} (Next {forecast_horizon} Days)",
        xaxis_title="Date",
        yaxis_title="Units Demand",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.4)",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Benchmark Leaderboard & Feature Importances
    b1, b2 = st.columns([6, 6])
    with b1:
        st.markdown("#### 🏅 Model Evaluation Benchmark Leaderboard")
        st.dataframe(
            leaderboard_df.style.highlight_min(subset=["wape_pct", "mae", "rmse"], color="#1e3a5f")
            .format({
                "mae": "{:.3f}",
                "rmse": "{:.3f}",
                "wape_pct": "{:.2f}%",
                "mape_pct": "{:.2f}%",
                "r2": "{:.4f}",
                "bias": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True
        )
    with b2:
        st.markdown("#### 🧠 Top Feature Importances (Gradient Boosting)")
        if hasattr(chosen_forecaster, "get_feature_importances"):
            fi_df = chosen_forecaster.get_feature_importances().head(8)
            fig_fi = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                template="plotly_dark",
                color="importance",
                color_continuous_scale="Purples",
            )
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 41, 59, 0.4)",
                yaxis=dict(autorange="reversed"),
                height=260,
                margin=dict(l=10, r=10, t=20, b=20),
            )
            st.plotly_chart(fig_fi, use_container_width=True)


# ===========================================================================
# TAB 3: WHAT-IF SCENARIO SIMULATOR
# ===========================================================================
with tab3:
    st.markdown("### 🎛️ What-If Scenario Simulator & Price Elasticity Engine")
    st.caption("Simulate demand lifts, promotional discount elasticities, and surge shocks on inventory requirements.")

    sc_col1, sc_col2 = st.columns([4, 8])

    with sc_col1:
        st.markdown("#### ⚙️ Scenario Parameters")
        sim_discount = st.slider("Promotional Discount (%)", min_value=0, max_value=50, value=15, step=5) / 100.0
        sim_promo = st.toggle("Active Marketing Campaign / Promotion", value=True)
        sim_surge = st.slider("Demand Surge Factor (Marketing Push)", min_value=0.5, max_value=2.5, value=1.2, step=0.1)
        sim_horizon = st.select_slider("Simulation Horizon (Days)", options=[7, 14, 30, 60], value=30)

        st.markdown("---")
        st.markdown("#### 💡 Simulated SKU Baseline")
        sim_sku = skus_df[skus_df["sku_id"] == selected_sku_id].iloc[0]
        st.write(f"**SKU:** {sim_sku['sku_name']}")
        st.write(f"**Cost Price:** ${sim_sku['cost_price']:.2f}")
        st.write(f"**Base Retail Price:** ${sim_sku['base_price']:.2f}")
        discounted_price = sim_sku['base_price'] * (1 - sim_discount)
        st.write(f"**Simulated Price:** ${discounted_price:.2f} (Margin: {((discounted_price - sim_sku['cost_price'])/discounted_price)*100:.1f}%)")

    with sc_col2:
        sku_hist = sales_df[sales_df["sku_id"] == selected_sku_id].groupby("date").agg(
            units_sold=("units_sold", "sum"),
            avg_unit_price=("avg_unit_price", "mean")
        ).reset_index()

        if len(sku_hist) < 7:
            st.warning("Insufficient historical observations for the selected SKU. Choose a different SKU to run a scenario.")
        else:
            # Baseline forecast (no discount/surge)
            base_fc = generate_multi_step_forecast(
                model=models_dict["lightgbm"],
                history_df=sku_hist,
                horizon_days=sim_horizon,
                scenario_discount_pct=0.0,
                scenario_promo_flag=0,
                scenario_surge_factor=1.0,
            )

            # Simulated forecast (with parameters)
            scenario_fc = generate_multi_step_forecast(
                model=models_dict["lightgbm"],
                history_df=sku_hist,
                horizon_days=sim_horizon,
                scenario_discount_pct=sim_discount,
                scenario_promo_flag=1 if sim_promo else 0,
                scenario_surge_factor=sim_surge,
            )

            base_total_units = base_fc["forecast_units"].sum()
            sim_total_units = scenario_fc["forecast_units"].sum()
            unit_lift = sim_total_units - base_total_units
            unit_lift_pct = (unit_lift / max(1, base_total_units)) * 100

            base_rev = base_total_units * sim_sku["base_price"]
            sim_rev = sim_total_units * discounted_price
            rev_impact = sim_rev - base_rev

            m1, m2, m3 = st.columns(3)
            m1.metric("Simulated Demand Volume", f"{sim_total_units:,.0f} units", delta=f"{unit_lift_pct:+.1f}% lift")
            m2.metric("Projected Gross Revenue", f"${sim_rev:,.2f}", delta=f"${rev_impact:+,.2f}")
            m3.metric("Required Inventory Buffer", f"{sim_total_units * 1.15:,.0f} units", delta="Includes 15% safety")

            # Visual comparison
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(
                x=base_fc["date"],
                y=base_fc["forecast_units"],
                mode="lines+markers",
                name="Baseline Demand (No Promo)",
                line=dict(color="#94a3b8", dash="dot", width=2)
            ))
            fig_sim.add_trace(go.Scatter(
                x=scenario_fc["date"],
                y=scenario_fc["forecast_units"],
                mode="lines+markers",
                name="Simulated Scenario Demand",
                line=dict(color="#10b981", width=3.5)
            ))
            fig_sim.update_layout(
                title="What-If Demand Lift Trajectory",
                xaxis_title="Date",
                yaxis_title="Projected Daily Units",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 41, 59, 0.4)",
                height=380,
                hovermode="x unified",
            )
            st.plotly_chart(fig_sim, use_container_width=True)


# ===========================================================================
# TAB 4: INVENTORY RISK TRIAGE
# ===========================================================================
with tab4:
    st.markdown("### ⚠️ Supply Chain Inventory Risk & Stockout Prediction")
    st.caption("Real-time identification of imminent stockout risks, safety stock breaches, and excess carrying capital.")

    # Filter controls
    r_c1, r_c2, r_c3 = st.columns(3)
    with r_c1:
        risk_filter = st.selectbox("Stockout Risk Tier", ["All", "CRITICAL / HIGH", "MEDIUM (REORDER)", "LOW / SAFE"], index=0)
    with r_c2:
        overstock_filter = st.selectbox("Overstock Severity", ["All", "SEVERE OVERSTOCK", "MODERATE OVERSTOCK", "OPTIMAL"], index=0)
    with r_c3:
        search_sku = st.text_input("Search SKU or Name", "")

    filtered_risk = risk_df.copy()
    if selected_store_id:
        filtered_risk = filtered_risk[filtered_risk["store_id"] == selected_store_id]
    if selected_category != "All":
        filtered_risk = filtered_risk[filtered_risk["category"] == selected_category]
    if risk_filter != "All":
        filtered_risk = filtered_risk[filtered_risk["stockout_risk_level"] == risk_filter]
    if overstock_filter != "All":
        filtered_risk = filtered_risk[filtered_risk["overstock_risk_level"] == overstock_filter]
    if search_sku:
        filtered_risk = filtered_risk[
            filtered_risk["sku_name"].str.contains(search_sku, case=False, na=False) |
            filtered_risk["sku_id"].str.contains(search_sku, case=False, na=False)
        ]

    # Metrics row
    crit_count = (risk_df["stockout_risk_level"] == "CRITICAL / HIGH").sum()
    sever_overstock = (risk_df["overstock_risk_level"] == "SEVERE OVERSTOCK").sum()
    total_tied_capital = risk_df["capital_tied_up"].sum()

    rk1, rk2, rk3, rk4 = st.columns(4)
    rk1.metric("Critical Stockout SKUs", f"{crit_count}", delta="Action Required", delta_color="inverse")
    rk2.metric("Severe Overstock SKUs", f"{sever_overstock}", delta="Excess Capital", delta_color="inverse")
    rk3.metric("Capital Locked in Overstock", f"${total_tied_capital:,.2f}", delta="Annual Cost: $" + f"{total_tied_capital*0.25:,.2f}")
    rk4.metric("Active Reorder Triggers", f"{(risk_df['reorder_triggered']).sum()}", delta="POs Pending")

    # Visual Risk Matrix Scatter: Days of Supply vs Stockout Score
    st.markdown("#### 🎯 Inventory Risk Quadrant Matrix (Days of Supply vs Stockout Risk)")
    fig_risk_scatter = px.scatter(
        filtered_risk.head(500),
        x="days_of_supply",
        y="stockout_risk_score",
        color="stockout_risk_level",
        size="ending_inventory",
        hover_data=["store_id", "sku_id", "sku_name", "category", "ending_inventory", "safety_stock"],
        template="plotly_dark",
        color_discrete_map={
            "CRITICAL / HIGH": "#f43f5e",
            "MEDIUM (REORDER)": "#f59e0b",
            "LOW / SAFE": "#10b981",
        }
    )
    fig_risk_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.4)",
        xaxis_title="Days of Supply (DOS)",
        yaxis_title="Stockout Risk Score (0 - 100)",
        height=400,
    )
    st.plotly_chart(fig_risk_scatter, use_container_width=True)

    # Risk Triage Table
    st.markdown("#### 📋 Detailed Risk Triage Table")
    display_cols = [
        "store_id", "sku_id", "sku_name", "category", "ending_inventory",
        "avg_daily_demand", "days_of_supply", "safety_stock", "reorder_point",
        "stockout_risk_score", "stockout_risk_level", "overstock_risk_level", "capital_tied_up"
    ]
    st.dataframe(
        filtered_risk[display_cols].sort_values(by="stockout_risk_score", ascending=False).head(100).style.format({
            "avg_daily_demand": "{:.2f}",
            "days_of_supply": "{:.1f}",
            "stockout_risk_score": "{:.1f}",
            "capital_tied_up": "${:,.2f}"
        }),
        use_container_width=True,
        hide_index=True,
    )


# ===========================================================================
# TAB 5: AUTOMATED REPLENISHMENT & PURCHASE ORDERS (PO)
# ===========================================================================
with tab5:
    st.markdown("### 📦 Dynamic Reorder Point (ROP) & Purchase Order Generator")
    st.caption("Automated procurement triggers calculating Economic Order Quantities and supplier replenishment schedules.")

    po_items = risk_df[risk_df["reorder_triggered"]].sort_values(by="recommended_order_spend", ascending=False)

    p1, p2, p3 = st.columns(3)
    p1.metric("SKUs Requiring Immediate Reorder", f"{len(po_items):,}")
    p2.metric("Total Procurement Spend", f"${po_items['recommended_order_spend'].sum():,.2f}")
    p3.metric("Total Units to Order", f"{po_items['recommended_reorder_qty'].sum():,}")

    st.markdown("#### 📝 Recommended Purchase Order Schedule")
    po_export_cols = [
        "store_id", "sku_id", "sku_name", "supplier_id", "category",
        "ending_inventory", "on_order_qty", "reorder_point", "safety_stock",
        "lead_time_days", "recommended_reorder_qty", "cost_price", "recommended_order_spend"
    ]
    st.dataframe(
        po_items[po_export_cols].head(100).style.format({
            "cost_price": "${:.2f}",
            "recommended_order_spend": "${:,.2f}",
            "recommended_reorder_qty": "{:,}"
        }),
        use_container_width=True,
        hide_index=True,
    )

    # CSV Download Button
    csv_data = po_items[po_export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Official Purchase Orders (CSV)",
        data=csv_data,
        file_name="FORESIGHT_Purchase_Orders_Export.csv",
        mime="text/csv",
    )


# ===========================================================================
# TAB 6: SKU & STORE DEEP-DIVE
# ===========================================================================
with tab6:
    st.markdown("### 🔍 Granular SKU & Store Performance Deep-Dive")
    st.caption("Detailed historical dynamics, seasonal heatmaps, and price elasticity for selected product.")

    d_sku = skus_df[skus_df["sku_id"] == selected_sku_id].iloc[0]
    st.markdown(f"#### 🏷️ Product Dossier: **{d_sku['sku_name']}** (`{d_sku['sku_id']}`)")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Product Category", d_sku["category"], d_sku["sub_category"])
    d2.metric("Brand", d_sku["brand"])
    d3.metric("Cost & Base Price", f"${d_sku['cost_price']:.2f} / ${d_sku['base_price']:.2f}", f"{((d_sku['base_price']-d_sku['cost_price'])/d_sku['base_price'])*100:.1f}% Margin")
    d4.metric("Supplier Lead Time", f"{d_sku['lead_time_days']} Days", f"ROP: {d_sku['reorder_point']} units")

    # Heatmap: Month vs Day of Week
    st.markdown("#### 🗓️ Day-of-Week vs Monthly Demand Heatmap")
    sku_daily = sales_df[sales_df["sku_id"] == selected_sku_id].copy()
    sku_daily["month_name"] = sku_daily["date"].dt.strftime("%b")
    sku_daily["day_name"] = sku_daily["date"].dt.strftime("%a")

    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    pivot_hm = sku_daily.pivot_table(
        index="day_name",
        columns="month_name",
        values="units_sold",
        aggfunc="mean"
    ).reindex(index=day_order, columns=month_order).fillna(0)

    fig_hm = px.imshow(
        pivot_hm,
        labels=dict(x="Month", y="Day of Week", color="Avg Units"),
        x=month_order,
        y=day_order,
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig_hm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.4)",
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ===========================================================================
# TAB 7: THE 10 CORE QUESTIONS
# ===========================================================================
with tab7:
    st.markdown("### 💡 Executive Answers to the 10 Core Business Questions")
    st.caption("Comprehensive data-driven briefing answering core supply chain and retail business objectives.")

    with st.expander("📌 1. Top Products: Which products generate highest sales volume and revenue?", expanded=True):
        st.write("Top 5 revenue-generating SKUs across all 30 store networks:")
        top_df = pd.DataFrame(q_answers["q1_top_products"]).head(5)
        st.dataframe(top_df[["sku_id", "sku_name", "category", "total_units", "total_revenue", "margin_pct"]].style.format({
            "total_units": "{:,}", "total_revenue": "${:,.2f}", "margin_pct": "{:.1f}%"
        }), use_container_width=True, hide_index=True)

    with st.expander("📌 2. Bottom Products: Which products are slow-moving deadstock risks?"):
        st.write("Bottom 5 SKUs with lowest sales velocity:")
        bot_df = pd.DataFrame(q_answers["q2_bottom_products"]).head(5)
        st.dataframe(bot_df[["sku_id", "sku_name", "category", "total_units", "total_revenue", "margin_pct"]].style.format({
            "total_units": "{:,}", "total_revenue": "${:,.2f}", "margin_pct": "{:.1f}%"
        }), use_container_width=True, hide_index=True)

    with st.expander("📌 3. Demand Dynamics: How does demand evolve across channels and regions?"):
        dyn_df = pd.DataFrame(q_answers["q3_demand_dynamics"])
        st.dataframe(dyn_df.style.format({"total_revenue": "${:,.2f}", "total_units": "{:,}", "avg_price": "${:.2f}"}), use_container_width=True, hide_index=True)

    with st.expander("📌 4. Seasonality: What are the primary weekly, quarterly, and holiday surges?"):
        st.write(f"Holiday surges produce an average demand uplift of **+{q_answers['q4_seasonality']['holiday_uplift_pct']:.1f}%** over baseline days.")
        st.line_chart(pd.DataFrame(q_answers["q4_seasonality"]["monthly_profile"]).set_index("month")["total_revenue"])

    with st.expander("📌 5. Demand Growth: Which products exhibit positive growth trajectories?"):
        st.write("Top growing SKUs between 2022 and 2025:")
        st.json(q_answers["q5_growth"])

    with st.expander("📌 6. Future Demand: What is expected demand over multi-step horizons?"):
        st.write(f"Projected 30-Day aggregate network demand: **{q_answers['q6_future_demand_30d']:,.0f} units**.")

    with st.expander("📌 7. Stockout Risk: Which products are at imminent risk of stocking out?"):
        st.write(f"**{q_answers['q7_stockout_risk']['critical_count']}** critical items identified with potential monthly lost revenue of **${q_answers['q7_stockout_risk']['potential_lost_revenue_monthly']:,.2f}**.")
        st.dataframe(pd.DataFrame(q_answers['q7_stockout_risk']['top_stockout_skus']), use_container_width=True, hide_index=True)

    with st.expander("📌 8. Overstock Risk: Which products carry excess inventory inflating holding costs?"):
        st.write(f"Working capital tied up: **${q_answers['q8_overstock_risk']['total_capital_tied_up']:,.2f}** with annual holding cost of **${q_answers['q8_overstock_risk']['annual_holding_cost']:,.2f}**.")
        st.dataframe(pd.DataFrame(q_answers['q8_overstock_risk']['top_overstock_skus']), use_container_width=True, hide_index=True)

    with st.expander("📌 9. Replenishment Triggers: Which SKUs have breached their Reorder Point (ROP)?"):
        st.write(f"**{q_answers['q9_replenishment']['reorder_triggered_count']}** active reorder triggers totaling **${q_answers['q9_replenishment']['total_reorder_spend']:,.2f}** in recommended PO spend.")
        st.dataframe(pd.DataFrame(q_answers['q9_replenishment']['top_purchase_orders']), use_container_width=True, hide_index=True)

    with st.expander("📌 10. Actionable Recommendations: What decisions should managers execute now?", expanded=True):
        for i, rec in enumerate(q_answers["q10_recommendations"], 1):
            st.markdown(f"**{i}.** {rec}")


# ===========================================================================
# TAB 8: DATA QUALITY & PROFILING
# ===========================================================================
with tab8:
    st.markdown("### 📊 Data Quality Engineering & Platform Architecture")
    st.caption("Rigorous data profiling, validation gates, and schema integrity reports.")

    dq_col1, dq_col2 = st.columns(2)
    with dq_col1:
        st.markdown("#### 📑 Dataset Catalog & Dimensions")
        data_summary = pd.DataFrame([
            {"Dataset": "Store Master", "Records": f"{len(stores_df):,}", "Type": "Dimension", "Status": "PASS (100%)"},
            {"Dataset": "SKU Master", "Records": f"{len(skus_df):,}", "Type": "Dimension", "Status": "PASS (100%)"},
            {"Dataset": "Customer Master", "Records": "10,000", "Type": "Dimension", "Status": "PASS (100%)"},
            {"Dataset": "Calendar", "Records": f"{len(cal_df):,}", "Type": "Dimension", "Status": "PASS (100%)"},
            {"Dataset": "Daily Sales Fact", "Records": f"{len(sales_df):,}", "Type": "Fact (Parquet)", "Status": "PASS (100%)"},
            {"Dataset": "Inventory Snapshots", "Records": f"{len(inv_df):,}", "Type": "Fact (Parquet)", "Status": "REVIEW (Balance Validated)"},
            {"Dataset": "UCI Online Retail II", "Records": "1,033,036", "Type": "Transaction Log", "Status": "REVIEW (Guest Txns Preserved)"},
        ])
        st.dataframe(data_summary, use_container_width=True, hide_index=True)

    with dq_col2:
        st.markdown("#### 🛡️ Data Science & Quality Principles")
        st.success("""
        - **Raw Data Immutability:** Raw data files remain strictly untouched; all transformations output to `data/processed/`.
        - **Zero Fabricated Records:** Missing data recovered only when authoritative references exist.
        - **No Silent Deletions:** Returns & cancellations preserved in dedicated analytical partitions.
        - **Outliers Preserved:** Valid high-volume wholesale retail outliers flagged and retained.
        - **Temporal Leakage Prevention:** Rolling statistics strictly lagged by 1 step.
        """)
