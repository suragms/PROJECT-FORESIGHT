"""Phase 23 — Application theme and design system (Streamlit light)."""

import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stApp {
        background-color: #ffffff;
        color: #0f172a;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    .foresight-brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.3;
    }

    .foresight-brand-sub {
        font-size: 0.72rem;
        color: #64748b;
        line-height: 1.35;
    }

    .foresight-tagline {
        font-size: 0.68rem;
        color: #94a3b8;
        font-style: italic;
    }

    .foresight-section-header {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin: 1rem 0 0.35rem 0;
        text-transform: uppercase;
    }

    .status-pass {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-warning {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        background: #fef3c7;
        color: #92400e;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-fail {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        background: #fee2e2;
        color: #991b1b;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-pending {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        background: #e2e8f0;
        color: #475569;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .metric-card {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        min-height: 88px;
    }

    .metric-card-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .metric-card-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.35rem;
    }

    div[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: #e04343 !important;
        border-color: #e04343 !important;
    }

    div[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #334155 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    div[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #cbd5e1 !important;
    }

    div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #ff4b4b !important;
        border-bottom-color: #ff4b4b !important;
    }

    div[data-testid="stAlert"] div[data-baseweb="notification"] {
        background-color: #e8f4fd !important;
        color: #1e40af !important;
        border-radius: 8px !important;
    }

    hr {
        border-color: #e2e8f0 !important;
    }
</style>
        """,
        unsafe_allow_html=True,
    )
