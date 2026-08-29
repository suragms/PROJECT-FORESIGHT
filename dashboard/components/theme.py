"""Phase 23.5 — Application theme and professional sidebar design system."""

import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stApp {
        background-color: #ffffff;
        color: #0f172a;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* —— Sidebar shell —— */
    div[data-testid="stSidebar"] {
        background: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
        min-width: 260px !important;
        max-width: 280px !important;
    }

    div[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.75rem;
        padding-bottom: 1rem;
    }

    section[data-testid="stSidebar"] {
        width: 270px !important;
    }

    /* Brand */
    .fs-sidebar-brand {
        padding: 0.25rem 0.15rem 0.85rem 0.15rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 0.75rem;
    }

    .fs-brand-row {
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
    }

    .fs-brand-mark {
        width: 36px;
        height: 36px;
        border-radius: 9px;
        background: #ff4b4b;
        color: #fff;
        font-weight: 800;
        font-size: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        letter-spacing: -0.02em;
    }

    .fs-brand-title {
        font-size: 0.92rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        line-height: 1.25;
    }

    .fs-brand-sub {
        font-size: 0.68rem;
        color: #64748b;
        line-height: 1.35;
        margin-top: 0.15rem;
        font-weight: 500;
    }

    .fs-nav-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        margin: 0.35rem 0 0.55rem 0.15rem;
    }

    .fs-section-header {
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin: 0.85rem 0 0.3rem 0.15rem;
        text-transform: uppercase;
    }

    /* Nav buttons as clean list rows (no radio circles) */
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stSidebar"] .stButton > button {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        text-align: left !important;
        height: auto !important;
        min-height: 2.35rem;
        line-height: 1.3 !important;
    }

    div[data-testid="stSidebar"] .stButton {
        margin-bottom: 0.2rem;
    }

    div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: #ff4b4b !important;
        border: 1px solid #ff4b4b !important;
        border-left: 3px solid #b91c1c !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        padding: 0.45rem 0.7rem !important;
    }

    div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: #e04343 !important;
        border-color: #e04343 !important;
    }

    div[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-left: 3px solid transparent !important;
        color: #334155 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        padding: 0.45rem 0.7rem !important;
    }

    div[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #e2e8f0 !important;
        border-left: 3px solid #cbd5e1 !important;
    }

    /* User footer */
    .fs-sidebar-spacer {
        height: 0.75rem;
    }

    .fs-user-block {
        border-top: 1px solid #e2e8f0;
        padding-top: 0.85rem;
        margin-top: 0.5rem;
    }

    .fs-user-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.55rem;
    }

    .fs-user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 999px;
        background: #ff4b4b;
        color: #fff;
        font-weight: 700;
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .fs-user-name {
        font-size: 0.8rem;
        font-weight: 600;
        color: #0f172a;
        line-height: 1.2;
        word-break: break-word;
    }

    .fs-user-email {
        font-size: 0.68rem;
        color: #64748b;
        word-break: break-all;
        line-height: 1.25;
    }

    .fs-sidebar-foot {
        font-size: 0.62rem;
        color: #94a3b8;
        line-height: 1.35;
        margin: 0.35rem 0 0.5rem 0;
        text-align: center;
    }

    /* Legacy class aliases */
    .foresight-brand-title { font-size: 1.05rem; font-weight: 700; color: #1e293b; }
    .foresight-brand-sub { font-size: 0.72rem; color: #64748b; }
    .foresight-tagline { font-size: 0.68rem; color: #94a3b8; font-style: italic; }
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
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.35rem;
        word-break: break-word;
        overflow-wrap: anywhere;
        line-height: 1.35;
    }

    div[data-testid="stDataFrame"] {
        width: 100%;
        overflow-x: auto;
    }

    div[data-testid="stMetricValue"] {
        word-break: break-word;
        overflow-wrap: anywhere;
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

    /* Hide default radio circles if any radio nav remains */
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: transparent;
        border-radius: 8px;
        padding: 0.4rem 0.55rem;
    }
</style>
        """,
        unsafe_allow_html=True,
    )
