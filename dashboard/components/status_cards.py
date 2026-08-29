"""Phase 23 — Status badges and metric cards."""

from __future__ import annotations

import streamlit as st


def status_badge(status: str) -> str:
    s = (status or "").upper()
    if s in ("PASS", "HEALTHY", "READY", "MEASURED"):
        cls = "status-pass"
    elif s in ("WARNING", "WATCH", "PARTIAL", "DEGRADED"):
        cls = "status-warning"
    elif s in ("FAIL", "CRITICAL"):
        cls = "status-fail"
    else:
        cls = "status-pending"
    return f'<span class="{cls}">{status}</span>'


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-card-label">{label}</div>'
        f'<div class="metric-card-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def validation_disclaimer() -> None:
    st.info(
        "**VALIDATION / BACKTEST METRICS** — Overall WAPE 13.96%, h1–h6 WAPE 11.03%. "
        "These are **not** live production performance. Live performance: **PENDING ACTUALS**."
    )
