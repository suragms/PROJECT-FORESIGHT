"""Phase 23 — Sidebar navigation."""

from __future__ import annotations

import streamlit as st

from dashboard.navigation import NAV_GROUPS, nav_label_map


def render_brand() -> None:
    st.markdown(
        """
<div class="foresight-brand">
  <div class="foresight-brand-title">📊 PROJECT FORESIGHT</div>
  <div class="foresight-brand-sub">AI-Powered Demand &amp;<br/>Inventory Intelligence</div>
  <div class="foresight-tagline">Forecast. Monitor. Optimize. Decide.</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def render_navigation() -> str:
    labels = nav_label_map()
    keys = list(labels.keys())
    display = [labels[k] for k in keys]

    if "foresight_page" not in st.session_state:
        st.session_state.foresight_page = "home"

    default_idx = keys.index(st.session_state.foresight_page) if st.session_state.foresight_page in keys else 0

    for group_name, items in NAV_GROUPS:
        st.markdown(f'<div class="foresight-section-header">{group_name}</div>', unsafe_allow_html=True)
        group_keys = [item.key for item in items]
        group_labels = [labels[k] for k in group_keys]
        for key, label in zip(group_keys, group_labels):
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if st.session_state.foresight_page == key else "secondary"):
                st.session_state.foresight_page = key
                st.rerun()

    return st.session_state.foresight_page
