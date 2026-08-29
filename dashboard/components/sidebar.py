"""Phase 23 — Sidebar navigation."""

from __future__ import annotations

import streamlit as st

from dashboard.navigation import filtered_nav_groups, nav_label_map
from dashboard.session_auth import current_role, current_user_name, logout_user, page_allowed


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
    name = current_user_name()
    if name:
        st.markdown(f"**Welcome,** {name}")
    st.divider()


def render_navigation() -> str:
    labels = nav_label_map()
    role = current_role()

    if "foresight_page" not in st.session_state:
        st.session_state.foresight_page = "home"

    for group_name, items in filtered_nav_groups(role):
        st.markdown(f'<div class="foresight-section-header">{group_name}</div>', unsafe_allow_html=True)
        for item in items:
            label = labels[item.key]
            if st.button(
                label,
                key=f"nav_{item.key}",
                use_container_width=True,
                type="primary" if st.session_state.foresight_page == item.key else "secondary",
            ):
                st.session_state.foresight_page = item.key
                st.rerun()

    page = st.session_state.foresight_page
    if not page_allowed(page, role):
        st.session_state.foresight_page = "home"
        page = "home"

    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        logout_user()
        st.rerun()

    return page
