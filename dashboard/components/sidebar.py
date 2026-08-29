"""Phase 23.5 — Professional left sidebar navigation."""

from __future__ import annotations

import streamlit as st

from dashboard.navigation import filtered_nav_groups, nav_label_map
from dashboard.session_auth import (
    current_role,
    current_user_email,
    current_user_name,
    logout_user,
    page_allowed,
)


def render_brand() -> None:
    st.markdown(
        """
<div class="fs-sidebar-brand">
  <div class="fs-brand-row">
    <div class="fs-brand-mark">F</div>
    <div class="fs-brand-text">
      <div class="fs-brand-title">PROJECT FORESIGHT</div>
      <div class="fs-brand-sub">AI-Powered Demand &amp;<br/>Inventory Intelligence</div>
    </div>
  </div>
</div>
<div class="fs-nav-label">Navigate to:</div>
        """,
        unsafe_allow_html=True,
    )


def render_user_footer() -> None:
    name = current_user_name() or "User"
    email = current_user_email() or ""
    initial = (name.strip()[:1] or "U").upper()
    st.markdown(
        f"""
<div class="fs-user-block">
  <div class="fs-user-row">
    <div class="fs-user-avatar">{initial}</div>
    <div class="fs-user-meta">
      <div class="fs-user-name">{name}</div>
      <div class="fs-user-email">{email}</div>
    </div>
  </div>
</div>
<div class="fs-sidebar-foot">
  PROJECT FORESIGHT<br/>
  Demand &amp; Inventory Intelligence
</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚪  Logout", key="nav_logout", use_container_width=True):
        logout_user()
        st.rerun()


def render_navigation() -> str:
    labels = nav_label_map()
    role = current_role()

    if "foresight_page" not in st.session_state:
        st.session_state.foresight_page = "home"

    for group_name, items in filtered_nav_groups(role):
        st.markdown(
            f'<div class="fs-section-header">{group_name}</div>',
            unsafe_allow_html=True,
        )
        for item in items:
            active = st.session_state.foresight_page == item.key
            if st.button(
                labels[item.key],
                key=f"nav_{item.key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.foresight_page = item.key
                st.rerun()

    page = st.session_state.foresight_page
    if not page_allowed(page, role):
        st.session_state.foresight_page = "home"
        page = "home"

    return page


def render_sidebar() -> str:
    """Full authenticated sidebar: brand → nav → user footer."""
    render_brand()
    page = render_navigation()
    st.markdown('<div class="fs-sidebar-spacer"></div>', unsafe_allow_html=True)
    render_user_footer()
    return page
