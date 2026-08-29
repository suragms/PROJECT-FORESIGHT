"""Phase 23.1 — Login and registration UI."""

from __future__ import annotations

import streamlit as st

from dashboard.session_auth import login_user, register_user
from src.auth.service import AuthError


def render_auth_screen() -> None:
    st.markdown(
        """
<div style="text-align:center; padding: 1rem 0 1.5rem 0;">
  <div style="font-size:1.6rem; font-weight:700; color:#0f172a;">📊 PROJECT FORESIGHT</div>
  <div style="font-size:0.95rem; color:#64748b; margin-top:0.35rem;">
    AI-Powered Demand &amp; Inventory Intelligence
  </div>
  <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.25rem; font-style:italic;">
    Forecast. Monitor. Optimize. Decide.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("#### Welcome")
        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            _render_login_form()

        with tab_register:
            _render_register_form()


def _render_login_form() -> None:
    st.markdown("##### Existing User")
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("📧 Email Address")
        password = st.text_input("🔒 Password", type="password")
        submitted = st.form_submit_button("LOGIN", use_container_width=True, type="primary")
        if submitted:
            if not email or not password:
                st.error("Email and password are required.")
                return
            try:
                login_user(email, password)
                st.success("Login successful.")
                st.rerun()
            except AuthError as exc:
                st.error(str(exc))
    st.caption("Don't have an account? Switch to the **Register** tab.")


def _render_register_form() -> None:
    st.markdown("##### New User")
    with st.form("register_form", clear_on_submit=True):
        full_name = st.text_input("👤 Full Name")
        email = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type="password")
        confirm = st.text_input("🔒 Confirm Password", type="password")
        submitted = st.form_submit_button("CREATE ACCOUNT", use_container_width=True, type="primary")
        if submitted:
            if not full_name.strip():
                st.error("Name is required.")
                return
            try:
                created_email = register_user(full_name, email, password, confirm)
                st.success(f"Registration successful. Account created for {created_email}. Please log in.")
                st.session_state.auth_mode = "login"
            except AuthError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(str(exc))
    st.caption("Already have an account? Switch to the **Login** tab.")
