"""Phase 23.1 — Streamlit session authentication helpers."""

from __future__ import annotations

import streamlit as st

from src.auth.service import AuthError, get_auth_service

SESSION_KEYS = (
    "authenticated",
    "access_token",
    "user_id",
    "full_name",
    "email",
    "role",
    "foresight_page",
    "auth_mode",
)


def is_authenticated() -> bool:
    return bool(st.session_state.get("authenticated"))


def current_role() -> str:
    return st.session_state.get("role", "USER")


def current_user_name() -> str:
    return st.session_state.get("full_name", "")


def current_user_email() -> str:
    return st.session_state.get("email", "")


def login_user(email: str, password: str) -> None:
    user, token = get_auth_service().login(email, password)
    st.session_state.authenticated = True
    st.session_state.access_token = token
    st.session_state.user_id = user.id
    st.session_state.full_name = user.full_name
    st.session_state.email = user.email
    st.session_state.role = user.role
    st.session_state.foresight_page = "home"


def register_user(full_name: str, email: str, password: str, confirm_password: str) -> str:
    user = get_auth_service().register(full_name, email, password, confirm_password)
    return user.email


def logout_user() -> None:
    for key in SESSION_KEYS:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.auth_mode = "login"


def page_allowed(page_key: str, role: str) -> bool:
    """Any authenticated role may open any unified-app page."""
    return True
