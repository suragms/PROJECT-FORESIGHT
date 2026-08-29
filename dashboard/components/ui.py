"""Phase 23.4 — Shared dashboard UI helpers (no Markdown tables)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st


def page_header(title: str, description: str = "") -> None:
    st.title(title)
    if description:
        st.caption(description)


def show_empty(message: str = "No data available for the selected filters.") -> None:
    st.info(message)


def show_error(message: str) -> None:
    """User-facing error without stack traces or file paths."""
    clean = str(message).split("File ")[0].strip() or "Unable to load this view."
    if len(clean) > 280:
        clean = clean[:277] + "..."
    st.error(clean)


def na(value: Any, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return fallback
    return text


def kv_table(
    rows: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    field_col: str = "Field",
    value_col: str = "Value",
) -> None:
    """Render a key/value table with st.dataframe (never Markdown pipes)."""
    if isinstance(rows, Mapping):
        pairs = list(rows.items())
    else:
        pairs = list(rows)
    df = pd.DataFrame(
        [
            {field_col: str(k), value_col: na(v)}
            for k, v in pairs
        ]
    )
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            field_col: st.column_config.TextColumn(field_col, width="medium"),
            value_col: st.column_config.TextColumn(value_col, width="large"),
        },
    )


def safe_dataframe(df: pd.DataFrame | None, *, empty_message: str | None = None) -> None:
    if df is None or len(df) == 0:
        show_empty(empty_message or "No data available for the selected filters.")
        return
    out = df.copy()
    out = out.fillna("N/A")
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].astype(str).replace({"nan": "N/A", "None": "N/A", "<NA>": "N/A"})
    st.dataframe(out, use_container_width=True, hide_index=True)


def metric_row(items: Sequence[tuple[str, Any]]) -> None:
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, na(value))
