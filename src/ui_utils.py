# Data preprocessing utilities
import streamlit as st


def page_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def show_metric_cards(total_rows, total_columns, missing_values, duplicate_rows):
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Rows", f"{total_rows:,}")
    col2.metric("Total Columns", total_columns)
    col3.metric("Missing Values", f"{missing_values:,}")
    col4.metric("Duplicate Rows", f"{duplicate_rows:,}")