# Dataset page
import streamlit as st
import pandas as pd

from src.data_loader import load_master_data, build_master_dataset, dataset_summary
from src.ui_utils import page_header, show_metric_cards

st.set_page_config(
    page_title="Dataset | USA Flight Delay Analysis",
    page_icon="📂",
    layout="wide",
)

page_header(
    "📂 Dataset Overview",
    "Check dataset size, missing values, duplicate rows, and data preview.",
)

with st.sidebar:
    st.markdown("## Dataset Options")
    force_rebuild = st.button("Rebuild Processed Dataset")

try:
    if force_rebuild:
        with st.spinner("Rebuilding dataset from raw files..."):
            df = build_master_dataset()
        st.success("Dataset rebuilt successfully.")
    else:
        df = load_master_data()

    summary = dataset_summary(df)

    show_metric_cards(
        summary["rows"],
        summary["columns"],
        summary["missing_values"],
        summary["duplicate_rows"],
    )

    st.markdown("## Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("## Column Details")

    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(df[col].dtype) for col in df.columns],
            "Missing Values": [df[col].isna().sum() for col in df.columns],
            "Unique Values": [df[col].nunique() for col in df.columns],
        }
    )

    st.dataframe(column_info, use_container_width=True)

    st.markdown("## Delay Target Distribution")

    if "Delay" in df.columns:
        delay_counts = df["Delay"].value_counts().reset_index()
        delay_counts.columns = ["Delay", "Count"]
        delay_counts["Delay"] = delay_counts["Delay"].map({0: "Not Delayed", 1: "Delayed"})
        st.dataframe(delay_counts, use_container_width=True)
    else:
        st.warning("Delay column not found in dataset.")

except Exception as e:
    st.error("Dataset could not be loaded.")
    st.exception(e)