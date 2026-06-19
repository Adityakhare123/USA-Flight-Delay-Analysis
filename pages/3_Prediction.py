import json

import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import MODEL_FILE, METRICS_FILE
from src.data_loader import load_master_data
from src.model_utils import train_and_save_model, load_saved_model, predict_delay
from src.insight_utils import (
    add_analysis_columns,
    get_prediction_reason_breakdown,
)
from src.ui_utils import page_header

st.set_page_config(
    page_title="Delay Prediction | USA Flight Delay Analysis",
    page_icon="🤖",
    layout="wide",
)

page_header(
    "🤖 Flight Delay Prediction",
    "Predict delay and show data-based reasons behind the prediction.",
)

try:
    df = load_master_data()
    df = add_analysis_columns(df)

    st.sidebar.markdown("## Model Options")

    train_button = st.sidebar.button("Train / Retrain Model")

    if train_button:
        with st.spinner("Training model. Please wait..."):
            metrics = train_and_save_model(df)

        st.success("Model trained successfully.")
        st.metric("Model Accuracy", f"{metrics['accuracy'] * 100:.2f}%")

    model_package = load_saved_model()

    if model_package is None:
        st.warning("Model is not trained yet. Click **Train / Retrain Model** from sidebar first.")
    else:
        st.success("Model loaded successfully.")

        if METRICS_FILE.exists():
            with open(METRICS_FILE, "r") as f:
                metrics = json.load(f)

            st.metric("Saved Model Accuracy", f"{metrics['accuracy'] * 100:.2f}%")

    st.markdown("---")

    st.markdown("## Enter Flight Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        airline_options = sorted(df["Airline"].dropna().astype(str).unique().tolist())
        airline = st.selectbox("Airline", airline_options)

        airport_from_options = sorted(df["AirportFrom"].dropna().astype(str).unique().tolist())
        airport_from = st.selectbox("Source Airport", airport_from_options)

    with col2:
        airport_to_options = sorted(df["AirportTo"].dropna().astype(str).unique().tolist())
        airport_to = st.selectbox("Destination Airport", airport_to_options)

        day_options = sorted(df["DayOfWeek"].dropna().unique().tolist())
        day_of_week = st.selectbox("Day of Week", day_options)

    with col3:
        time = st.number_input(
            "Scheduled Time",
            min_value=int(df["Time"].min()),
            max_value=int(df["Time"].max()),
            value=int(df["Time"].median()),
            help="In this dataset, Time is usually minutes from midnight.",
        )

        length = st.number_input(
            "Flight Duration / Length",
            min_value=int(df["Length"].min()),
            max_value=int(df["Length"].max()),
            value=int(df["Length"].median()),
        )

    def get_airport_values(airport_code: str, airport_col: str, prefix: str) -> dict:
        temp = df[df[airport_col].astype(str) == str(airport_code)]

        type_col = f"type_{prefix}_airport"
        elevation_col = f"elevation_ft_{prefix}_airport"
        runway_col = f"runway_count_{prefix}_airport"

        values = {}

        if type_col in df.columns:
            if temp.empty:
                values[type_col] = "Unknown"
            else:
                values[type_col] = temp[type_col].mode()[0]

        if elevation_col in df.columns:
            if temp.empty:
                values[elevation_col] = float(df[elevation_col].median())
            else:
                values[elevation_col] = float(temp[elevation_col].median())

        if runway_col in df.columns:
            if temp.empty:
                values[runway_col] = float(df[runway_col].median())
            else:
                values[runway_col] = float(temp[runway_col].median())

        return values

    source_values = get_airport_values(airport_from, "AirportFrom", "source")
    dest_values = get_airport_values(airport_to, "AirportTo", "dest")

    input_data = {
        "Airline": airline,
        "AirportFrom": airport_from,
        "AirportTo": airport_to,
        "DayOfWeek": day_of_week,
        "Time": time,
        "Length": length,
        **source_values,
        **dest_values,
    }

    input_df = pd.DataFrame([input_data])

    st.markdown("### Selected Flight Input")
    st.dataframe(input_df, use_container_width=True)

    predict_button = st.button("Predict Delay", type="primary")

    if predict_button:
        if not MODEL_FILE.exists():
            st.error("Model not found. Please train the model first from the sidebar.")
            st.stop()

        result = predict_delay(input_df)

        prediction = result["prediction"]
        probability = result["probability"]

        st.markdown("---")
        st.markdown("## Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("✈️ Prediction: Flight is likely to be DELAYED.")
            else:
                st.success("✅ Prediction: Flight is likely to be ON TIME.")

        with col2:
            if probability is not None:
                st.metric("Prediction Confidence", f"{probability * 100:.2f}%")
            else:
                st.metric("Prediction Confidence", "Not available")

        st.markdown("---")
        st.markdown("## Why this prediction?")

        reason_df = get_prediction_reason_breakdown(df, input_data)

        if reason_df.empty:
            st.info("Reason breakdown could not be generated.")
        else:
            top_reason = reason_df.iloc[0]

            st.info(
                f"Top risk factor: **{top_reason['Reason']}** "
                f"with selected input delay rate of "
                f"**{top_reason['Selected Input Delay Rate (%)']}%**."
            )

            col3, col4 = st.columns(2)

            with col3:
                pie_df = reason_df[reason_df["Risk Score"] > 0].copy()

                if pie_df.empty:
                    pie_df = reason_df.copy()

                fig = px.pie(
                    pie_df,
                    names="Reason",
                    values="Risk Score",
                    hole=0.45,
                    title="Prediction Reason Breakdown",
                )

                st.plotly_chart(fig, use_container_width=True)

            with col4:
                fig = px.bar(
                    reason_df,
                    x="Selected Input Delay Rate (%)",
                    y="Reason",
                    orientation="h",
                    text="Selected Input Delay Rate (%)",
                    title="Selected Input Delay Rate by Factor",
                )

                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Reason Breakdown Table")
            st.dataframe(reason_df, use_container_width=True)

            st.markdown("### How to read this?")

            st.write(
                """
                The model predicts delayed or not delayed using the selected flight details.
                The reason breakdown compares your selected airline, route, airport, day, time,
                and duration with the overall dataset average delay rate.
                A higher risk gap means that factor historically had more delays.
                """
            )

except Exception as e:
    st.error("Prediction page could not be loaded.")
    st.exception(e)