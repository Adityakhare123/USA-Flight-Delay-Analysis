import streamlit as st
import plotly.express as px

from src.data_loader import load_master_data
from src.ui_utils import page_header
from src.insight_utils import (
    add_analysis_columns,
    get_group_delay_rate,
    get_route_delay_rate,
    get_actual_reason_summary,
    get_proxy_reason_summary,
)

st.set_page_config(
    page_title="EDA Dashboard | USA Flight Delay Analysis",
    page_icon="📊",
    layout="wide",
)

page_header(
    "📊 EDA Dashboard",
    "Visual analysis of delay patterns, time impact, airline performance, airport impact, and delay reason factors.",
)

try:
    df = load_master_data()
    df = add_analysis_columns(df)

    if "Delay" not in df.columns:
        st.error("Delay column not found in dataset.")
        st.stop()

    st.sidebar.markdown("## Filters")

    filtered_df = df.copy()

    if "Airline" in df.columns:
        airlines = sorted(df["Airline"].dropna().astype(str).unique().tolist())

        selected_airlines = st.sidebar.multiselect(
            "Select Airlines",
            airlines,
            default=airlines[:6] if len(airlines) > 6 else airlines,
        )

        if selected_airlines:
            filtered_df = filtered_df[filtered_df["Airline"].astype(str).isin(selected_airlines)]

    if "AirportFrom" in df.columns:
        source_airports = sorted(df["AirportFrom"].dropna().astype(str).unique().tolist())

        selected_source_airport = st.sidebar.selectbox(
            "Source Airport",
            ["All"] + source_airports,
        )

        if selected_source_airport != "All":
            filtered_df = filtered_df[
                filtered_df["AirportFrom"].astype(str) == selected_source_airport
            ]

    if "AirportTo" in df.columns:
        dest_airports = sorted(df["AirportTo"].dropna().astype(str).unique().tolist())

        selected_dest_airport = st.sidebar.selectbox(
            "Destination Airport",
            ["All"] + dest_airports,
        )

        if selected_dest_airport != "All":
            filtered_df = filtered_df[
                filtered_df["AirportTo"].astype(str) == selected_dest_airport
            ]

    st.markdown("## Overall Delay Summary")

    total_flights = len(filtered_df)
    delayed_flights = int(filtered_df["Delay"].sum())
    not_delayed_flights = total_flights - delayed_flights
    delay_rate = (delayed_flights / total_flights * 100) if total_flights else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Flights", f"{total_flights:,}")
    col2.metric("Delayed Flights", f"{delayed_flights:,}")
    col3.metric("Not Delayed", f"{not_delayed_flights:,}")
    col4.metric("Delay Rate", f"{delay_rate:.2f}%")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Delay Overview",
            "Time & Duration Analysis",
            "Airline & Airport Analysis",
            "Delay Reason Factors",
        ]
    )

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Delayed vs Not Delayed")

            delay_count = (
                filtered_df["Delay_Label"]
                .value_counts()
                .reset_index()
            )

            delay_count.columns = ["Delay Status", "Count"]

            fig = px.pie(
                delay_count,
                names="Delay Status",
                values="Count",
                hole=0.45,
                title="Flight Delay Distribution",
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Delay Rate by Day of Week")

            if "DayOfWeek" in filtered_df.columns:
                day_df = get_group_delay_rate(
                    filtered_df,
                    "DayOfWeek",
                    min_flights=1,
                    top_n=10,
                ).sort_values("DayOfWeek")

                fig = px.bar(
                    day_df,
                    x="DayOfWeek",
                    y="Delay Rate (%)",
                    text="Delay Rate (%)",
                    hover_data=["total_flights", "delayed_flights"],
                    title="Average Delay Rate by Day of Week",
                )

                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig.update_layout(yaxis_title="Delay Rate (%)")

                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Dataset Preview After Filters")
        st.dataframe(filtered_df.head(100), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Delay Rate by Time Period")

            if "Time_Period" in filtered_df.columns:
                time_df = get_group_delay_rate(
                    filtered_df,
                    "Time_Period",
                    min_flights=1,
                    top_n=10,
                )

                order = ["Late Night", "Morning", "Afternoon", "Evening / Night", "Unknown"]
                time_df["Time_Period"] = time_df["Time_Period"].astype(str)
                time_df["sort_order"] = time_df["Time_Period"].apply(
                    lambda x: order.index(x) if x in order else 99
                )
                time_df = time_df.sort_values("sort_order")

                fig = px.bar(
                    time_df,
                    x="Time_Period",
                    y="Delay Rate (%)",
                    text="Delay Rate (%)",
                    hover_data=["total_flights", "delayed_flights"],
                    title="Delay Rate by Flight Time Period",
                )

                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Delayed Flights by Time Period")

            if "Time_Period" in filtered_df.columns:
                delayed_time = filtered_df[filtered_df["Delay"] == 1]

                time_count = (
                    delayed_time["Time_Period"]
                    .value_counts()
                    .reset_index()
                )

                time_count.columns = ["Time Period", "Delayed Flights"]

                fig = px.pie(
                    time_count,
                    names="Time Period",
                    values="Delayed Flights",
                    hole=0.45,
                    title="Share of Delayed Flights by Time Period",
                )

                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### Delay Rate by Flight Duration")

            if "duration_category" in filtered_df.columns:
                duration_df = get_group_delay_rate(
                    filtered_df,
                    "duration_category",
                    min_flights=1,
                    top_n=10,
                )

                fig = px.bar(
                    duration_df,
                    x="duration_category",
                    y="Delay Rate (%)",
                    text="Delay Rate (%)",
                    hover_data=["total_flights", "delayed_flights"],
                    title="Delay Rate by Flight Duration Category",
                )

                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

                st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown("### Flight Length Distribution")

            if "Length" in filtered_df.columns:
                fig = px.histogram(
                    filtered_df,
                    x="Length",
                    color="Delay_Label",
                    nbins=40,
                    title="Flight Length Distribution by Delay Status",
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Top Airlines by Delay Rate")

            if "Airline" in filtered_df.columns:
                airline_df = get_group_delay_rate(
                    filtered_df,
                    "Airline",
                    min_flights=50,
                    top_n=15,
                )

                fig = px.bar(
                    airline_df,
                    x="Airline",
                    y="Delay Rate (%)",
                    text="Delay Rate (%)",
                    hover_data=["total_flights", "delayed_flights"],
                    title="Top Airlines with Highest Delay Rate",
                )

                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Top Routes by Delay Rate")

            route_df = get_route_delay_rate(
                filtered_df,
                top_n=15,
                min_flights=30,
            )

            if not route_df.empty:
                fig = px.bar(
                    route_df,
                    x="Route",
                    y="Delay Rate (%)",
                    text="Delay Rate (%)",
                    hover_data=["total_flights", "delayed_flights"],
                    title="Top Routes with Highest Delay Rate",
                )

                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig.update_layout(xaxis_tickangle=-45)

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough route data available after filters.")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### Top Source Airports by Delay Rate")

            if "AirportFrom" in filtered_df.columns:
                source_df = get_group_delay_rate(
                    filtered_df,
                    "AirportFrom",
                    min_flights=50,
                    top_n=15,
                )

                fig = px.bar(
                    source_df,
                    x="AirportFrom",
                    y="Delay Rate (%)",
                    text="Delay Rate (%)",
                    hover_data=["total_flights", "delayed_flights"],
                    title="Source Airports with Highest Delay Rate",
                )

                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

                st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown("### Top Destination Airports by Delay Rate")

            if "AirportTo" in filtered_df.columns:
                dest_df = get_group_delay_rate(
                    filtered_df,
                    "AirportTo",
                    min_flights=50,
                    top_n=15,
                )

                fig = px.bar(
                    dest_df,
                    x="AirportTo",
                    y="Delay Rate (%)",
                    text="Delay Rate (%)",
                    hover_data=["total_flights", "delayed_flights"],
                    title="Destination Airports with Highest Delay Rate",
                )

                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

                st.plotly_chart(fig, use_container_width=True)

        col5, col6 = st.columns(2)

        with col5:
            st.markdown("### Source Airport Type Impact")

            if "type_source_airport" in filtered_df.columns:
                source_type_df = get_group_delay_rate(
                    filtered_df,
                    "type_source_airport",
                    min_flights=1,
                    top_n=10,
                )

                fig = px.bar(
                    source_type_df,
                    x="type_source_airport",
                    y="Delay Rate (%)",
                    text="Delay Rate (%)",
                    hover_data=["total_flights", "delayed_flights"],
                    title="Delay Rate by Source Airport Type",
                )

                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

                st.plotly_chart(fig, use_container_width=True)

        with col6:
            st.markdown("### Destination Airport Type Impact")

            if "type_dest_airport" in filtered_df.columns:
                dest_type_df = get_group_delay_rate(
                    filtered_df,
                    "type_dest_airport",
                    min_flights=1,
                    top_n=10,
                )

                fig = px.bar(
                    dest_type_df,
                    x="type_dest_airport",
                    y="Delay Rate (%)",
                    text="Delay Rate (%)",
                    hover_data=["total_flights", "delayed_flights"],
                    title="Delay Rate by Destination Airport Type",
                )

                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("## Delay Reason Analysis")

        actual_reason_df = get_actual_reason_summary(filtered_df)

        if not actual_reason_df.empty:
            st.success("Actual delay reason columns found in dataset.")

            fig = px.pie(
                actual_reason_df,
                names="Reason",
                values="Value",
                hole=0.45,
                title="Actual Delay Reasons",
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(actual_reason_df, use_container_width=True)

        else:
            st.warning(
                "Actual delay reason columns like WeatherDelay, CarrierDelay, NASDelay are not present. "
                "Showing data-based reason factor analysis instead."
            )

            proxy_reason_df = get_proxy_reason_summary(filtered_df)

            if not proxy_reason_df.empty:
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.pie(
                        proxy_reason_df,
                        names="Reason Factor",
                        values="Risk Score",
                        hole=0.45,
                        title="Data-Based Delay Reason Factors",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.bar(
                        proxy_reason_df,
                        x="Risk Score",
                        y="Reason Factor",
                        orientation="h",
                        text="Risk Score",
                        title="Risk Gap by Reason Factor",
                    )

                    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})

                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Reason Factor Table")
                st.dataframe(proxy_reason_df, use_container_width=True)

            else:
                st.info("Reason factor analysis could not be generated.")

except Exception as e:
    st.error("EDA dashboard could not be loaded.")
    st.exception(e)