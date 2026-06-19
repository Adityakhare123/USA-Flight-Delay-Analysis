import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="USA Flight Delay Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.template = "plotly_white"


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

AIRLINES_FILE = DATA_DIR / "Airlines.csv"
AIRPORTS_FILE = DATA_DIR / "airports.csv"
RUNWAYS_FILE = DATA_DIR / "runways.csv"

MODEL_FILE = MODEL_DIR / "flight_delay_model.pkl"
METRICS_FILE = MODEL_DIR / "metrics.json"


# ============================================================
# PREMIUM UI STYLE WITH VISIBILITY FIXES
# ============================================================

st.markdown(
    """
    <style>
    /* ================================
       GLOBAL APP STYLE
    ================================ */

    html, body, [class*="css"] {
        color: #061522 !important;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f7fbff 0%, #eef5ff 45%, #ffffff 100%);
        color: #061522 !important;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .block-container p,
    .block-container span,
    .block-container label,
    .block-container div,
    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container h4,
    .block-container h5,
    .block-container h6 {
        color: #061522;
    }

    /* ================================
       SIDEBAR
    ================================ */

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061522 0%, #0b263c 52%, #0e3a5c 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] *,
    [data-testid="stSidebar"] div[data-baseweb="input"] *,
    [data-testid="stSidebar"] input {
        color: #061522 !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] div[data-baseweb="input"] > div {
        background: #ffffff !important;
        border-color: #dbe4ef !important;
    }

    /* ================================
       HERO SECTION
    ================================ */

    .hero-card {
        position: relative;
        padding: 40px 44px;
        border-radius: 30px;
        background: linear-gradient(135deg, #061522 0%, #0e4d78 55%, #13a3d8 100%);
        color: white !important;
        box-shadow: 0 24px 70px rgba(14, 76, 120, 0.28);
        overflow: hidden;
        margin-bottom: 28px;
    }

    .hero-card::before {
        content: "";
        position: absolute;
        top: -90px;
        right: -95px;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.15);
    }

    .hero-card::after {
        content: "";
        position: absolute;
        bottom: -130px;
        right: 145px;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
    }

    .hero-card,
    .hero-card *,
    .hero-kicker,
    .hero-title,
    .hero-subtitle,
    .hero-tag {
        color: #ffffff !important;
    }

    .hero-kicker {
        position: relative;
        z-index: 2;
        display: inline-block;
        padding: 8px 15px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.16);
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.6px;
        margin-bottom: 15px;
    }

    .hero-title {
        position: relative;
        z-index: 2;
        font-size: 50px;
        line-height: 1.08;
        font-weight: 950;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        position: relative;
        z-index: 2;
        font-size: 18px;
        line-height: 1.7;
        max-width: 890px;
        color: rgba(255, 255, 255, 0.92) !important;
        margin-bottom: 24px;
    }

    .hero-tags {
        position: relative;
        z-index: 2;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }

    .hero-tag {
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.18);
        font-size: 13px;
        font-weight: 700;
    }

    /* ================================
       SECTION TITLES
    ================================ */

    .section-title {
        font-size: 29px;
        font-weight: 900;
        color: #061522 !important;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #475569 !important;
        font-size: 15.5px;
        margin-bottom: 19px;
    }

    /* ================================
       CUSTOM METRIC CARDS
    ================================ */

    .metric-card {
        background: rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(18px);
        border: 1px solid rgba(203, 213, 225, 0.95);
        border-radius: 23px;
        padding: 23px 23px;
        box-shadow: 0 16px 45px rgba(15, 23, 42, 0.07);
        transition: all 0.22s ease;
        min-height: 132px;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 22px 55px rgba(15, 23, 42, 0.12);
    }

    .metric-card,
    .metric-card * {
        color: #061522 !important;
    }

    .metric-label {
        color: #475569 !important;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #061522 !important;
        font-size: 31px;
        font-weight: 950;
        margin-bottom: 4px;
    }

    .metric-helper {
        color: #64748b !important;
        font-size: 13px;
        font-weight: 650;
    }

    /* ================================
       FEATURE CARDS
    ================================ */

    .feature-card {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(203, 213, 225, 0.96);
        border-radius: 25px;
        padding: 25px;
        min-height: 194px;
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.07);
        transition: all 0.22s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
    }

    .feature-card,
    .feature-card * {
        color: #061522 !important;
    }

    .feature-icon {
        font-size: 35px;
        margin-bottom: 13px;
    }

    .feature-title {
        color: #061522 !important;
        font-size: 19px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .feature-text {
        color: #475569 !important;
        font-size: 14.5px;
        line-height: 1.65;
    }

    /* ================================
       STATUS CARDS
    ================================ */

    .status-card {
        border-radius: 24px;
        padding: 24px 26px;
        margin-top: 14px;
        margin-bottom: 16px;
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(203, 213, 225, 0.92);
    }

    .status-card,
    .status-card * {
        color: #061522 !important;
    }

    .status-delay {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border-left: 7px solid #e11d48;
    }

    .status-ontime {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 7px solid #059669;
    }

    .status-title {
        font-size: 25px;
        font-weight: 950;
        margin-bottom: 6px;
        color: #061522 !important;
    }

    .status-text {
        font-size: 15px;
        color: #475569 !important;
        line-height: 1.6;
    }

    /* ================================
       STREAMLIT COMPONENT VISIBILITY FIXES
    ================================ */

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.96) !important;
        border: 1px solid rgba(203, 213, 225, 0.96) !important;
        padding: 18px !important;
        border-radius: 21px !important;
        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06) !important;
    }

    div[data-testid="stMetric"],
    div[data-testid="stMetric"] *,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricDelta"] {
        color: #061522 !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #475569 !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #061522 !important;
        font-weight: 950 !important;
    }

    div[data-testid="stTabs"] button,
    div[data-testid="stTabs"] button *,
    div[data-testid="stTabs"] button p {
        color: #334155 !important;
        font-weight: 850 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"] *,
    div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: #ef4444 !important;
    }

    div[data-testid="stTabs"] button {
        font-size: 15px !important;
        padding: 13px 18px !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid #cbd5e1;
        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stDataFrame"] *,
    div[data-testid="stTable"] * {
        color: #061522 !important;
    }

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary *,
    div[data-testid="stExpander"] details,
    div[data-testid="stExpander"] details * {
        color: #061522 !important;
    }

    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    div[data-baseweb="input"],
    div[data-baseweb="input"] *,
    input,
    textarea {
        color: #061522 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
    }

    .stButton > button {
        border-radius: 15px;
        padding: 0.72rem 1.25rem;
        font-weight: 850;
        border: none;
        background: linear-gradient(135deg, #0e76a8 0%, #13a3d8 100%);
        color: white !important;
        box-shadow: 0 14px 34px rgba(14, 118, 168, 0.25);
        transition: all 0.18s ease;
    }

    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: white !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 40px rgba(14, 118, 168, 0.34);
        color: white !important;
    }

    .stAlert,
    .stAlert *,
    div[data-testid="stAlert"],
    div[data-testid="stAlert"] * {
        color: #061522 !important;
    }

    hr {
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UI HELPER FUNCTIONS
# ============================================================

def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">MACHINE LEARNING • DATA ANALYTICS • AVIATION</div>
            <div class="hero-title">✈️ USA Flight Delay Analysis</div>
            <div class="hero-subtitle">
                A modern interactive dashboard to analyze U.S. airline delay patterns,
                understand possible delay reasons, and predict whether a selected flight
                may be delayed using machine learning.
            </div>
            <div class="hero-tags">
                <div class="hero-tag">📊 EDA Dashboard</div>
                <div class="hero-tag">🥧 Delay Reason Charts</div>
                <div class="hero-tag">🤖 ML Prediction</div>
                <div class="hero-tag">🛫 Airline & Airport Insights</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_metric_card(label, value, helper=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(icon, title, text):
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_status(is_delayed, probability):
    if is_delayed:
        st.markdown(
            f"""
            <div class="status-card status-delay">
                <div class="status-title">✈️ Flight is likely to be DELAYED</div>
                <div class="status-text">
                    The selected flight combination shows a higher delay possibility based on the trained model.
                    Prediction confidence: <b>{probability}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-card status-ontime">
                <div class="status-title">✅ Flight is likely to be ON TIME</div>
                <div class="status-text">
                    The selected flight combination is predicted to be on time based on the trained model.
                    Prediction confidence: <b>{probability}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def format_figure(fig, height=440):
    """
    Fixes all graph text visibility:
    - graph title
    - axis labels
    - tick labels
    - legend text
    - pie labels
    - bar text
    - hover labels
    """

    fig.update_layout(
        height=height,
        margin=dict(l=45, r=45, t=78, b=55),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        title=dict(
            font=dict(size=20, family="Arial", color="#061522"),
            x=0.02,
            xanchor="left",
        ),
        font=dict(family="Arial", size=13, color="#061522"),
        legend=dict(
            font=dict(size=13, color="#061522"),
            title=dict(font=dict(size=13, color="#061522")),
            bgcolor="rgba(255,255,255,0)",
            bordercolor="rgba(255,255,255,0)",
        ),
        hoverlabel=dict(
            bgcolor="#061522",
            font_size=13,
            font_family="Arial",
            font_color="#ffffff",
        ),
    )

    fig.update_xaxes(
        title_font=dict(size=14, color="#061522"),
        tickfont=dict(size=12, color="#061522"),
        showline=True,
        linewidth=1,
        linecolor="#cbd5e1",
        gridcolor="rgba(148, 163, 184, 0.25)",
        zerolinecolor="rgba(148, 163, 184, 0.25)",
    )

    fig.update_yaxes(
        title_font=dict(size=14, color="#061522"),
        tickfont=dict(size=12, color="#061522"),
        showline=True,
        linewidth=1,
        linecolor="#cbd5e1",
        gridcolor="rgba(148, 163, 184, 0.25)",
        zerolinecolor="rgba(148, 163, 184, 0.25)",
    )

    fig.update_traces(
        textfont=dict(size=12, color="#061522"),
        marker_line_color="rgba(255,255,255,0.75)",
        marker_line_width=1,
    )

    for trace in fig.data:
        if trace.type == "pie":
            trace.textfont = dict(size=13, color="#061522")
            trace.insidetextfont = dict(size=13, color="#ffffff")
            trace.outsidetextfont = dict(size=13, color="#061522")
            trace.textinfo = "label+percent"
            trace.hoverlabel = dict(bgcolor="#061522", font_color="#ffffff")

        if trace.type == "bar":
            trace.textfont = dict(size=12, color="#061522")
            trace.cliponaxis = False

    return fig


# ============================================================
# DATA FUNCTIONS
# ============================================================

def clean_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_data():
    if not AIRLINES_FILE.exists():
        st.error("Airlines.csv not found. Put it inside the data folder.")
        st.stop()

    airlines = pd.read_csv(AIRLINES_FILE)
    airlines = clean_columns(airlines)

    required_cols = [
        "Airline",
        "Flight",
        "AirportFrom",
        "AirportTo",
        "DayOfWeek",
        "Time",
        "Length",
        "Delay",
    ]

    missing_cols = [col for col in required_cols if col not in airlines.columns]

    if missing_cols:
        st.error(f"Missing columns in Airlines.csv: {missing_cols}")
        st.stop()

    df = airlines.copy()

    if AIRPORTS_FILE.exists() and RUNWAYS_FILE.exists():
        try:
            airports = pd.read_csv(AIRPORTS_FILE)
            runways = pd.read_csv(RUNWAYS_FILE)

            airports = clean_columns(airports)
            runways = clean_columns(runways)

            if {"ident", "iata_code", "type", "elevation_ft"}.issubset(airports.columns) and "airport_ident" in runways.columns:
                runway_count = (
                    runways.groupby("airport_ident")
                    .size()
                    .reset_index(name="runway_count")
                    .rename(columns={"airport_ident": "ident"})
                )

                airport_features = airports.merge(runway_count, on="ident", how="left")

                airport_features = airport_features[
                    ["iata_code", "type", "elevation_ft", "runway_count"]
                ]

                airport_features["runway_count"] = airport_features["runway_count"].fillna(0)

                df = df.merge(
                    airport_features,
                    left_on="AirportFrom",
                    right_on="iata_code",
                    how="left",
                )

                df = df.rename(
                    columns={
                        "type": "type_source_airport",
                        "elevation_ft": "elevation_ft_source_airport",
                        "runway_count": "runway_count_source_airport",
                    }
                )

                df = df.drop(columns=["iata_code"], errors="ignore")

                df = df.merge(
                    airport_features,
                    left_on="AirportTo",
                    right_on="iata_code",
                    how="left",
                )

                df = df.rename(
                    columns={
                        "type": "type_dest_airport",
                        "elevation_ft": "elevation_ft_dest_airport",
                        "runway_count": "runway_count_dest_airport",
                    }
                )

                df = df.drop(columns=["iata_code"], errors="ignore")

        except Exception as e:
            st.warning(f"Airport/runway merge skipped because of error: {e}")

    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    categorical_cols = df.select_dtypes(exclude=["int64", "float64", "int32", "float32"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    df["Delay_Label"] = df["Delay"].map({0: "Not Delayed", 1: "Delayed"})

    if df["Time"].max() > 24:
        df["Hour"] = (df["Time"] // 60).astype(int)
    else:
        df["Hour"] = df["Time"].astype(int)

    def get_time_period(hour):
        if 0 <= hour <= 5:
            return "Late Night"
        if 6 <= hour <= 11:
            return "Morning"
        if 12 <= hour <= 17:
            return "Afternoon"
        if 18 <= hour <= 23:
            return "Evening / Night"
        return "Unknown"

    df["Time_Period"] = df["Hour"].apply(get_time_period)

    df["Duration_Category"] = pd.cut(
        df["Length"],
        bins=3,
        labels=["Short", "Medium", "Long"],
        include_lowest=True,
    ).astype(str)

    df["Route"] = df["AirportFrom"].astype(str) + " → " + df["AirportTo"].astype(str)

    return df


def get_delay_rate_table(df, group_col, min_flights=20, top_n=15):
    if group_col not in df.columns or df.empty:
        return pd.DataFrame()

    result = (
        df.groupby(group_col)
        .agg(
            total_flights=("Delay", "count"),
            delayed_flights=("Delay", "sum"),
            delay_rate=("Delay", "mean"),
        )
        .reset_index()
    )

    result = result[result["total_flights"] >= min_flights]

    if result.empty:
        return result

    result["Delay Rate (%)"] = result["delay_rate"] * 100
    result = result.sort_values("Delay Rate (%)", ascending=False).head(top_n)

    return result


def get_actual_reason_columns(df):
    reason_map = {
        "CarrierDelay": "Carrier / Airline Delay",
        "WeatherDelay": "Weather Delay",
        "NASDelay": "Air Traffic / NAS Delay",
        "SecurityDelay": "Security Delay",
        "LateAircraftDelay": "Late Aircraft Delay",
        "carrier_delay": "Carrier / Airline Delay",
        "weather_delay": "Weather Delay",
        "nas_delay": "Air Traffic / NAS Delay",
        "security_delay": "Security Delay",
        "late_aircraft_delay": "Late Aircraft Delay",
    }

    found = {}

    normalized_columns = {
        str(col).lower().replace("_", "").replace(" ", ""): col
        for col in df.columns
    }

    for possible_col, label in reason_map.items():
        key = possible_col.lower().replace("_", "").replace(" ", "")

        if key in normalized_columns:
            found[normalized_columns[key]] = label

    return found


def get_reason_summary(df):
    actual_reason_cols = get_actual_reason_columns(df)

    if actual_reason_cols:
        rows = []

        for col, label in actual_reason_cols.items():
            value = pd.to_numeric(df[col], errors="coerce").fillna(0).sum()

            rows.append(
                {
                    "Reason": label,
                    "Value": value,
                }
            )

        result = pd.DataFrame(rows)
        result = result[result["Value"] > 0]
        result = result.sort_values("Value", ascending=False)

        return result, True

    overall_rate = df["Delay"].mean() * 100 if not df.empty else 0

    reason_rows = []

    factor_cols = [
        ("Airline performance pattern", "Airline"),
        ("Source airport congestion pattern", "AirportFrom"),
        ("Destination airport congestion pattern", "AirportTo"),
        ("Route pattern", "Route"),
        ("Day of week pattern", "DayOfWeek"),
        ("Time period pattern", "Time_Period"),
        ("Flight duration pattern", "Duration_Category"),
    ]

    for label, col in factor_cols:
        if col not in df.columns:
            continue

        temp = get_delay_rate_table(df, col, min_flights=20, top_n=1)

        if temp.empty:
            continue

        top_delay_rate = float(temp["Delay Rate (%)"].iloc[0])
        risk_score = max(top_delay_rate - overall_rate, 0)

        reason_rows.append(
            {
                "Reason": label,
                "Value": round(risk_score, 2),
                "Highest Delay Rate (%)": round(top_delay_rate, 2),
            }
        )

    result = pd.DataFrame(reason_rows)

    if not result.empty:
        result = result.sort_values("Value", ascending=False)

    return result, False


# ============================================================
# MODEL FUNCTIONS
# ============================================================

FEATURE_COLUMNS = [
    "Airline",
    "AirportFrom",
    "AirportTo",
    "DayOfWeek",
    "Time",
    "Length",
    "type_source_airport",
    "elevation_ft_source_airport",
    "runway_count_source_airport",
    "type_dest_airport",
    "elevation_ft_dest_airport",
    "runway_count_dest_airport",
]


def get_model_features(df):
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def train_model(df):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    features = get_model_features(df)

    if not features:
        st.error("No valid model features found.")
        st.stop()

    X = df[features]
    y = df["Delay"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = [col for col in X_train.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    model_package = {
        "pipeline": pipeline,
        "features": features,
        "accuracy": accuracy,
    }

    joblib.dump(model_package, MODEL_FILE)

    metrics = {
        "accuracy": accuracy,
        "features": features,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def load_model():
    if not MODEL_FILE.exists():
        return None

    return joblib.load(MODEL_FILE)


def get_airport_values(df, airport_code, airport_col, prefix):
    temp = df[df[airport_col].astype(str) == str(airport_code)]

    output = {}

    type_col = f"type_{prefix}_airport"
    elevation_col = f"elevation_ft_{prefix}_airport"
    runway_col = f"runway_count_{prefix}_airport"

    if type_col in df.columns:
        output[type_col] = temp[type_col].mode()[0] if not temp.empty else "Unknown"

    if elevation_col in df.columns:
        output[elevation_col] = (
            float(temp[elevation_col].median())
            if not temp.empty
            else float(df[elevation_col].median())
        )

    if runway_col in df.columns:
        output[runway_col] = (
            float(temp[runway_col].median())
            if not temp.empty
            else float(df[runway_col].median())
        )

    return output


def get_prediction_reasons(df, input_data):
    overall_rate = df["Delay"].mean() * 100

    rows = []

    def add_reason(label, mask):
        temp = df[mask]

        if temp.empty:
            selected_rate = 0
            sample_size = 0
        else:
            selected_rate = temp["Delay"].mean() * 100
            sample_size = len(temp)

        risk_gap = selected_rate - overall_rate

        rows.append(
            {
                "Reason": label,
                "Dataset Average (%)": round(overall_rate, 2),
                "Selected Input Delay Rate (%)": round(selected_rate, 2),
                "Risk Gap (%)": round(risk_gap, 2),
                "Risk Score": round(max(risk_gap, 0), 2),
                "Sample Size": sample_size,
            }
        )

    add_reason(
        f"Airline: {input_data['Airline']}",
        df["Airline"].astype(str) == str(input_data["Airline"]),
    )

    add_reason(
        f"Source Airport: {input_data['AirportFrom']}",
        df["AirportFrom"].astype(str) == str(input_data["AirportFrom"]),
    )

    add_reason(
        f"Destination Airport: {input_data['AirportTo']}",
        df["AirportTo"].astype(str) == str(input_data["AirportTo"]),
    )

    route = str(input_data["AirportFrom"]) + " → " + str(input_data["AirportTo"])

    add_reason(
        f"Route: {route}",
        df["Route"].astype(str) == route,
    )

    add_reason(
        f"Day of Week: {input_data['DayOfWeek']}",
        df["DayOfWeek"].astype(str) == str(input_data["DayOfWeek"]),
    )

    if df["Time"].max() > 24:
        input_hour = int(input_data["Time"] // 60)
    else:
        input_hour = int(input_data["Time"])

    if 0 <= input_hour <= 5:
        input_period = "Late Night"
    elif 6 <= input_hour <= 11:
        input_period = "Morning"
    elif 12 <= input_hour <= 17:
        input_period = "Afternoon"
    elif 18 <= input_hour <= 23:
        input_period = "Evening / Night"
    else:
        input_period = "Unknown"

    add_reason(
        f"Time Period: {input_period}",
        df["Time_Period"].astype(str) == input_period,
    )

    try:
        _, bins = pd.cut(
            df["Length"],
            bins=3,
            labels=["Short", "Medium", "Long"],
            retbins=True,
            include_lowest=True,
        )

        input_duration = pd.cut(
            [input_data["Length"]],
            bins=bins,
            labels=["Short", "Medium", "Long"],
            include_lowest=True,
        )[0]

        input_duration = str(input_duration)

    except Exception:
        input_duration = "Unknown"

    add_reason(
        f"Flight Duration: {input_duration}",
        df["Duration_Category"].astype(str) == input_duration,
    )

    reason_df = pd.DataFrame(rows)

    if reason_df["Risk Score"].sum() == 0:
        reason_df["Risk Score"] = reason_df["Selected Input Delay Rate (%)"]

    reason_df = reason_df.sort_values("Risk Score", ascending=False)

    return reason_df


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner("Loading airline dataset..."):
    df = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ✈️ USA Flight Delay")
st.sidebar.caption("Interactive ML dashboard")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Project Status")

if MODEL_FILE.exists():
    st.sidebar.success("Model file found")
else:
    st.sidebar.warning("Model not trained yet")

st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.write(f"Flights: **{len(df):,}**")
st.sidebar.write(f"Airlines: **{df['Airline'].nunique():,}**")
st.sidebar.write(f"Airports: **{pd.concat([df['AirportFrom'], df['AirportTo']]).nunique():,}**")


# ============================================================
# HEADER
# ============================================================

render_hero()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏠 Overview",
        "📊 Analytics Dashboard",
        "🤖 Delay Predictor",
        "📂 Dataset Explorer",
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with tab1:
    render_section(
        "Project Overview",
        "A modern analytics system for airline delay analysis, visual insights, and prediction.",
    )

    st.write(
        """
        This project analyzes U.S. airline flight delay data. Users can view delay patterns,
        understand possible delay reasons, and predict whether a selected flight may be delayed
        using machine learning.
        """
    )

    total_flights = len(df)
    delayed_flights = int(df["Delay"].sum())
    not_delayed = total_flights - delayed_flights
    delay_rate = delayed_flights / total_flights * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card("Total Flights", f"{total_flights:,}", "Complete records analyzed")

    with col2:
        render_metric_card("Delayed Flights", f"{delayed_flights:,}", "Flights marked as delayed")

    with col3:
        render_metric_card("On-Time Flights", f"{not_delayed:,}", "Flights not delayed")

    with col4:
        render_metric_card("Delay Rate", f"{delay_rate:.2f}%", "Overall delay percentage")

    st.markdown("<br>", unsafe_allow_html=True)

    render_section(
        "What Users Can Explore",
        "The app provides visual analysis and prediction in one clean interface.",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        render_feature_card(
            "📊",
            "Interactive Dashboard",
            "Explore airline, airport, route, weekday, time period, and duration-based delay trends.",
        )

    with c2:
        render_feature_card(
            "🥧",
            "Delay Reason Factors",
            "View pie charts and risk-factor charts showing possible delay reasons based on dataset patterns.",
        )

    with c3:
        render_feature_card(
            "🤖",
            "ML Prediction",
            "Select flight details and predict whether the flight is likely to be delayed or on time.",
        )


# ============================================================
# ANALYTICS DASHBOARD TAB
# ============================================================

with tab2:
    render_section(
        "Analytics Dashboard",
        "Explore graph-based insights about delay distribution, timing, airlines, airports, routes, and risk factors.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Dashboard Filters")

    filtered_df = df.copy()

    airline_options = sorted(df["Airline"].dropna().astype(str).unique().tolist())

    selected_airlines = st.sidebar.multiselect(
        "Select Airlines",
        airline_options,
        default=airline_options[:8] if len(airline_options) > 8 else airline_options,
    )

    if selected_airlines:
        filtered_df = filtered_df[filtered_df["Airline"].astype(str).isin(selected_airlines)]

    source_options = sorted(df["AirportFrom"].dropna().astype(str).unique().tolist())
    selected_source = st.sidebar.selectbox("Source Airport", ["All"] + source_options)

    if selected_source != "All":
        filtered_df = filtered_df[filtered_df["AirportFrom"].astype(str) == selected_source]

    destination_options = sorted(df["AirportTo"].dropna().astype(str).unique().tolist())
    selected_destination = st.sidebar.selectbox("Destination Airport", ["All"] + destination_options)

    if selected_destination != "All":
        filtered_df = filtered_df[filtered_df["AirportTo"].astype(str) == selected_destination]

    total_filtered = len(filtered_df)
    delayed_filtered = int(filtered_df["Delay"].sum())
    ontime_filtered = total_filtered - delayed_filtered
    filtered_delay_rate = delayed_filtered / total_filtered * 100 if total_filtered else 0

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        render_metric_card("Filtered Flights", f"{total_filtered:,}", "After selected filters")

    with m2:
        render_metric_card("Delayed", f"{delayed_filtered:,}", "Delayed flights")

    with m3:
        render_metric_card("On Time", f"{ontime_filtered:,}", "Non-delayed flights")

    with m4:
        render_metric_card("Delay Rate", f"{filtered_delay_rate:.2f}%", "Filtered delay percentage")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        delay_count = filtered_df["Delay_Label"].value_counts().reset_index()
        delay_count.columns = ["Delay Status", "Count"]

        fig = px.pie(
            delay_count,
            names="Delay Status",
            values="Count",
            hole=0.48,
            title="Delayed vs Not Delayed Flights",
        )

        st.plotly_chart(format_figure(fig), use_container_width=True)

    with col2:
        time_count = (
            filtered_df[filtered_df["Delay"] == 1]["Time_Period"]
            .value_counts()
            .reset_index()
        )

        time_count.columns = ["Time Period", "Delayed Flights"]

        fig = px.pie(
            time_count,
            names="Time Period",
            values="Delayed Flights",
            hole=0.48,
            title="Delayed Flights by Time Period",
        )

        st.plotly_chart(format_figure(fig), use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        airline_delay = get_delay_rate_table(filtered_df, "Airline", min_flights=50, top_n=15)

        if not airline_delay.empty:
            fig = px.bar(
                airline_delay,
                x="Airline",
                y="Delay Rate (%)",
                text="Delay Rate (%)",
                hover_data=["total_flights", "delayed_flights"],
                title="Top Airlines by Delay Rate",
            )

            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(format_figure(fig), use_container_width=True)
        else:
            st.info("Not enough airline data for this filter.")

    with col4:
        day_delay = get_delay_rate_table(filtered_df, "DayOfWeek", min_flights=1, top_n=10)

        if not day_delay.empty:
            day_delay = day_delay.sort_values("DayOfWeek")

            fig = px.bar(
                day_delay,
                x="DayOfWeek",
                y="Delay Rate (%)",
                text="Delay Rate (%)",
                hover_data=["total_flights", "delayed_flights"],
                title="Delay Rate by Day of Week",
            )

            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(format_figure(fig), use_container_width=True)
        else:
            st.info("Not enough weekday data for this filter.")

    col5, col6 = st.columns(2)

    with col5:
        source_delay = get_delay_rate_table(filtered_df, "AirportFrom", min_flights=50, top_n=15)

        if not source_delay.empty:
            fig = px.bar(
                source_delay,
                x="AirportFrom",
                y="Delay Rate (%)",
                text="Delay Rate (%)",
                hover_data=["total_flights", "delayed_flights"],
                title="Top Source Airports by Delay Rate",
            )

            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(format_figure(fig), use_container_width=True)
        else:
            st.info("Not enough source airport data for this filter.")

    with col6:
        dest_delay = get_delay_rate_table(filtered_df, "AirportTo", min_flights=50, top_n=15)

        if not dest_delay.empty:
            fig = px.bar(
                dest_delay,
                x="AirportTo",
                y="Delay Rate (%)",
                text="Delay Rate (%)",
                hover_data=["total_flights", "delayed_flights"],
                title="Top Destination Airports by Delay Rate",
            )

            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(format_figure(fig), use_container_width=True)
        else:
            st.info("Not enough destination airport data for this filter.")

    col7, col8 = st.columns(2)

    with col7:
        route_delay = get_delay_rate_table(filtered_df, "Route", min_flights=30, top_n=15)

        if not route_delay.empty:
            fig = px.bar(
                route_delay,
                x="Route",
                y="Delay Rate (%)",
                text="Delay Rate (%)",
                hover_data=["total_flights", "delayed_flights"],
                title="Top Routes by Delay Rate",
            )

            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(format_figure(fig, height=480), use_container_width=True)
        else:
            st.info("Not enough route data for this filter.")

    with col8:
        duration_delay = get_delay_rate_table(
            filtered_df,
            "Duration_Category",
            min_flights=1,
            top_n=10,
        )

        if not duration_delay.empty:
            fig = px.bar(
                duration_delay,
                x="Duration_Category",
                y="Delay Rate (%)",
                text="Delay Rate (%)",
                hover_data=["total_flights", "delayed_flights"],
                title="Delay Rate by Flight Duration",
            )

            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(format_figure(fig), use_container_width=True)
        else:
            st.info("Not enough duration data for this filter.")

    render_section(
        "Delay Reason Factors",
        "If actual reason columns are missing, this section uses data-based risk patterns.",
    )

    reason_df, actual_reason_found = get_reason_summary(filtered_df)

    if actual_reason_found:
        st.success("Actual delay reason columns found in dataset.")
    else:
        st.warning(
            "Actual reason columns like WeatherDelay or CarrierDelay were not found. "
            "So this chart shows reason factors based on available dataset patterns."
        )

    if not reason_df.empty:
        col9, col10 = st.columns(2)

        with col9:
            fig = px.pie(
                reason_df,
                names="Reason",
                values="Value",
                hole=0.48,
                title="Delay Reason / Risk Factor Share",
            )

            st.plotly_chart(format_figure(fig), use_container_width=True)

        with col10:
            fig = px.bar(
                reason_df,
                x="Value",
                y="Reason",
                orientation="h",
                text="Value",
                title="Delay Reason / Risk Factor Score",
            )

            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(format_figure(fig), use_container_width=True)

        st.dataframe(reason_df, use_container_width=True)
    else:
        st.info("Reason factor analysis is not available for this filter.")


# ============================================================
# DELAY PREDICTOR TAB
# ============================================================

with tab3:
    render_section(
        "Flight Delay Predictor",
        "Load the trained model, select flight details, and get delay prediction with reason breakdown.",
    )

    col_train1, col_train2, col_train3 = st.columns([1.2, 1.2, 2])

    with col_train1:
        train_clicked = st.button("Train Model Again", type="primary")

    if train_clicked:
        with st.spinner("Training model. Please wait..."):
            metrics = train_model(df)

        st.success("Model trained successfully.")

    model_package = load_model()

    with col_train2:
        if model_package is None:
            st.warning("Model not trained yet.")
        else:
            st.success("Model loaded successfully")

    with col_train3:
        if model_package is not None:
            render_metric_card(
                "Saved Model Accuracy",
                f"{model_package['accuracy'] * 100:.2f}%",
                "Current trained model performance",
            )
        else:
            render_metric_card(
                "Saved Model Accuracy",
                "Not available",
                "Train the model to generate accuracy",
            )

    st.markdown("---")

    render_section("Enter Flight Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox(
            "Airline",
            sorted(df["Airline"].dropna().astype(str).unique().tolist()),
        )

        airport_from = st.selectbox(
            "Source Airport",
            sorted(df["AirportFrom"].dropna().astype(str).unique().tolist()),
        )

    with col2:
        airport_to = st.selectbox(
            "Destination Airport",
            sorted(df["AirportTo"].dropna().astype(str).unique().tolist()),
        )

        day_of_week = st.selectbox(
            "Day of Week",
            sorted(df["DayOfWeek"].dropna().unique().tolist()),
        )

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

    source_values = get_airport_values(df, airport_from, "AirportFrom", "source")
    dest_values = get_airport_values(df, airport_to, "AirportTo", "dest")

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

    with st.expander("Selected Input Preview", expanded=False):
        st.dataframe(input_df, use_container_width=True)

    predict_clicked = st.button("Predict Delay", type="primary")

    if predict_clicked:
        model_package = load_model()

        if model_package is None:
            with st.spinner("Model not found. Training model automatically..."):
                train_model(df)

            model_package = load_model()

            if model_package is None:
                st.error("Model training failed. Please check terminal errors.")
                st.stop()

            st.success("Model trained successfully. Running prediction now...")

        pipeline = model_package["pipeline"]
        features = model_package["features"]

        prediction = pipeline.predict(input_df[features])[0]

        probability = None

        if hasattr(pipeline, "predict_proba"):
            probability = max(pipeline.predict_proba(input_df[features])[0])

        probability_text = f"{probability * 100:.2f}%" if probability is not None else "Not available"

        render_prediction_status(int(prediction) == 1, probability_text)

        render_section(
            "Why this prediction?",
            "The reason breakdown compares your selected inputs with historical delay rates in the dataset.",
        )

        reason_prediction_df = get_prediction_reasons(df, input_data)

        top_reason = reason_prediction_df.iloc[0]

        st.info(
            f"Top factor: **{top_reason['Reason']}**. "
            f"Selected input delay rate is **{top_reason['Selected Input Delay Rate (%)']}%**."
        )

        col_reason1, col_reason2 = st.columns(2)

        with col_reason1:
            pie_df = reason_prediction_df[reason_prediction_df["Risk Score"] > 0]

            if pie_df.empty:
                pie_df = reason_prediction_df

            fig = px.pie(
                pie_df,
                names="Reason",
                values="Risk Score",
                hole=0.48,
                title="Prediction Reason Breakdown",
            )

            st.plotly_chart(format_figure(fig), use_container_width=True)

        with col_reason2:
            fig = px.bar(
                reason_prediction_df,
                x="Selected Input Delay Rate (%)",
                y="Reason",
                orientation="h",
                text="Selected Input Delay Rate (%)",
                title="Selected Input Delay Rate by Factor",
            )

            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(format_figure(fig), use_container_width=True)

        st.markdown("### Reason Table")
        st.dataframe(reason_prediction_df, use_container_width=True)


# ============================================================
# DATASET EXPLORER TAB
# ============================================================

with tab4:
    render_section(
        "Dataset Explorer",
        "Preview the dataset, check columns, missing values, and unique value counts.",
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card("Rows", f"{df.shape[0]:,}", "Total dataset records")

    with col2:
        render_metric_card("Columns", df.shape[1], "Available features")

    with col3:
        render_metric_card("Missing Values", f"{df.isna().sum().sum():,}", "After preprocessing")

    with col4:
        render_metric_card("Duplicate Rows", f"{df.duplicated().sum():,}", "Duplicate records found")

    st.markdown("### Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("### Column Information")

    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(df[col].dtype) for col in df.columns],
            "Missing Values": [df[col].isna().sum() for col in df.columns],
            "Unique Values": [df[col].nunique() for col in df.columns],
        }
    )

    st.dataframe(column_info, use_container_width=True)