# About project page
import streamlit as st

from src.ui_utils import page_header

st.set_page_config(
    page_title="About Project | USA Flight Delay Analysis",
    page_icon="ℹ️",
    layout="wide",
)

page_header(
    "ℹ️ About Project",
    "Project details, objective, modules, technology stack, and future scope.",
)

st.markdown("## Project Title")

st.write("**USA Flight Delay Analysis Using Machine Learning and Streamlit**")

st.markdown("## Objective")

st.write(
    """
    The objective of this project is to analyze U.S. airline flight delay data,
    identify important delay patterns, and build a machine learning model that
    predicts whether a flight may be delayed.
    """
)

st.markdown("## Main Modules")

st.markdown(
    """
    1. **Data Collection and Preprocessing**  
       Load airline, airport, and runway datasets. Clean missing values and prepare data.

    2. **Exploratory Data Analysis**  
       Analyze delay patterns by airline, airport, day of week, time, and flight duration.

    3. **Machine Learning Prediction**  
       Train a Random Forest model to predict delayed and non-delayed flights.

    4. **Visualization Dashboard**  
       Display interactive charts using Plotly and Streamlit.

    5. **User Interface**  
       Provide a simple web UI where users can view results and make predictions.
    """
)

st.markdown("## Technology Stack")

st.markdown(
    """
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Plotly
    - Streamlit
    - Joblib
    """
)

st.markdown("## Future Scope")

st.markdown(
    """
    - Add real-time flight tracking API.
    - Add weather API integration.
    - Add live airport congestion data.
    - Improve model accuracy using XGBoost or deep learning.
    - Deploy the app on Streamlit Community Cloud.
    - Add user login and saved prediction history.
    """
)

st.markdown("## Team")

st.markdown(
    """
    - Atharv Bhoot  
    - Aaditya Pandey  
    - Aditya Khare  
    - Deep Jarwal  
    """
)

st.markdown("## Guide")

st.write("**Prof. Jayshree Pargee**")