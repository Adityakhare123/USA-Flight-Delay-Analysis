# USA Flight Delay Analysis

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:020617,50:0E76A8,100:38BDF8&height=220&section=header&text=USA%20Flight%20Delay%20Analysis&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Python%20%7C%20Streamlit%20%7C%20Plotly%20%7C%20Machine%20Learning&descAlignY=60&descAlign=50&descSize=18" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&duration=2500&pause=800&color=0E76A8&center=true&vCenter=true&width=900&lines=Analyze+USA+Flight+Delay+Patterns;Predict+Flight+Delays+Using+Machine+Learning;Interactive+Dashboard+with+Streamlit;Visualize+Airline%2C+Airport%2C+Route+and+Time+Insights" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,git,github,vscode" />
</p>

<p align="center">
  A modern Streamlit web application for analyzing USA airline flight delay patterns and predicting flight delays using machine learning.
</p>

---

## Live Project

<p align="center">
  <img src="https://img.shields.io/badge/Live%20Demo-Coming%20Soon-0E76A8?style=for-the-badge&logo=streamlit&logoColor=white" />
  <a href="https://github.com/Adityakhare123/USA-Flight-Delay-Analysis">
    <img src="https://img.shields.io/badge/GitHub-Open%20Repository-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</p>

Live:

```text
https://usa-flight-delay-analysis-kpv4ahwtxccdbq2jn2nnez.streamlit.app/
```

---

## Project Overview

USA Flight Delay Analysis is a data analytics and machine learning project designed to analyze flight delay patterns across airlines, airports, routes, days, time periods, and flight durations.

The project provides a clean Streamlit-based user interface where users can explore visual insights, filter flight data, train or load a machine learning model, and predict whether a selected flight is likely to be delayed.

---

## Project Objectives

This project focuses on:

* Analyzing USA airline flight delay data
* Identifying delay trends by airline, airport, route, weekday, time, and duration
* Building a machine learning model for flight delay prediction
* Creating an interactive dashboard for users
* Showing historical risk factors behind each prediction
* Preparing the application for Streamlit Cloud deployment

---

## Key Features

* Interactive Streamlit dashboard
* Flight delay distribution analysis
* Airline-wise delay rate analysis
* Source airport delay analysis
* Destination airport delay analysis
* Route-wise delay analysis
* Day-of-week delay analysis
* Time-period delay analysis
* Flight duration delay analysis
* Data-based delay reason factor analysis
* Machine learning delay prediction
* Prediction confidence score
* Prediction reason breakdown
* Dataset preview and column summary
* Modern and clean UI design

---

## Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,git,github,vscode" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-Core%20Language-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20Interface-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Joblib-Model%20Storage-111827?style=for-the-badge&logo=python&logoColor=white" />
</p>

---

## Application Modules

| Module              | Description                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| Overview            | Displays project summary and key dataset metrics                       |
| Analytics Dashboard | Shows interactive visual analysis of flight delays                     |
| Delay Predictor     | Predicts whether a selected flight may be delayed                      |
| Dataset Explorer    | Displays dataset preview, column information, and data quality summary |

---

## Dashboard Insights

The analytics dashboard includes:

* Delayed vs not delayed flight distribution
* Delayed flights by time period
* Top airlines by delay rate
* Delay rate by day of week
* Top source airports by delay rate
* Top destination airports by delay rate
* Top routes by delay rate
* Delay rate by flight duration
* Delay reason or risk factor analysis

---

## Machine Learning Model

The project uses a Random Forest Classifier to predict whether a flight is delayed or not.

### Model Input Features

The model uses available features such as:

```text
Airline
AirportFrom
AirportTo
DayOfWeek
Time
Length
type_source_airport
elevation_ft_source_airport
runway_count_source_airport
type_dest_airport
elevation_ft_dest_airport
runway_count_dest_airport
```

### Prediction Output

```text
0 = Not Delayed
1 = Delayed
```

### Current Model Accuracy

```text
64.71%
```

### Saved Model Files

```text
models/flight_delay_model.pkl
models/metrics.json
```

---

## Delay Reason Analysis

If actual delay reason columns are available in the dataset, the app can analyze those columns directly.

Supported actual delay reason columns include:

```text
CarrierDelay
WeatherDelay
NASDelay
SecurityDelay
LateAircraftDelay
```

If these columns are not available, the app generates data-based risk factors using available features such as:

* Airline performance pattern
* Source airport delay pattern
* Destination airport delay pattern
* Route delay pattern
* Day of week pattern
* Time period pattern
* Flight duration pattern

---

## Project Structure

```text
USA-Flight-Delay-Analysis/
│
├── assets/
│
├── data/
│   ├── Airlines.csv
│   ├── airports.csv
│   ├── runways.csv
│   └── usa.csv
│
├── models/
│   ├── flight_delay_model.pkl
│   └── metrics.json
│
├── pages/
│
├── src/
│
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
└── train_model.py
```

---

## Dataset Information

The main dataset contains flight-level information.

| Column      | Description           |
| ----------- | --------------------- |
| Airline     | Airline code          |
| Flight      | Flight number         |
| AirportFrom | Source airport        |
| AirportTo   | Destination airport   |
| DayOfWeek   | Day of week           |
| Time        | Scheduled flight time |
| Length      | Flight duration       |
| Delay       | Target variable       |

Additional airport and runway datasets are used to enrich flight records with airport type, elevation, and runway count.

---

## How the App Works

```text
Load Flight Dataset
        |
        v
Clean and Prepare Data
        |
        v
Generate Dashboard Insights
        |
        v
Train or Load ML Model
        |
        v
User Selects Flight Details
        |
        v
Predict Delay Status
        |
        v
Show Confidence and Reason Breakdown
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Adityakhare123/USA-Flight-Delay-Analysis.git
cd USA-Flight-Delay-Analysis
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run the training script:

```bash
python train_model.py
```

After successful training, the following files will be generated:

```text
models/flight_delay_model.pkl
models/metrics.json
```

---

## Run Locally

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Requirements

```txt
streamlit
pandas
plotly
scikit-learn
joblib
openpyxl
```

---

## Deployment

This project can be deployed using Streamlit Community Cloud.

Deployment configuration:

```text
Repository: Adityakhare123/USA-Flight-Delay-Analysis
Branch: main
Main file path: app.py
Python version: 3.x
```

Steps:

```text
1. Push the project to GitHub
2. Open Streamlit Community Cloud
3. Click New App
4. Select the GitHub repository
5. Set main file path as app.py
6. Click Deploy
```

---

## Large File Note

The file `data/usa.csv` is larger than GitHub's recommended file size limit.

If deployment becomes slow or fails, remove the large CSV from Git tracking:

```bash
git rm --cached data/usa.csv
echo data/usa.csv >> .gitignore
git add .gitignore
git commit -m "Remove large CSV from repository tracking"
git push
```

---

## Current Capability

The current application can:

* Load and preprocess flight data
* Analyze delay distribution
* Analyze delays by airline
* Analyze delays by airport
* Analyze delays by route
* Analyze delays by time period
* Analyze delays by flight duration
* Train a Random Forest model
* Save and load the trained model
* Predict flight delay status
* Display prediction confidence
* Show prediction reason breakdown
* Display dataset preview and column information

---

## Future Improvements

* Add real-time flight tracking API
* Add weather API integration
* Add airport congestion data
* Improve model accuracy using XGBoost or LightGBM
* Add model comparison dashboard
* Add downloadable prediction reports
* Add prediction history
* Deploy live Streamlit app
* Add more detailed delay reason datasets

---

## Author

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=soft&color=0:020617,100:0E76A8&height=120&section=footer&text=Aditya%20Khare&fontSize=36&fontColor=ffffff&animation=fadeIn" />
</p>

**Aditya Khare**

<p>
  <a href="https://github.com/Adityakhare123">
    <img src="https://img.shields.io/badge/GitHub-Adityakhare123-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
  <a href="https://github.com/Adityakhare123/USA-Flight-Delay-Analysis">
    <img src="https://img.shields.io/badge/Project-Repository-0E76A8?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</p>

---

## Support

If you like this project, consider giving it a star on GitHub.

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:38BDF8,50:0E76A8,100:020617&height=120&section=footer" />
</p>
