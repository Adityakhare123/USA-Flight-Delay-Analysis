# ✈️ USA Flight Delay Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge\&logo=plotly\&logoColor=white)

### Interactive airline delay analysis and machine learning prediction system

</div>

---

## 📌 Project Overview

**USA Flight Delay Analysis** is a data analytics and machine learning project built to analyze flight delay patterns across U.S. airlines, airports, routes, time periods, and flight durations.

The project provides a modern **Streamlit web UI** where users can explore delay insights using interactive charts and predict whether a selected flight is likely to be delayed.

---

## 🚀 Live Demo

🔗 **Live App:** Coming Soon
📂 **GitHub Repository:** [USA Flight Delay Analysis](https://github.com/Adityakhare123/USA-Flight-Delay-Analysis)

---

## 🎯 Objectives

* Analyze U.S. airline flight delay data.
* Identify delay trends by airline, airport, route, weekday, time, and flight duration.
* Build a machine learning model to predict flight delay.
* Provide an interactive dashboard for users.
* Show data-based reason factors behind delay predictions.

---

## ✨ Key Features

### 🏠 Overview Page

* Total flights analyzed
* Delayed flights count
* On-time flights count
* Overall delay rate
* Project feature summary

### 📊 Analytics Dashboard

Interactive charts for:

* Delayed vs not delayed flights
* Delayed flights by time period
* Top airlines by delay rate
* Delay rate by day of week
* Top source airports by delay rate
* Top destination airports by delay rate
* Top routes by delay rate
* Delay rate by flight duration
* Delay reason/risk factor analysis

### 🤖 Delay Predictor

Users can select:

* Airline
* Source airport
* Destination airport
* Day of week
* Scheduled time
* Flight duration

The model predicts:

* ✅ Flight likely to be on time
* ✈️ Flight likely to be delayed
* Prediction confidence
* Reason breakdown using historical delay rates

### 📂 Dataset Explorer

* Dataset preview
* Column information
* Missing value count
* Unique value count
* Duplicate row count

---

## 🧠 Machine Learning Model

The project uses a **Random Forest Classifier** for delay prediction.

### Model Features

The model uses available features such as:

* Airline
* Source airport
* Destination airport
* Day of week
* Scheduled time
* Flight duration
* Source airport type
* Destination airport type
* Airport elevation
* Runway count

### Output

The model predicts:

```text
0 = Not Delayed
1 = Delayed
```

The trained model is saved as:

```text
models/flight_delay_model.pkl
```

Model metrics are saved as:

```text
models/metrics.json
```

---

## 🛠️ Tech Stack

| Category             | Tools        |
| -------------------- | ------------ |
| Programming Language | Python       |
| Web Framework        | Streamlit    |
| Data Processing      | Pandas       |
| Visualization        | Plotly       |
| Machine Learning     | Scikit-learn |
| Model Storage        | Joblib       |
| Version Control      | Git & GitHub |

---

## 📁 Project Structure

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

## 📊 Dataset Information

The main dataset contains flight-related details such as:

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

Additional airport and runway datasets are used to enrich flight information.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Adityakhare123/USA-Flight-Delay-Analysis.git
cd USA-Flight-Delay-Analysis
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Train the Model

Run:

```bash
python train_model.py
```

After training, the following files will be generated:

```text
models/flight_delay_model.pkl
models/metrics.json
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal:

```text
http://localhost:8501
```

---

## 📦 Requirements

```text
streamlit
pandas
plotly
scikit-learn
joblib
openpyxl
```

---

## 🖥️ App Pages

| Page                | Description                           |
| ------------------- | ------------------------------------- |
| Overview            | Project summary and key metrics       |
| Analytics Dashboard | Interactive charts and delay analysis |
| Delay Predictor     | ML-based delay prediction             |
| Dataset Explorer    | Dataset preview and column summary    |

---

## 📈 Model Accuracy

Current trained model accuracy:

```text
64.71%
```

This can be improved further using advanced models and additional external data.

---

## 🔍 Delay Reason Analysis

If actual delay reason columns like `WeatherDelay`, `CarrierDelay`, `NASDelay`, `SecurityDelay`, or `LateAircraftDelay` are available, the app can show actual reason-based analysis.

If those columns are not available, the app generates **data-based delay risk factors** using:

* Airline performance
* Source airport delay pattern
* Destination airport delay pattern
* Route delay pattern
* Time period pattern
* Day of week pattern
* Flight duration pattern

---

## 🚀 Deployment

This project can be deployed on:

* Streamlit Community Cloud
* Render
* Hugging Face Spaces

### Streamlit Deployment Steps

1. Push project to GitHub.
2. Go to Streamlit Community Cloud.
3. Click **New App**.
4. Select this repository.
5. Set main file path:

```text
app.py
```

6. Click **Deploy**.

---

## ⚠️ Large File Note

The dataset file `data/usa.csv` is larger than GitHub’s recommended file size limit.

For better deployment performance, large unused files can be removed from Git tracking:

```bash
git rm --cached data/usa.csv
echo data/usa.csv >> .gitignore
git add .gitignore
git commit -m "Remove large CSV from repository tracking"
git push
```

---

## 🔮 Future Improvements

* Add real-time flight API integration.
* Add weather data for better prediction.
* Add airport congestion data.
* Improve model accuracy using XGBoost or LightGBM.
* Add model comparison dashboard.
* Add downloadable prediction reports.
* Deploy live Streamlit app.
* Add authentication for users.

---

## 👨‍💻 Author

**Aditya Khare**

GitHub: [@Adityakhare123](https://github.com/Adityakhare123)

---

## ⭐ Support

If you like this project, give it a star on GitHub.

```text
Made with Python, Streamlit, Plotly, and Machine Learning.
```
