<div align="center">

# ✈️ USA Flight Delay Analysis

### 🚀 Interactive Flight Delay Dashboard + Machine Learning Prediction App

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&duration=2500&pause=800&color=0E76A8&center=true&vCenter=true&width=900&lines=Analyze+USA+Flight+Delay+Patterns;Predict+Flight+Delay+Using+Machine+Learning;Visualize+Airline%2C+Airport%2C+Route+%26+Time+Insights;Built+with+Python+%2B+Streamlit+%2B+Plotly" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge\&logo=plotly\&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge\&logo=github\&logoColor=white)

<br/>

<img src="https://skillicons.dev/icons?i=python,git,github,vscode" />

</div>

---

## 📌 About The Project

**USA Flight Delay Analysis** is a data analytics and machine learning project built to analyze U.S. airline flight delay patterns and predict whether a selected flight is likely to be delayed.

The project includes a modern **Streamlit web application** where users can explore flight delay insights using interactive charts, filter airline/airport data, and use a trained machine learning model for delay prediction.

---

## 🚀 Live Demo

🔗 **Live App:** Coming Soon
📂 **GitHub Repository:** [USA Flight Delay Analysis](https://github.com/Adityakhare123/USA-Flight-Delay-Analysis)

---

## 🎯 Project Objectives

* 📊 Analyze U.S. airline flight delay data.
* 🛫 Identify delay trends by airline, airport, route, weekday, time, and flight duration.
* 🤖 Build a machine learning model to predict flight delays.
* 📈 Create an interactive dashboard for users.
* 🧠 Show data-based reason factors behind delay predictions.
* 🌐 Prepare the project for live deployment using Streamlit.

---

## ✨ Key Features

<table>
  <tr>
    <td width="50%">
      <h3>🏠 Overview Dashboard</h3>
      <ul>
        <li>Total flights analyzed</li>
        <li>Delayed flights count</li>
        <li>On-time flights count</li>
        <li>Overall delay percentage</li>
      </ul>
    </td>
    <td width="50%">
      <h3>📊 Analytics Dashboard</h3>
      <ul>
        <li>Delay distribution pie charts</li>
        <li>Airline delay analysis</li>
        <li>Airport delay analysis</li>
        <li>Route and duration insights</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🤖 Delay Predictor</h3>
      <ul>
        <li>ML-based delay prediction</li>
        <li>Prediction confidence score</li>
        <li>Reason breakdown charts</li>
        <li>Historical delay risk factors</li>
      </ul>
    </td>
    <td width="50%">
      <h3>📂 Dataset Explorer</h3>
      <ul>
        <li>Dataset preview</li>
        <li>Column information</li>
        <li>Missing value summary</li>
        <li>Unique value count</li>
      </ul>
    </td>
  </tr>
</table>

---

## 🧠 Machine Learning Model

The project uses a **Random Forest Classifier** to predict whether a flight is delayed or not.

### ✅ Model Input Features

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

### 🎯 Prediction Output

```text
0 = Not Delayed
1 = Delayed
```

### 📌 Current Model Accuracy

```text
64.71%
```

The model file is saved as:

```text
models/flight_delay_model.pkl
```

Model metrics are saved as:

```text
models/metrics.json
```

---

## 📊 Dashboard Insights

The app provides interactive visual analysis for:

* 🥧 Delayed vs not delayed flights
* ⏰ Delayed flights by time period
* 🏢 Top airlines by delay rate
* 📅 Delay rate by day of week
* 🛫 Top source airports by delay rate
* 🛬 Top destination airports by delay rate
* 🧭 Top routes by delay rate
* 📏 Delay rate by flight duration
* 🧠 Delay reason/risk factor analysis

---

## 🛠️ Tech Stack

<div align="center">

| Technology                                                                                                                                    | Purpose                    |
| --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="35"/> **Python**                 | Core programming language  |
| <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="35"/> **Streamlit**                                              | Web app and dashboard UI   |
| <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" width="35"/> **Pandas**                 | Data cleaning and analysis |
| <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" width="35"/> **NumPy**                    | Numerical operations       |
| <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/scikitlearn/scikitlearn-original.svg" width="35"/> **Scikit-learn** | Machine learning model     |
| 📈 **Plotly**                                                                                                                                 | Interactive visualizations |
| 💾 **Joblib**                                                                                                                                 | Model saving and loading   |
| <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/git/git-original.svg" width="35"/> **Git**                          | Version control            |
| <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/github/github-original.svg" width="35"/> **GitHub**                 | Code hosting               |

</div>

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

## 📂 Dataset Information

The main dataset contains flight-level details.

| Column        | Description           |
| ------------- | --------------------- |
| `Airline`     | Airline code          |
| `Flight`      | Flight number         |
| `AirportFrom` | Source airport        |
| `AirportTo`   | Destination airport   |
| `DayOfWeek`   | Day of week           |
| `Time`        | Scheduled flight time |
| `Length`      | Flight duration       |
| `Delay`       | Target variable       |

Additional airport and runway datasets are used to enrich the analysis with airport type, elevation, and runway count.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone The Repository

```bash
git clone https://github.com/Adityakhare123/USA-Flight-Delay-Analysis.git
cd USA-Flight-Delay-Analysis
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Train The Model

Run this command:

```bash
python train_model.py
```

After successful training, these files will be generated:

```text
models/flight_delay_model.pkl
models/metrics.json
```

---

## ▶️ Run The Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal:

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

| Page                   | Description                           |
| ---------------------- | ------------------------------------- |
| 🏠 Overview            | Project summary and key metrics       |
| 📊 Analytics Dashboard | Interactive charts and delay analysis |
| 🤖 Delay Predictor     | ML-based flight delay prediction      |
| 📂 Dataset Explorer    | Dataset preview and column summary    |

---

## 🔍 Delay Reason Analysis

If actual delay reason columns like:

```text
WeatherDelay
CarrierDelay
NASDelay
SecurityDelay
LateAircraftDelay
```

are available, the app can show actual reason-based analysis.

If these columns are not available, the app generates **data-based delay risk factors** using:

* Airline performance pattern
* Source airport delay pattern
* Destination airport delay pattern
* Route delay pattern
* Day of week pattern
* Time period pattern
* Flight duration pattern

---

## 🚀 Deployment

This project can be deployed on:

* Streamlit Community Cloud
* Render
* Hugging Face Spaces

### Streamlit Community Cloud Deployment

1. Push the project to GitHub.
2. Go to Streamlit Community Cloud.
3. Click **New App**.
4. Select this repository.
5. Set the main file path:

```text
app.py
```

6. Click **Deploy**.

---

## ⚠️ Large File Note

The file `data/usa.csv` is larger than GitHub’s recommended file size limit.

If deployment becomes slow or fails, remove the large CSV from Git tracking:

```bash
git rm --cached data/usa.csv
echo data/usa.csv >> .gitignore
git add .gitignore
git commit -m "Remove large CSV from repository tracking"
git push
```

---

## 🔮 Future Enhancements

* 🌦️ Add real-time weather API integration
* 🛫 Add live flight tracking API
* 🧠 Improve accuracy using XGBoost or LightGBM
* 📊 Add model comparison dashboard
* 📄 Add downloadable prediction reports
* 🌐 Deploy live app on Streamlit Cloud
* 🔐 Add user authentication
* 🧾 Add prediction history

---

## 👨‍💻 Author

<div align="center">

### **Aditya Khare**

[![GitHub](https://img.shields.io/badge/GitHub-Adityakhare123-181717?style=for-the-badge\&logo=github)](https://github.com/Adityakhare123)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Aditya%20Khare-0A66C2?style=for-the-badge\&logo=linkedin\&logoColor=white)](#)

</div>

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub.

<div align="center">

```text
Made with ❤️ using Python, Streamlit, Plotly, and Machine Learning
```

</div>
