from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
# DATA LOADING
# ============================================================

def clean_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_and_prepare_data():
    if not AIRLINES_FILE.exists():
        raise FileNotFoundError("Airlines.csv not found inside data folder.")

    df = pd.read_csv(AIRLINES_FILE)
    df = clean_columns(df)

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

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in Airlines.csv: {missing_cols}")

    # Optional airport + runway merge
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

                airport_features = airports.merge(
                    runway_count,
                    on="ident",
                    how="left",
                )

                airport_features = airport_features[
                    ["iata_code", "type", "elevation_ft", "runway_count"]
                ]

                airport_features["runway_count"] = airport_features["runway_count"].fillna(0)

                # Source airport merge
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

                # Destination airport merge
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

                print("Airport and runway data merged successfully.")

        except Exception as e:
            print(f"Airport/runway merge skipped because of error: {e}")

    # Fill missing values
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    categorical_cols = df.select_dtypes(exclude=["int64", "float64", "int32", "float32"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    return df


# ============================================================
# MODEL TRAINING
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

TARGET_COLUMN = "Delay"


def train_model():
    df = load_and_prepare_data()

    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]

    if len(available_features) == 0:
        raise ValueError("No valid feature columns found for training.")

    X = df[available_features]
    y = df[TARGET_COLUMN].astype(int)

    print("Training features:")
    for col in available_features:
        print(f"- {col}")

    print("\nTraining features:")
    for col in available_features:
        print(f"- {col}")

    print(f"\nTotal rows: {len(df):,}")
    print(f"Delayed flights: {int(y.sum()):,}")
    print(f"Not delayed flights: {int((y == 0).sum()):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = [
        col for col in X_train.columns if col not in numeric_features
    ]

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

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel training completed.")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    model_package = {
        "pipeline": pipeline,
        "features": available_features,
        "accuracy": accuracy,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_package, MODEL_FILE)

    metrics = {
        "accuracy": accuracy,
        "features": available_features,
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

    print(f"\nModel saved at: {MODEL_FILE}")
    print(f"Metrics saved at: {METRICS_FILE}")


if __name__ == "__main__":
    train_model()