# Prediction utilities
import json
from typing import Any

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_FILE, METRICS_FILE
from src.data_loader import load_master_data


TARGET_COLUMN = "Delay"

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


def get_available_features(df: pd.DataFrame) -> list[str]:
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def prepare_training_data(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    features = get_available_features(df)

    if len(features) == 0:
        raise ValueError("No valid feature columns found for model training.")

    model_df = df[features + [TARGET_COLUMN]].copy()
    model_df = model_df.dropna(subset=[TARGET_COLUMN])

    X = model_df[features]
    y = model_df[TARGET_COLUMN].astype(int)

    return X, y, features


def build_model_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

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

    classifier = RandomForestClassifier(
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
            ("classifier", classifier),
        ]
    )

    return pipeline


def train_and_save_model(df: pd.DataFrame | None = None) -> dict[str, Any]:
    if df is None:
        df = load_master_data()

    X, y, features = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_model_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred).tolist()

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "pipeline": pipeline,
        "features": features,
        "accuracy": accuracy,
    }

    joblib.dump(model_package, MODEL_FILE)

    metrics = {
        "accuracy": accuracy,
        "features": features,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def load_saved_model():
    if not MODEL_FILE.exists():
        return None

    return joblib.load(MODEL_FILE)


def predict_delay(input_df: pd.DataFrame) -> dict:
    model_package = load_saved_model()

    if model_package is None:
        raise FileNotFoundError("Model not found. Please train the model first.")

    pipeline = model_package["pipeline"]
    features = model_package["features"]

    input_df = input_df[features]

    prediction = pipeline.predict(input_df)[0]

    probability = None

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(input_df)[0]
        probability = float(max(proba))

    return {
        "prediction": int(prediction),
        "probability": probability,
    }


if __name__ == "__main__":
    data = load_master_data()
    result = train_and_save_model(data)
    print(result)