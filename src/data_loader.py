# Data loading utilities
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    AIRLINES_FILE,
    AIRPORTS_FILE,
    RUNWAYS_FILE,
    DATA_DICTIONARY_FILE,
    PROCESSED_FILE,
)


def read_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file format: {path.name}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_airlines_data() -> pd.DataFrame:
    df = read_file(AIRLINES_FILE)
    df = clean_column_names(df)
    return df


def load_airports_data() -> pd.DataFrame:
    df = read_file(AIRPORTS_FILE)
    df = clean_column_names(df)
    return df


def load_runways_data() -> pd.DataFrame:
    df = read_file(RUNWAYS_FILE)
    df = clean_column_names(df)
    return df


def load_data_dictionary() -> pd.DataFrame | None:
    if not DATA_DICTIONARY_FILE.exists():
        return None

    try:
        df = read_file(DATA_DICTIONARY_FILE)
        df = clean_column_names(df)
        return df
    except Exception:
        return None


def create_airport_features(airports: pd.DataFrame, runways: pd.DataFrame) -> pd.DataFrame:
    airports = clean_column_names(airports)
    runways = clean_column_names(runways)

    required_airport_cols = ["ident", "iata_code", "type", "elevation_ft"]
    required_runway_cols = ["airport_ident"]

    for col in required_airport_cols:
        if col not in airports.columns:
            raise ValueError(f"Missing column in airports.csv: {col}")

    for col in required_runway_cols:
        if col not in runways.columns:
            raise ValueError(f"Missing column in runways.csv: {col}")

    runway_count = (
        runways.groupby("airport_ident")
        .size()
        .reset_index(name="runway_count")
        .rename(columns={"airport_ident": "ident"})
    )

    airport_features = airports.merge(runway_count, on="ident", how="left")

    airport_features = airport_features[
        ["ident", "iata_code", "type", "elevation_ft", "runway_count"]
    ]

    airport_features["runway_count"] = airport_features["runway_count"].fillna(0)

    return airport_features


def build_master_dataset() -> pd.DataFrame:
    airlines = load_airlines_data()
    airports = load_airports_data()
    runways = load_runways_data()

    required_airline_cols = [
        "Airline",
        "Flight",
        "AirportFrom",
        "AirportTo",
        "DayOfWeek",
        "Time",
        "Length",
        "Delay",
    ]

    for col in required_airline_cols:
        if col not in airlines.columns:
            raise ValueError(f"Missing column in Airlines.csv: {col}")

    airport_features = create_airport_features(airports, runways)

    df = airlines.merge(
        airport_features[["iata_code", "type", "elevation_ft", "runway_count"]],
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
        airport_features[["iata_code", "type", "elevation_ft", "runway_count"]],
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

    df["duration_category"] = pd.cut(
        df["Length"],
        bins=3,
        labels=["Short", "Medium", "Long"],
    ).astype(str)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)

    return df


def load_master_data(force_rebuild: bool = False) -> pd.DataFrame:
    if PROCESSED_FILE.exists() and not force_rebuild:
        return pd.read_csv(PROCESSED_FILE)

    return build_master_dataset()


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }