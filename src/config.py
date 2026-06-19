# EDA helper functions
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

AIRLINES_FILE = DATA_DIR / "Airlines.csv"
AIRPORTS_FILE = DATA_DIR / "airports.csv"
RUNWAYS_FILE = DATA_DIR / "runways.csv"
DATA_DICTIONARY_FILE = DATA_DIR / "Data Dictionary.csv"

PROCESSED_FILE = DATA_DIR / "processed_usa_flight_delay.csv"

MODEL_FILE = MODEL_DIR / "flight_delay_model.pkl"
METRICS_FILE = MODEL_DIR / "metrics.json"