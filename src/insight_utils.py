import numpy as np
import pandas as pd


ACTUAL_REASON_COLUMN_MAP = {
    "carrierdelay": "Carrier / Airline Delay",
    "weatherdelay": "Weather Delay",
    "nasdelay": "Air Traffic / NAS Delay",
    "securitydelay": "Security Delay",
    "lateaircraftdelay": "Late Aircraft Delay",
    "carrier_delay": "Carrier / Airline Delay",
    "weather_delay": "Weather Delay",
    "nas_delay": "Air Traffic / NAS Delay",
    "security_delay": "Security Delay",
    "late_aircraft_delay": "Late Aircraft Delay",
}


def normalize_column_name(col: str) -> str:
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def get_actual_reason_columns(df: pd.DataFrame) -> dict:
    """
    Detects actual delay reason columns if your dataset contains columns like:
    CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay.
    """
    found = {}

    for col in df.columns:
        normalized = normalize_column_name(col)

        for key, label in ACTUAL_REASON_COLUMN_MAP.items():
            normalized_key = normalize_column_name(key)

            if normalized == normalized_key:
                found[col] = label

    return found


def time_to_period(value, use_minutes: bool = True) -> str:
    """
    Converts flight time into readable period.

    In your Airlines dataset, Time is usually minutes from midnight.
    Example:
    15 = 00:15
    600 = 10:00
    1020 = 17:00
    """
    try:
        value = float(value)
    except Exception:
        return "Unknown"

    if use_minutes:
        hour = int(value // 60)
    else:
        hour = int(value)

    if hour < 0:
        return "Unknown"

    if 0 <= hour <= 5:
        return "Late Night"

    if 6 <= hour <= 11:
        return "Morning"

    if 12 <= hour <= 17:
        return "Afternoon"

    if 18 <= hour <= 23:
        return "Evening / Night"

    return "Unknown"


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds clean columns used for charts and prediction explanation.
    """
    df = df.copy()

    if "Delay" in df.columns:
        df["Delay_Label"] = df["Delay"].map(
            {
                0: "Not Delayed",
                1: "Delayed",
            }
        )

    if "Time" in df.columns:
        use_minutes = df["Time"].max() > 24
        df["Time_Period"] = df["Time"].apply(lambda x: time_to_period(x, use_minutes))

    if "Length" in df.columns and "duration_category" not in df.columns:
        df["duration_category"] = pd.cut(
            df["Length"],
            bins=3,
            labels=["Short", "Medium", "Long"],
            include_lowest=True,
        ).astype(str)

    return df


def delay_rate(df: pd.DataFrame) -> float:
    if df.empty or "Delay" not in df.columns:
        return 0.0

    return float(df["Delay"].mean() * 100)


def get_group_delay_rate(
    df: pd.DataFrame,
    group_col: str,
    min_flights: int = 50,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Returns delay rate by a selected column.
    """
    if group_col not in df.columns or "Delay" not in df.columns:
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
    result["Delay Rate (%)"] = result["delay_rate"] * 100
    result = result.sort_values("Delay Rate (%)", ascending=False).head(top_n)

    return result


def get_actual_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    If actual reason columns are present, create delay reason summary.
    """
    reason_cols = get_actual_reason_columns(df)

    if not reason_cols:
        return pd.DataFrame()

    rows = []

    for col, label in reason_cols.items():
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

    return result


def get_proxy_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    When actual reason columns are missing, this gives data-based reason factors.
    These are not official delay causes; they are analytical indicators from available features.
    """
    df = add_analysis_columns(df)

    if "Delay" not in df.columns:
        return pd.DataFrame()

    overall = delay_rate(df)

    factors = []

    def add_factor(label, column, min_flights=50):
        if column not in df.columns:
            return

        grouped = (
            df.groupby(column)
            .agg(total_flights=("Delay", "count"), delay_rate=("Delay", "mean"))
            .reset_index()
        )

        grouped = grouped[grouped["total_flights"] >= min_flights]

        if grouped.empty:
            return

        grouped["Delay Rate (%)"] = grouped["delay_rate"] * 100
        top_rate = grouped["Delay Rate (%)"].max()
        risk_gap = max(top_rate - overall, 0)

        factors.append(
            {
                "Reason Factor": label,
                "Risk Score": round(risk_gap, 2),
                "Highest Delay Rate (%)": round(top_rate, 2),
            }
        )

    add_factor("Airline performance pattern", "Airline")
    add_factor("Source airport congestion pattern", "AirportFrom")
    add_factor("Destination airport congestion pattern", "AirportTo")
    add_factor("Day of week pattern", "DayOfWeek")
    add_factor("Time period pattern", "Time_Period")
    add_factor("Flight duration pattern", "duration_category")
    add_factor("Source airport type pattern", "type_source_airport")
    add_factor("Destination airport type pattern", "type_dest_airport")

    result = pd.DataFrame(factors)

    if result.empty:
        return result

    result = result.sort_values("Risk Score", ascending=False)

    return result


def get_route_delay_rate(df: pd.DataFrame, top_n: int = 15, min_flights: int = 30) -> pd.DataFrame:
    if not {"AirportFrom", "AirportTo", "Delay"}.issubset(df.columns):
        return pd.DataFrame()

    route_df = df.copy()
    route_df["Route"] = route_df["AirportFrom"].astype(str) + " → " + route_df["AirportTo"].astype(str)

    result = (
        route_df.groupby("Route")
        .agg(
            total_flights=("Delay", "count"),
            delayed_flights=("Delay", "sum"),
            delay_rate=("Delay", "mean"),
        )
        .reset_index()
    )

    result = result[result["total_flights"] >= min_flights]
    result["Delay Rate (%)"] = result["delay_rate"] * 100
    result = result.sort_values("Delay Rate (%)", ascending=False).head(top_n)

    return result


def get_input_duration_category(df: pd.DataFrame, length_value) -> str:
    if "Length" not in df.columns:
        return "Unknown"

    try:
        length_value = float(length_value)
    except Exception:
        return "Unknown"

    try:
        _, bins = pd.cut(
            df["Length"],
            bins=3,
            labels=["Short", "Medium", "Long"],
            retbins=True,
            include_lowest=True,
        )

        category = pd.cut(
            [length_value],
            bins=bins,
            labels=["Short", "Medium", "Long"],
            include_lowest=True,
        )[0]

        return str(category)
    except Exception:
        q1 = df["Length"].quantile(0.33)
        q2 = df["Length"].quantile(0.66)

        if length_value <= q1:
            return "Short"

        if length_value <= q2:
            return "Medium"

        return "Long"


def get_filter_delay_rate(df: pd.DataFrame, mask) -> tuple:
    temp = df[mask]

    if temp.empty:
        return 0.0, 0

    return delay_rate(temp), len(temp)


def get_prediction_reason_breakdown(df: pd.DataFrame, input_data: dict) -> pd.DataFrame:
    """
    Explains prediction using historical delay rate of selected user inputs.
    """
    df = add_analysis_columns(df)

    overall_rate = delay_rate(df)

    use_minutes = True
    if "Time" in df.columns:
        use_minutes = df["Time"].max() > 24

    input_time_period = time_to_period(input_data.get("Time"), use_minutes)
    input_duration_category = get_input_duration_category(df, input_data.get("Length"))

    rows = []

    def add_reason(label, mask):
        selected_rate, sample_size = get_filter_delay_rate(df, mask)
        risk_gap = selected_rate - overall_rate

        rows.append(
            {
                "Reason": label,
                "Dataset Average Delay Rate (%)": round(overall_rate, 2),
                "Selected Input Delay Rate (%)": round(selected_rate, 2),
                "Risk Gap (%)": round(risk_gap, 2),
                "Sample Size": sample_size,
                "Risk Score": round(max(risk_gap, 0), 2),
            }
        )

    if "Airline" in df.columns:
        add_reason(
            f"Airline risk: {input_data.get('Airline')}",
            df["Airline"].astype(str) == str(input_data.get("Airline")),
        )

    if "AirportFrom" in df.columns:
        add_reason(
            f"Source airport risk: {input_data.get('AirportFrom')}",
            df["AirportFrom"].astype(str) == str(input_data.get("AirportFrom")),
        )

    if "AirportTo" in df.columns:
        add_reason(
            f"Destination airport risk: {input_data.get('AirportTo')}",
            df["AirportTo"].astype(str) == str(input_data.get("AirportTo")),
        )

    if {"AirportFrom", "AirportTo"}.issubset(df.columns):
        add_reason(
            f"Route risk: {input_data.get('AirportFrom')} → {input_data.get('AirportTo')}",
            (df["AirportFrom"].astype(str) == str(input_data.get("AirportFrom")))
            & (df["AirportTo"].astype(str) == str(input_data.get("AirportTo"))),
        )

    if "DayOfWeek" in df.columns:
        add_reason(
            f"Weekday risk: Day {input_data.get('DayOfWeek')}",
            df["DayOfWeek"].astype(str) == str(input_data.get("DayOfWeek")),
        )

    if "Time_Period" in df.columns:
        add_reason(
            f"Time period risk: {input_time_period}",
            df["Time_Period"].astype(str) == str(input_time_period),
        )

    if "duration_category" in df.columns:
        add_reason(
            f"Flight duration risk: {input_duration_category}",
            df["duration_category"].astype(str) == str(input_duration_category),
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = result.sort_values("Risk Score", ascending=False)

    if result["Risk Score"].sum() == 0:
        result["Risk Score"] = result["Selected Input Delay Rate (%)"]

    return result