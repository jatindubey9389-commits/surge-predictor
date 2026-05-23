"""Feature engineering for the surge pricing predictor."""

import numpy as np
import pandas as pd


RUSH_HOURS = {7, 8, 9, 17, 18, 19}  # AM: 7-9, PM: 5-7


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw NYC TLC trip data.

    Args:
        df: Raw parquet DataFrame from data/raw.parquet.

    Returns:
        DataFrame with engineered columns appended; rows with invalid
        fare or duration are dropped.
    """
    out = df.copy()

    # Time features
    pickup = pd.to_datetime(out["tpep_pickup_datetime"])
    dropoff = pd.to_datetime(out["tpep_dropoff_datetime"])

    out["hour"] = pickup.dt.hour
    out["day_of_week"] = pickup.dt.dayofweek          # 0=Mon … 6=Sun
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_rush_hour"] = out["hour"].isin(RUSH_HOURS).astype(int)

    # Cyclical encoding so model sees 23→0 and Sun→Mon as adjacent
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    # Location
    out["pickup_zone"] = out["PULocationID"].astype("category")

    # Trip duration
    out["trip_duration_minutes"] = (
        (dropoff - pickup).dt.total_seconds() / 60
    )

    # Drop nonsensical rows
    out = out[out["trip_duration_minutes"] > 0]
    out = out[out["fare_amount"] > 0]

    # Speed: low speed signals heavy traffic, which correlates with surge demand
    out["trip_speed_mph"] = (
        out["trip_distance"] / (out["trip_duration_minutes"] / 60)
    ).clip(upper=80.0)

    # Per-zone median fare captures neighbourhood-level pricing pressure
    out["zone_median_fare"] = out.groupby("PULocationID")["fare_amount"].transform("median")

    # Surge label: 1 if fare > 1.5× median
    median_fare = out["fare_amount"].median()
    out["surge_label"] = (out["fare_amount"] > 1.5 * median_fare).astype(int)

    return out.reset_index(drop=True)
