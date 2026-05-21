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

    # Location
    out["pickup_zone"] = out["PULocationID"].astype("category")

    # Trip duration
    out["trip_duration_minutes"] = (
        (dropoff - pickup).dt.total_seconds() / 60
    )

    # Drop nonsensical rows
    out = out[out["trip_duration_minutes"] > 0]
    out = out[out["fare_amount"] > 0]

    # Surge label: 1 if fare > 1.5× median
    median_fare = out["fare_amount"].median()
    out["surge_label"] = (out["fare_amount"] > 1.5 * median_fare).astype(int)

    return out.reset_index(drop=True)
