"""Inference helper: load the trained model and predict surge probability."""

import pathlib
import pickle

import numpy as np
import pandas as pd

from features import RUSH_HOURS

MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "surge_model.pkl"
ZONE_FARE_MAP_PATH = pathlib.Path(__file__).parent.parent / "models" / "zone_fare_map.pkl"

_model = None
_zone_fare_map = None


def _get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def _get_zone_fare_map() -> dict:
    global _zone_fare_map
    if _zone_fare_map is None:
        with open(ZONE_FARE_MAP_PATH, "rb") as f:
            _zone_fare_map = pickle.load(f)
    return _zone_fare_map


def predict_surge_probability(
    pickup_hour: int,
    day_of_week: int,
    pickup_zone: int,
    trip_distance: float,
    passenger_count: int = 1,
    trip_duration_minutes: float = 10.0,
    month: int = 1,
) -> float:
    """Return the probability that a trip is a surge fare.

    Args:
        pickup_hour: Hour of pickup (0–23).
        day_of_week: Day of week (0=Monday … 6=Sunday).
        pickup_zone: NYC TLC PULocationID (1–263).
        trip_distance: Trip distance in miles.
        passenger_count: Number of passengers (default 1).
        trip_duration_minutes: Estimated trip duration in minutes (default 10).
        month: Calendar month (1–12, default 1).

    Returns:
        Float in [0, 1] — probability that surge_label == 1.
    """
    is_weekend = int(day_of_week >= 5)
    is_rush_hour = int(pickup_hour in RUSH_HOURS)
    is_peak_season = int(month in {6, 7, 8, 12})

    row = pd.DataFrame(
        [
            {
                "hour": pickup_hour,
                "day_of_week": day_of_week,
                "hour_sin": np.sin(2 * np.pi * pickup_hour / 24),
                "hour_cos": np.cos(2 * np.pi * pickup_hour / 24),
                "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
                "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
                "is_weekend": is_weekend,
                "is_rush_hour": is_rush_hour,
                "month": month,
                "is_peak_season": is_peak_season,
                "trip_duration_minutes": trip_duration_minutes,
                "pickup_zone": pickup_zone,
                "passenger_count": passenger_count,
                "trip_distance": trip_distance,
                "trip_speed_mph": min(
                    trip_distance / max(trip_duration_minutes / 60, 1e-6), 80.0
                ),
            }
        ]
    )

    zone_fare_map = _get_zone_fare_map()
    values = list(zone_fare_map.values())
    fallback = float(np.median(values)) if values else 12.5
    row["zone_median_fare"] = float(zone_fare_map.get(pickup_zone, fallback))

    model = _get_model()
    prob: float = model.predict_proba(row)[0][1]
    return float(prob)


if __name__ == "__main__":
    p = predict_surge_probability(
        pickup_hour=8, day_of_week=1, pickup_zone=132, trip_distance=3.5
    )
    print(f"Surge probability: {p:.4f}")
