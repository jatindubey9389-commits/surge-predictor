"""Tests for src/features.py::build_features."""

import pandas as pd
import pytest

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from features import build_features

EXPECTED_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "pickup_zone",
    "trip_duration_minutes",
    "surge_label",
]


def _make_raw(n: int = 20) -> pd.DataFrame:
    """Return a minimal raw-TLC-style DataFrame."""
    base = pd.Timestamp("2024-01-15 08:00:00")
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": [base + pd.Timedelta(minutes=i * 10) for i in range(n)],
            "tpep_dropoff_datetime": [
                base + pd.Timedelta(minutes=i * 10 + 15) for i in range(n)
            ],
            "fare_amount": [float(5 + i) for i in range(n)],
            "PULocationID": [i % 263 + 1 for i in range(n)],
            "passenger_count": [1] * n,
            "trip_distance": [float(1 + i % 5) for i in range(n)],
        }
    )


def test_expected_columns_present():
    df = build_features(_make_raw())
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_no_nulls_in_key_fields():
    df = build_features(_make_raw())
    key_fields = ["hour", "day_of_week", "is_weekend", "is_rush_hour", "surge_label"]
    for col in key_fields:
        assert df[col].isna().sum() == 0, f"Nulls found in {col}"


def test_surge_label_is_binary():
    df = build_features(_make_raw())
    unique = set(df["surge_label"].unique())
    assert unique <= {0, 1}, f"surge_label has non-binary values: {unique}"


def test_trip_duration_positive():
    df = build_features(_make_raw())
    assert (df["trip_duration_minutes"] > 0).all()


def test_invalid_rows_dropped():
    raw = _make_raw()
    # Inject one row with negative fare and one with zero duration
    raw.loc[0, "fare_amount"] = -1.0
    raw.loc[1, "tpep_dropoff_datetime"] = raw.loc[1, "tpep_pickup_datetime"]
    df = build_features(raw)
    assert len(df) < len(raw)


def test_is_weekend_values():
    df = build_features(_make_raw())
    assert set(df["is_weekend"].unique()) <= {0, 1}


def test_is_rush_hour_values():
    df = build_features(_make_raw())
    assert set(df["is_rush_hour"].unique()) <= {0, 1}
