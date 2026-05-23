"""Tests for src/train.py::load_and_prepare."""

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import train


def _raw_df(n: int = 30) -> pd.DataFrame:
    base = pd.Timestamp("2024-01-15 08:00:00")
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": [base + pd.Timedelta(minutes=i * 10) for i in range(n)],
            "tpep_dropoff_datetime": [
                base + pd.Timedelta(minutes=i * 10 + 15) for i in range(n)
            ],
            "fare_amount": [float(5 + i) for i in range(n)],
            "PULocationID": [i % 10 + 1 for i in range(n)],
            "passenger_count": [1] * n,
            "trip_distance": [float(1 + i % 5) for i in range(n)],
        }
    )


@pytest.fixture()
def patched_train(tmp_path, monkeypatch):
    raw = _raw_df(30)
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path)
    monkeypatch.setattr(train, "RAW_PATH", raw_path)
    monkeypatch.setattr(train, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(train, "ZONE_FARE_MAP_PATH", tmp_path / "zone_fare_map.pkl")
    return train


def test_load_and_prepare_returns_matching_shapes(patched_train):
    X, y = patched_train.load_and_prepare()
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0


def test_load_and_prepare_y_is_binary(patched_train):
    _, y = patched_train.load_and_prepare()
    assert set(y.unique()) <= {0, 1}
    assert y.name == "surge_label"


def test_load_and_prepare_no_object_columns(patched_train):
    X, _ = patched_train.load_and_prepare()
    obj_cols = X.select_dtypes(["category", "object"]).columns.tolist()
    assert obj_cols == [], f"Unencoded columns remain: {obj_cols}"


def test_load_and_prepare_expected_features_present(patched_train):
    X, _ = patched_train.load_and_prepare()
    for col in ["hour", "trip_duration_minutes", "trip_speed_mph", "zone_median_fare"]:
        assert col in X.columns, f"Expected feature '{col}' missing from X"


def test_features_list_has_no_duplicates():
    seen = set()
    for feat in train.FEATURES:
        assert feat not in seen, f"Duplicate in FEATURES: {feat}"
        seen.add(feat)


def test_zone_fare_map_written(patched_train, tmp_path):
    patched_train.load_and_prepare()
    assert (tmp_path / "zone_fare_map.pkl").exists()
