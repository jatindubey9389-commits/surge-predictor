"""Tests for src/predict.py — model and zone map are mocked so no artifacts needed."""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from predict import predict_surge_probability


def _mock_model(proba: float = 0.7):
    m = MagicMock()
    m.predict_proba.return_value = [[1 - proba, proba]]
    return m


_ZONE_MAP = {132: 12.5, 1: 8.0, 237: 20.0}


@patch("predict._get_zone_fare_map", return_value=_ZONE_MAP)
@patch("predict._get_model")
def test_returns_float_in_unit_interval(mock_model, mock_map):
    mock_model.return_value = _mock_model(0.7)
    result = predict_surge_probability(
        pickup_hour=8, day_of_week=1, pickup_zone=132, trip_distance=3.5
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@patch("predict._get_zone_fare_map", return_value=_ZONE_MAP)
@patch("predict._get_model")
def test_returns_exact_model_proba(mock_model, mock_map):
    mock_model.return_value = _mock_model(0.85)
    result = predict_surge_probability(
        pickup_hour=18, day_of_week=4, pickup_zone=237, trip_distance=5.0
    )
    assert result == pytest.approx(0.85)


@patch("predict._get_zone_fare_map", return_value={})
@patch("predict._get_model")
def test_handles_unknown_zone_gracefully(mock_model, mock_map):
    """Unknown zone should fall back to the hardcoded default, not raise."""
    mock_model.return_value = _mock_model(0.3)
    result = predict_surge_probability(
        pickup_hour=12, day_of_week=2, pickup_zone=999, trip_distance=1.0
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@patch("predict._get_zone_fare_map", return_value=_ZONE_MAP)
@patch("predict._get_model")
def test_model_receives_all_expected_features(mock_model, mock_map):
    mock_model.return_value = _mock_model()
    predict_surge_probability(
        pickup_hour=8, day_of_week=0, pickup_zone=132, trip_distance=2.0
    )
    call_args = mock_model.return_value.predict_proba.call_args
    df_passed = call_args[0][0]
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "zone_median_fare"]:
        assert col in df_passed.columns, f"Feature '{col}' missing from predict input"
