"""
Unit tests for the RSI indicator module.
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.rsi import compute_rsi, add_rsi_to_df


def make_series(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


class TestComputeRSI:

    def test_output_length_matches_input(self):
        prices = make_series(list(range(1, 51)))
        rsi = compute_rsi(prices, period=14)
        assert len(rsi) == len(prices)

    def test_first_period_values_are_nan(self):
        prices = make_series(list(range(1, 51)))
        rsi = compute_rsi(prices, period=14)
        # First 14 values should be NaN (warm-up)
        assert rsi.iloc[:14].isna().all()

    def test_rsi_bounded_between_0_and_100(self):
        np.random.seed(42)
        prices = make_series(np.cumsum(np.random.randn(200)) + 100)
        rsi = compute_rsi(prices, period=14).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_is_100_when_all_gains(self):
        """Strictly increasing prices should push RSI toward 100."""
        prices = make_series(list(range(1, 51)))
        rsi = compute_rsi(prices, period=14).dropna()
        assert (rsi > 90).all()

    def test_rsi_is_0_when_all_losses(self):
        """Strictly decreasing prices should push RSI toward 0."""
        prices = make_series(list(range(50, 0, -1)))
        rsi = compute_rsi(prices, period=14).dropna()
        assert (rsi < 10).all()

    def test_raises_on_insufficient_data(self):
        prices = make_series([100, 101, 102])
        with pytest.raises(ValueError, match="data points"):
            compute_rsi(prices, period=14)

    def test_raises_on_invalid_period(self):
        prices = make_series(list(range(50)))
        with pytest.raises(ValueError):
            compute_rsi(prices, period=1)

    def test_raises_on_non_series_input(self):
        with pytest.raises(TypeError):
            compute_rsi([1, 2, 3, 4, 5], period=3)  # type: ignore


class TestAddRsiToDf:

    def test_adds_rsi_column(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
        df = pd.DataFrame({"Close": list(range(1, 51))}, index=idx)
        result = add_rsi_to_df(df, period=14)
        assert "RSI_14" in result.columns

    def test_does_not_modify_original(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
        df = pd.DataFrame({"Close": list(range(1, 51))}, index=idx)
        _ = add_rsi_to_df(df, period=14)
        assert "RSI_14" not in df.columns
