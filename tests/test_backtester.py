"""
Integration tests for the backtesting engine.
Tests use synthetic price data to avoid network calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.backtester.engine import BacktestEngine
from src.data.data_cleaner import validate_and_clean


def make_synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data that triggers RSI signals."""
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    close = np.abs(close)  # ensure positive
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_ = close * (1 + np.random.randn(n) * 0.003)
    vol = np.abs(np.random.randn(n) * 1e6 + 5e6)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


SYNTHETIC_DATA = make_synthetic_ohlcv()


class TestBacktestEngine:

    def _run_engine(self, **kwargs) -> BacktestEngine:
        engine = BacktestEngine(
            commission_rate=0.001,
            slippage_rate=0.0005,
            initial_capital=10_000,
            **kwargs,
        )
        with patch("src.backtester.engine.load_ohlcv", return_value=SYNTHETIC_DATA):
            engine.run()
        return engine

    def test_engine_runs_without_error(self):
        engine = self._run_engine()
        assert engine._ran is True

    def test_equity_curve_has_correct_length(self):
        engine = self._run_engine()
        equity = engine.get_equity_curve()
        assert len(equity) == len(SYNTHETIC_DATA)

    def test_equity_curve_starts_near_initial_capital(self):
        engine = self._run_engine()
        equity = engine.get_equity_curve()
        assert abs(equity.iloc[0] - 10_000) < 100

    def test_equity_is_always_positive(self):
        engine = self._run_engine()
        equity = engine.get_equity_curve()
        assert (equity > 0).all()

    def test_get_trades_returns_dataframe(self):
        engine = self._run_engine()
        trades = engine.get_trades()
        assert isinstance(trades, pd.DataFrame)

    def test_trades_have_required_columns(self):
        engine = self._run_engine()
        trades = engine.get_trades()
        if len(trades) > 0:
            required = {
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "net_pnl",
            }
            assert required.issubset(set(trades.columns))

    def test_raises_if_not_run(self):
        engine = BacktestEngine()
        with pytest.raises(RuntimeError, match="run()"):
            engine.get_equity_curve()

    def test_no_open_positions_at_end(self):
        engine = self._run_engine()
        assert not engine.portfolio.in_position  # type: ignore[union-attr]

    def test_final_cash_is_positive(self):
        engine = self._run_engine()
        assert engine.portfolio.cash >= 0  # type: ignore[union-attr]


class TestDataCleaner:

    def test_clean_passes_valid_data(self):
        df = make_synthetic_ohlcv()
        result = validate_and_clean(df)
        assert len(result) == len(df)

    def test_clean_raises_on_missing_columns(self):
        df = make_synthetic_ohlcv().drop(columns=["Volume"])
        with pytest.raises(ValueError, match="Missing"):
            validate_and_clean(df)

    def test_clean_fills_nan_prices(self):
        df = make_synthetic_ohlcv().copy()
        df.iloc[5, df.columns.get_loc("Close")] = float("nan")
        result = validate_and_clean(df, fill_method="ffill")
        assert result["Close"].isna().sum() == 0

    def test_clean_drops_when_requested(self):
        df = make_synthetic_ohlcv().copy()
        df.iloc[5, df.columns.get_loc("Close")] = float("nan")
        result = validate_and_clean(df, fill_method="drop")
        assert len(result) < len(df)
