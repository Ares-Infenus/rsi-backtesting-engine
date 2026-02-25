"""
Unit tests for performance metrics (Sharpe, CAGR, Max Drawdown, Win Rate).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.metrics.performance import compute_metrics, format_metrics_table
from src.metrics.drawdown import compute_drawdown, max_drawdown, drawdown_duration
from src.execution.trade import Trade


def make_equity(values: list, start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def make_trade(entry_price=100.0, exit_price=110.0, shares=10.0, winner=True) -> Trade:
    t = Trade(
        trade_id=1,
        entry_date=datetime(2020, 1, 10, tzinfo=timezone.utc),
        entry_price=entry_price,
        entry_commission=1.0,
        shares=shares,
    )
    t.close(
        exit_date=datetime(2020, 2, 10, tzinfo=timezone.utc),
        exit_price=exit_price,
        exit_commission=1.0,
    )
    return t


class TestDrawdown:

    def test_no_drawdown_on_monotone_increase(self):
        equity = make_equity(list(range(100, 200)))
        assert max_drawdown(equity) == 0.0

    def test_drawdown_on_simple_decline(self):
        equity = make_equity([100, 90, 80, 70, 100])
        dd = max_drawdown(equity)
        assert dd == pytest.approx(-30.0, abs=0.1)

    def test_drawdown_pct_never_positive(self):
        equity = make_equity([100, 110, 90, 130, 120, 140])
        dd_df = compute_drawdown(equity)
        assert (dd_df["Drawdown_Pct"] <= 0).all()

    def test_drawdown_duration_positive_integer(self):
        equity = make_equity([100, 90, 80, 70, 100])
        dur = drawdown_duration(equity)
        assert isinstance(dur, int) and dur >= 0


class TestPerformanceMetrics:

    def test_metrics_keys_present(self):
        equity = make_equity(list(range(10_000, 11_000)))
        metrics = compute_metrics(equity, [])
        for key in [
            "Total Return (%)",
            "CAGR (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Win Rate (%)",
        ]:
            assert key in metrics

    def test_total_return_flat_equity(self):
        equity = make_equity([10_000] * 365)
        metrics = compute_metrics(equity, [])
        assert metrics["Total Return (%)"] == pytest.approx(0.0, abs=0.01)

    def test_positive_return_on_growing_equity(self):
        equity = make_equity(list(range(10_000, 12_001)))
        metrics = compute_metrics(equity, [])
        assert metrics["Total Return (%)"] > 0

    def test_win_rate_all_winners(self):
        trades = [make_trade(exit_price=110) for _ in range(5)]
        equity = make_equity(list(range(10_000, 10_500)))
        metrics = compute_metrics(equity, trades)
        assert metrics["Win Rate (%)"] == pytest.approx(100.0)

    def test_win_rate_all_losers(self):
        trades = [make_trade(exit_price=90) for _ in range(5)]
        equity = make_equity(list(range(10_000, 9_500, -1)))
        metrics = compute_metrics(equity, trades)
        assert metrics["Win Rate (%)"] == pytest.approx(0.0)

    def test_format_metrics_returns_dataframe(self):
        equity = make_equity(list(range(10_000, 11_000)))
        metrics = compute_metrics(equity, [])
        df = format_metrics_table(metrics)
        assert isinstance(df, pd.DataFrame)
        assert "Value" in df.columns


class TestTrade:

    def test_trade_net_pnl_winner(self):
        t = make_trade(entry_price=100, exit_price=120, shares=10)
        assert t.net_pnl == pytest.approx(
            200.0 - 2.0, abs=0.01
        )  # gross 200 − commissions 2

    def test_trade_net_pnl_loser(self):
        t = make_trade(entry_price=100, exit_price=80, shares=10)
        assert t.net_pnl < 0

    def test_trade_is_winner_flag(self):
        winner = make_trade(exit_price=110)
        loser = make_trade(exit_price=90)
        assert winner.is_winner is True
        assert loser.is_winner is False

    def test_trade_duration_positive(self):
        t = make_trade()
        assert t.duration_days > 0

    def test_trade_is_open_before_close(self):
        t = Trade(
            trade_id=99,
            entry_date=datetime(2021, 1, 1, tzinfo=timezone.utc),
            entry_price=100.0,
            entry_commission=1.0,
            shares=10.0,
        )
        assert t.is_open is True
