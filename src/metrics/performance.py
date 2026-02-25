"""
Performance metrics: computes Sharpe Ratio, CAGR, Max Drawdown, and Win Rate.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.config.settings import RISK_FREE_RATE, TRADING_DAYS
from src.execution.trade import Trade
from src.metrics.drawdown import max_drawdown, drawdown_duration


def compute_metrics(
    equity: pd.Series,
    trades: List[Trade],
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> dict:
    """
    Compute a full set of performance metrics from equity curve and trade list.

    Parameters
    ----------
    equity         : Time-indexed portfolio equity Series.
    trades         : List of completed Trade objects.
    risk_free_rate : Annual risk-free rate (decimal) for Sharpe calculation.
    trading_days   : Number of trading periods per year (365 for crypto, 252 for equities).

    Returns
    -------
    dict  Metric name → value.
    """
    metrics: dict = {}

    if len(equity) < 2:
        return {"error": "Insufficient data to compute metrics."}

    # ── Returns ────────────────────────────────────────────────────────────────
    daily_returns = equity.pct_change().dropna()
    total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    # ── CAGR ──────────────────────────────────────────────────────────────────
    n_years = len(equity) / trading_days
    cagr = (
        ((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100
        if n_years > 0
        else 0.0
    )

    # ── Sharpe Ratio ──────────────────────────────────────────────────────────
    daily_rf = risk_free_rate / trading_days
    excess = daily_returns - daily_rf
    sharpe = (
        (excess.mean() / excess.std() * np.sqrt(trading_days))
        if excess.std() > 0
        else 0.0
    )

    # ── Sortino Ratio (downside deviation only) ────────────────────────────────
    downside = excess[excess < 0]
    sortino = (
        (excess.mean() / downside.std() * np.sqrt(trading_days))
        if len(downside) > 0 and downside.std() > 0
        else 0.0
    )

    # ── Drawdown ───────────────────────────────────────────────────────────────
    max_dd = max_drawdown(equity)
    dd_dur = drawdown_duration(equity)

    # ── Calmar Ratio ──────────────────────────────────────────────────────────
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # ── Trade-level metrics ────────────────────────────────────────────────────
    n_trades = len(trades)
    if n_trades > 0:
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        win_rate = len(winners) / n_trades * 100

        avg_win = np.mean([t.net_pnl for t in winners]) if winners else 0.0
        avg_loss = np.mean([t.net_pnl for t in losers]) if losers else 0.0
        profit_factor = (
            abs(sum(t.net_pnl for t in winners) / sum(t.net_pnl for t in losers))
            if losers and sum(t.net_pnl for t in losers) != 0
            else float("inf")
        )
        avg_duration = np.mean(
            [t.duration_days for t in trades if t.duration_days is not None]
        )
        best_trade = max(t.net_pnl for t in trades)
        worst_trade = min(t.net_pnl for t in trades)
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_duration = 0.0
        best_trade = worst_trade = 0.0

    # ── Volatility ────────────────────────────────────────────────────────────
    annual_vol = daily_returns.std() * np.sqrt(trading_days) * 100

    metrics = {
        # Overview
        "Total Return (%)": round(total_return_pct, 2),
        "CAGR (%)": round(cagr, 2),
        "Annual Volatility (%)": round(annual_vol, 2),
        # Risk-adjusted
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Calmar Ratio": round(calmar, 3),
        # Drawdown
        "Max Drawdown (%)": round(max_dd, 2),
        "Max DD Duration (bars)": dd_dur,
        # Trades
        "Total Trades": n_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Avg Win ($)": round(avg_win, 2),
        "Avg Loss ($)": round(avg_loss, 2),
        "Profit Factor": round(profit_factor, 3),
        "Best Trade ($)": round(best_trade, 2),
        "Worst Trade ($)": round(worst_trade, 2),
        "Avg Duration (days)": round(avg_duration, 1) if avg_duration else 0,
    }

    return metrics


def format_metrics_table(metrics: dict) -> pd.DataFrame:
    """Convert the metrics dict to a nicely formatted single-column DataFrame."""
    df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    df.index.name = "Metric"
    return df
