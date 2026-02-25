"""
Portfolio: tracks capital, open position, and historical equity.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from src.config.settings import INITIAL_CAPITAL
from src.execution.trade import Trade

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Maintains the state of the portfolio throughout the backtest.

    The portfolio supports a single long position at a time (no shorting,
    no leverage, no partial fills). All available capital is deployed
    on each buy signal.

    Parameters
    ----------
    initial_capital : Starting cash balance in USD.
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL) -> None:
        self.initial_capital = initial_capital
        self.cash: float = initial_capital
        self.open_trade: Optional[Trade] = None
        self.closed_trades: List[Trade] = []
        self._equity_history: List[dict] = []

    # ── State accessors ────────────────────────────────────────────────────────

    @property
    def in_position(self) -> bool:
        return self.open_trade is not None

    def current_equity(self, current_price: float) -> float:
        """Mark-to-market portfolio value."""
        if self.in_position and self.open_trade is not None:
            return self.open_trade.shares * current_price
        return self.cash

    # ── Trade lifecycle ────────────────────────────────────────────────────────

    def open_position(self, trade: Trade) -> None:
        """Record a newly entered trade and deduct cash."""
        if self.in_position:
            raise RuntimeError("Cannot open a new position while one is already open.")
        self.open_trade = trade
        self.cash = 0.0
        logger.debug(
            "Portfolio: opened position at %.4f (trade #%d)",
            trade.entry_price,
            trade.trade_id,
        )

    def close_position(self, trade: Trade) -> None:
        """Record a closed trade and credit cash proceeds."""
        if not self.in_position:
            raise RuntimeError("Cannot close a position when none is open.")
        proceeds = trade.exit_price * trade.shares - trade.exit_commission  # type: ignore[operator]
        self.cash = proceeds
        self.open_trade = None
        self.closed_trades.append(trade)
        logger.debug(
            "Portfolio: closed trade #%d | net_pnl=%.2f | cash=%.2f",
            trade.trade_id,
            trade.net_pnl,
            self.cash,
        )

    # ── Equity tracking ───────────────────────────────────────────────────────

    def record_equity(self, date, current_price: float) -> None:
        """Snapshot current equity for a given bar."""
        self._equity_history.append(
            {"Date": date, "Equity": self.current_equity(current_price)}
        )

    def get_equity_curve(self) -> pd.Series:
        """Return the full equity curve as a time-indexed Series."""
        df = pd.DataFrame(self._equity_history).set_index("Date")
        return df["Equity"]

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "final_cash": self.cash,
            "total_trades": len(self.closed_trades),
            "open_trades": 1 if self.in_position else 0,
        }
