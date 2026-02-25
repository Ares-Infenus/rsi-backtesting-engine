"""
Broker simulator: applies commission and slippage to simulated order fills.

Slippage model: adverse price impact proportional to the fill price.
  - BUY  fills at price × (1 + slippage_rate)   — you pay more
  - SELL fills at price × (1 - slippage_rate)   — you receive less

Commission model: flat percentage of the total notional value per side.
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.config.settings import COMMISSION_RATE, SLIPPAGE_RATE
from src.execution.trade import Trade

logger = logging.getLogger(__name__)


class BrokerSimulator:
    """
    Simulates order execution with realistic transaction costs.

    Parameters
    ----------
    commission_rate : Fraction of notional charged per trade side (default 0.001 = 0.1%).
    slippage_rate   : Adverse price impact per fill (default 0.0005 = 0.05%).
    """

    def __init__(
        self,
        commission_rate: float = COMMISSION_RATE,
        slippage_rate: float = SLIPPAGE_RATE,
    ) -> None:
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self._trade_counter = 0

    # ── Public interface ───────────────────────────────────────────────────────

    def execute_buy(
        self,
        date: datetime,
        raw_price: float,
        capital: float,
    ) -> Trade:
        """
        Open a long position, spending all available *capital*.

        Returns
        -------
        Trade  The newly created open trade.
        """
        fill_price = self._apply_slippage(raw_price, side="buy")
        notional = capital
        commission = notional * self.commission_rate
        net_capital = notional - commission  # capital available for shares
        shares = net_capital / fill_price

        self._trade_counter += 1
        trade = Trade(
            trade_id=self._trade_counter,
            entry_date=date,
            entry_price=fill_price,
            entry_commission=commission,
            shares=shares,
        )

        logger.debug(
            "BUY  #%d | raw=%.4f → fill=%.4f | shares=%.6f | comm=%.4f",
            trade.trade_id,
            raw_price,
            fill_price,
            shares,
            commission,
        )
        return trade

    def execute_sell(
        self,
        date: datetime,
        raw_price: float,
        trade: Trade,
    ) -> Trade:
        """
        Close an open long position.

        Returns
        -------
        Trade  The same trade object, now populated with exit information.
        """
        fill_price = self._apply_slippage(raw_price, side="sell")
        notional = fill_price * trade.shares
        commission = notional * self.commission_rate

        trade.close(
            exit_date=date,
            exit_price=fill_price,
            exit_commission=commission,
        )

        logger.debug(
            "SELL #%d | raw=%.4f → fill=%.4f | net_pnl=%.4f (%.2f%%)",
            trade.trade_id,
            raw_price,
            fill_price,
            trade.net_pnl,
            trade.return_pct * 100,
        )
        return trade

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _apply_slippage(self, price: float, side: str) -> float:
        if side == "buy":
            return price * (1 + self.slippage_rate)
        return price * (1 - self.slippage_rate)
