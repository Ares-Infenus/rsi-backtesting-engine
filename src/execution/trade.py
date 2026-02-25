"""
Trade: immutable data model representing a single completed round-trip trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """
    Represents a single round-trip trade (entry → exit).

    All prices are *after* slippage has been applied.
    All P&L figures are *after* commissions have been deducted.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    trade_id: int

    # ── Entry ─────────────────────────────────────────────────────────────────
    entry_date: datetime
    entry_price: float  # fill price (post-slippage)
    entry_commission: float  # commission paid at entry

    # ── Exit ──────────────────────────────────────────────────────────────────
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_commission: Optional[float] = None

    # ── Position sizing ───────────────────────────────────────────────────────
    shares: float = 0.0  # number of units / fractional shares

    # ── Computed at close ─────────────────────────────────────────────────────
    gross_pnl: float = 0.0  # (exit - entry) × shares
    net_pnl: float = 0.0  # gross_pnl − total_commissions
    return_pct: float = 0.0  # net_pnl / (entry_price × shares)

    # ── Duration ──────────────────────────────────────────────────────────────
    duration_days: Optional[float] = None

    def close(
        self,
        exit_date: datetime,
        exit_price: float,
        exit_commission: float,
    ) -> None:
        """Finalise the trade once the position is closed."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_commission = exit_commission

        cost_basis = self.entry_price * self.shares
        self.gross_pnl = (exit_price - self.entry_price) * self.shares
        total_commissions = self.entry_commission + exit_commission
        self.net_pnl = self.gross_pnl - total_commissions
        self.return_pct = self.net_pnl / cost_basis if cost_basis else 0.0

        delta = exit_date - self.entry_date
        self.duration_days = delta.days + delta.seconds / 86_400

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    def __repr__(self) -> str:  # pragma: no cover
        status = "OPEN" if self.is_open else f"net_pnl={self.net_pnl:+.2f}"
        return (
            f"Trade(#{self.trade_id} | entry={self.entry_price:.4f} "
            f"@ {self.entry_date.date()} | {status})"
        )
