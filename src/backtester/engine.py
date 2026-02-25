"""
Backtesting engine: orchestrates data → signals → execution → portfolio.

The engine iterates bar-by-bar to avoid any look-ahead bias.
Trades are executed on the OPEN price of the bar *following* the signal.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.config.settings import (
    COMMISSION_RATE,
    SLIPPAGE_RATE,
    INITIAL_CAPITAL,
    RSI_PERIOD,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
)
from src.data.data_loader import load_ohlcv
from src.data.data_cleaner import validate_and_clean
from src.strategy.rsi_strategy import generate_signals
from src.execution.broker_simulator import BrokerSimulator
from src.portfolio.portfolio import Portfolio

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Main backtesting engine for the RSI mean-reversion strategy.

    Execution model
    ---------------
    - Signal generated at bar N close.
    - Order filled at bar N+1 open (to avoid intrabar look-ahead).
    - Full capital deployment per trade (no position sizing beyond 100%).

    Parameters
    ----------
    ticker, start, end, interval : Data parameters (passed to data_loader).
    initial_capital              : Starting portfolio value.
    commission_rate              : Per-side commission as a decimal fraction.
    slippage_rate                : Adverse price impact per fill.
    rsi_period                   : RSI calculation window.
    rsi_oversold                 : Buy-signal RSI threshold.
    rsi_overbought               : Sell-signal RSI threshold.
    """

    def __init__(
        self,
        ticker: str = "BTC-USD",
        start: str = "2020-01-01",
        end: str = "2024-12-31",
        interval: str = "1d",
        initial_capital: float = INITIAL_CAPITAL,
        commission_rate: float = COMMISSION_RATE,
        slippage_rate: float = SLIPPAGE_RATE,
        rsi_period: int = RSI_PERIOD,
        rsi_oversold: float = RSI_OVERSOLD,
        rsi_overbought: float = RSI_OVERBOUGHT,
    ) -> None:
        self.ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Set after run()
        self.data: Optional[pd.DataFrame] = None
        self.portfolio: Optional[Portfolio] = None
        self._ran = False

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self) -> "BacktestEngine":
        """
        Execute the full backtest pipeline.

        Returns self to allow method chaining: engine.run().get_results()
        """
        logger.info("═" * 60)
        logger.info(
            "Starting backtest: %s | %s → %s", self.ticker, self.start, self.end
        )
        logger.info(
            "RSI(%d) | oversold=%.0f | overbought=%.0f",
            self.rsi_period,
            self.rsi_oversold,
            self.rsi_overbought,
        )
        logger.info(
            "Commission=%.3f%%  Slippage=%.3f%%",
            self.commission_rate * 100,
            self.slippage_rate * 100,
        )

        # 1. Load & clean data
        raw_df = load_ohlcv(self.ticker, self.start, self.end, self.interval)
        clean_df = validate_and_clean(raw_df)

        # 2. Generate signals
        signal_df = generate_signals(
            clean_df,
            rsi_period=self.rsi_period,
            oversold=self.rsi_oversold,
            overbought=self.rsi_overbought,
        )
        self.data = signal_df

        # 3. Initialise broker and portfolio
        broker = BrokerSimulator(self.commission_rate, self.slippage_rate)
        portfolio = Portfolio(self.initial_capital)

        # 4. Bar-by-bar simulation
        bars = signal_df.reset_index()  # work with integer index for next-bar access

        for i, row in bars.iterrows():
            date = row["Date"] if "Date" in row else row.iloc[0]
            close = float(row["Close"])
            signal = int(row["Signal"])

            # Record equity snapshot at every bar
            portfolio.record_equity(date, close)

            # Check if we need to execute on THIS bar's OPEN
            # (signal was generated on the previous bar's close)
            if i > 0:
                prev_signal = int(bars.loc[i - 1, "Signal"])
                open_price = float(row["Open"])

                if prev_signal == 1 and not portfolio.in_position:
                    trade = broker.execute_buy(date, open_price, portfolio.cash)
                    portfolio.open_position(trade)

                elif prev_signal == -1 and portfolio.in_position:
                    trade = broker.execute_sell(date, open_price, portfolio.open_trade)  # type: ignore[arg-type]
                    portfolio.close_position(trade)

        # 5. Force-close any open position at last close
        if portfolio.in_position:
            last_row = bars.iloc[-1]
            last_date = last_row["Date"] if "Date" in last_row else last_row.iloc[0]
            last_close = float(last_row["Close"])
            logger.info("Force-closing open position at last bar (%.4f).", last_close)
            trade = broker.execute_sell(last_date, last_close, portfolio.open_trade)  # type: ignore[arg-type]
            portfolio.close_position(trade)

        self.portfolio = portfolio
        self._ran = True

        logger.info("Backtest complete. Total trades: %d", len(portfolio.closed_trades))
        logger.info("═" * 60)
        return self

    def get_equity_curve(self) -> pd.Series:
        self._check_ran()
        return self.portfolio.get_equity_curve()  # type: ignore[union-attr]

    def get_trades(self) -> pd.DataFrame:
        """Return a DataFrame with one row per completed trade."""
        self._check_ran()
        trades = self.portfolio.closed_trades  # type: ignore[union-attr]
        if not trades:
            return pd.DataFrame()
        records = [
            {
                "trade_id": t.trade_id,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "return_pct": t.return_pct * 100,
                "duration_days": t.duration_days,
                "winner": t.is_winner,
            }
            for t in trades
        ]
        return pd.DataFrame(records)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _check_ran(self) -> None:
        if not self._ran:
            raise RuntimeError("Call engine.run() before accessing results.")
