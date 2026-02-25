"""
main.py — Entry point for the RSI Backtesting Engine.

Usage
-----
  python main.py

To customise the run, edit src/config/settings.py or pass overrides
directly to BacktestEngine() below.
"""

from __future__ import annotations

import logging

from src.utils.helpers import (
    setup_logging,
    save_metrics_csv,
    save_trades_csv,
    print_metrics_table,
    ensure_dirs,
)
from src.backtester.engine import BacktestEngine
from src.metrics.performance import compute_metrics, format_metrics_table
from src.visualization.plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_rsi_with_trades,
    plot_full_report,
)
from src.config.settings import (
    DEFAULT_TICKER,
    DEFAULT_START,
    DEFAULT_END,
    DEFAULT_INTERVAL,
    INITIAL_CAPITAL,
    COMMISSION_RATE,
    SLIPPAGE_RATE,
    RSI_PERIOD,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    FIGURES_DIR,
    REPORTS_DIR,
)

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for file output


def main() -> None:
    # ── Setup ──────────────────────────────────────────────────────────────────
    setup_logging(logging.INFO)
    ensure_dirs(REPORTS_DIR, FIGURES_DIR)
    logger = logging.getLogger(__name__)

    # ── Run backtest ───────────────────────────────────────────────────────────
    engine = BacktestEngine(
        ticker=DEFAULT_TICKER,
        start=DEFAULT_START,
        end=DEFAULT_END,
        interval=DEFAULT_INTERVAL,
        initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE,
        slippage_rate=SLIPPAGE_RATE,
        rsi_period=RSI_PERIOD,
        rsi_oversold=RSI_OVERSOLD,
        rsi_overbought=RSI_OVERBOUGHT,
    )
    engine.run()

    # ── Extract results ────────────────────────────────────────────────────────
    equity = engine.get_equity_curve()
    trades_df = engine.get_trades()
    data = engine.data  # type: ignore[assignment]
    trades_list = engine.portfolio.closed_trades  # type: ignore[union-attr]

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(equity, trades_list)
    print_metrics_table(metrics)

    # ── Save reports ───────────────────────────────────────────────────────────
    save_metrics_csv(metrics, f"{REPORTS_DIR}/performance_summary.csv")
    save_trades_csv(trades_df, f"{REPORTS_DIR}/trades.csv")

    # ── Generate charts ────────────────────────────────────────────────────────
    rsi_col = f"RSI_{RSI_PERIOD}"

    plot_equity_curve(
        equity,
        benchmark_price=data["Close"],
        title=f"Equity Curve — {DEFAULT_TICKER} ({DEFAULT_START} → {DEFAULT_END})",
        save_path=f"{FIGURES_DIR}/equity_curve.png",
    )
    logger.info("Saved equity_curve.png")

    plot_drawdown(
        equity,
        title=f"Drawdown — {DEFAULT_TICKER}",
        save_path=f"{FIGURES_DIR}/drawdown.png",
    )
    logger.info("Saved drawdown.png")

    plot_rsi_with_trades(
        data,
        trades_list,
        rsi_col=rsi_col,
        oversold=RSI_OVERSOLD,
        overbought=RSI_OVERBOUGHT,
        title=f"RSI Signals — {DEFAULT_TICKER}",
        save_path=f"{FIGURES_DIR}/rsi_signals.png",
    )
    logger.info("Saved rsi_signals.png")

    plot_full_report(
        equity,
        data,
        trades_list,
        metrics,
        rsi_col=rsi_col,
        save_path=f"{FIGURES_DIR}/full_report.png",
    )
    logger.info("Saved full_report.png")

    logger.info("All outputs written to ./%s/", REPORTS_DIR)
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
