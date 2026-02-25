"""
Visualization: professional-grade charts for the backtest results.

Produces:
  - Equity curve (vs buy-and-hold benchmark)
  - Drawdown chart
  - Combined figure for reports
  - RSI + price chart with trade annotations
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config.settings import FIGURES_DIR
from src.execution.trade import Trade
from src.metrics.drawdown import compute_drawdown

# ── Style defaults ─────────────────────────────────────────────────────────────
STRATEGY_COLOR = "#2196F3"  # blue
BENCHMARK_COLOR = "#90A4AE"  # grey
DRAWDOWN_COLOR = "#EF5350"  # red
BUY_COLOR = "#26A69A"  # teal
SELL_COLOR = "#EF5350"  # red
RSI_COLOR = "#7E57C2"  # purple

plt.rcParams.update(
    {
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FFFFFF",
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.6,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_equity_curve(
    equity: pd.Series,
    benchmark_price: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot strategy equity curve vs optional buy-and-hold benchmark.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        equity.index,
        equity.values,
        color=STRATEGY_COLOR,
        linewidth=1.8,
        label="RSI Strategy",
    )

    if benchmark_price is not None:
        bm = benchmark_price.reindex(equity.index, method="ffill")
        bm_equity = (bm / bm.iloc[0]) * equity.iloc[0]
        ax.plot(
            bm.index,
            bm_equity.values,
            color=BENCHMARK_COLOR,
            linewidth=1.2,
            linestyle="--",
            label="Buy & Hold",
            alpha=0.8,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Portfolio Value (USD)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    ax.legend(framealpha=0.5)
    fig.tight_layout()

    _save_figure(fig, save_path, "equity_curve.png")
    return fig


def plot_drawdown(
    equity: pd.Series,
    title: str = "Drawdown",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the percentage drawdown from peak equity over time.
    """
    dd_df = compute_drawdown(equity)
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.fill_between(
        dd_df.index,
        dd_df["Drawdown_Pct"],
        0,
        color=DRAWDOWN_COLOR,
        alpha=0.4,
        label="Drawdown",
    )
    ax.plot(dd_df.index, dd_df["Drawdown_Pct"], color=DRAWDOWN_COLOR, linewidth=1.0)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    _save_figure(fig, save_path, "drawdown.png")
    return fig


def plot_rsi_with_trades(
    data: pd.DataFrame,
    trades: List[Trade],
    rsi_col: str = "RSI_14",
    oversold: float = 30.0,
    overbought: float = 70.0,
    title: str = "Price & RSI with Trade Signals",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel chart: price (top) with buy/sell markers, RSI (bottom) with thresholds.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.subplots_adjust(hspace=0.05)

    # ── Price panel ────────────────────────────────────────────────────────────
    ax1.plot(data.index, data["Close"], color="#37474F", linewidth=1.2, label="Close")

    for trade in trades:
        ax1.axvline(trade.entry_date, color=BUY_COLOR, linewidth=0.7, alpha=0.5)
        ax1.scatter(
            trade.entry_date,
            trade.entry_price,
            marker="^",
            color=BUY_COLOR,
            s=80,
            zorder=5,
            label="_nolegend_",
        )
        if trade.exit_date:
            ax1.scatter(
                trade.exit_date,
                trade.exit_price,
                marker="v",
                color=SELL_COLOR,
                s=80,
                zorder=5,
                label="_nolegend_",
            )

    # Custom legend entries
    ax1.scatter([], [], marker="^", color=BUY_COLOR, s=80, label="Entry (Buy)")
    ax1.scatter([], [], marker="v", color=SELL_COLOR, s=80, label="Exit (Sell)")
    ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(framealpha=0.5, loc="upper left")

    # ── RSI panel ──────────────────────────────────────────────────────────────
    if rsi_col in data.columns:
        ax2.plot(data.index, data[rsi_col], color=RSI_COLOR, linewidth=1.2, label="RSI")
        ax2.axhline(
            overbought,
            color=SELL_COLOR,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"Overbought ({overbought:.0f})",
        )
        ax2.axhline(
            oversold,
            color=BUY_COLOR,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"Oversold ({oversold:.0f})",
        )
        ax2.fill_between(
            data.index,
            data[rsi_col],
            overbought,
            where=(data[rsi_col] > overbought),
            color=SELL_COLOR,
            alpha=0.15,
        )
        ax2.fill_between(
            data.index,
            data[rsi_col],
            oversold,
            where=(data[rsi_col] < oversold),
            color=BUY_COLOR,
            alpha=0.15,
        )
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI", fontsize=11)
        ax2.legend(framealpha=0.5, loc="upper left")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    _save_figure(fig, save_path, "rsi_signals.png")
    return fig


def plot_full_report(
    equity: pd.Series,
    data: pd.DataFrame,
    trades: List[Trade],
    metrics: dict,
    rsi_col: str = "RSI_14",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel combined report figure:
    Top    : Equity curve with benchmark
    Middle : RSI with signal levels
    Bottom : Drawdown
    """
    dd_df = compute_drawdown(equity)
    bm_price = data["Close"]
    bm_equity = (
        bm_price.reindex(equity.index, method="ffill")
        / bm_price.reindex(equity.index, method="ffill").iloc[0]
    ) * equity.iloc[0]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 14),
        sharex=False,
        gridspec_kw={"height_ratios": [2.5, 1.5, 1.5]},
    )

    # ── Equity ─────────────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(
        equity.index,
        equity.values,
        color=STRATEGY_COLOR,
        linewidth=2,
        label="RSI Strategy",
    )
    ax1.plot(
        bm_equity.index,
        bm_equity.values,
        color=BENCHMARK_COLOR,
        linewidth=1.2,
        linestyle="--",
        label="Buy & Hold",
        alpha=0.8,
    )
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    ax1.set_title(
        f"RSI Backtesting Engine  ·  Total Return: {total_ret:+.1f}%  |  "
        f"Sharpe: {metrics.get('Sharpe Ratio', 0):.2f}  |  "
        f"Max DD: {metrics.get('Max Drawdown (%)', 0):.1f}%",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylabel("Portfolio Value (USD)", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(framealpha=0.5)

    # Entry/exit vertical shading
    for trade in trades:
        if trade.exit_date:
            ax1.axvspan(
                trade.entry_date,
                trade.exit_date,
                alpha=0.06,
                color=BUY_COLOR if trade.is_winner else SELL_COLOR,
            )

    # ── RSI ────────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    if rsi_col in data.columns:
        ax2.plot(data.index, data[rsi_col], color=RSI_COLOR, linewidth=1.2)
        ax2.axhline(70, color=SELL_COLOR, linestyle="--", linewidth=1.0, alpha=0.7)
        ax2.axhline(30, color=BUY_COLOR, linestyle="--", linewidth=1.0, alpha=0.7)
        ax2.fill_between(
            data.index,
            data[rsi_col],
            70,
            where=(data[rsi_col] > 70),
            color=SELL_COLOR,
            alpha=0.15,
        )
        ax2.fill_between(
            data.index,
            data[rsi_col],
            30,
            where=(data[rsi_col] < 30),
            color=BUY_COLOR,
            alpha=0.15,
        )
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI(14)", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.tick_params(axis="x", rotation=30)

    # ── Drawdown ───────────────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(
        dd_df.index, dd_df["Drawdown_Pct"], 0, color=DRAWDOWN_COLOR, alpha=0.4
    )
    ax3.plot(dd_df.index, dd_df["Drawdown_Pct"], color=DRAWDOWN_COLOR, linewidth=1.0)
    ax3.set_ylabel("Drawdown (%)", fontsize=11)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax3.tick_params(axis="x", rotation=30)

    fig.tight_layout(rect=[0, 0, 1, 1])
    _save_figure(fig, save_path, "full_report.png")
    return fig


# ── Internal helpers ───────────────────────────────────────────────────────────


def _save_figure(fig: plt.Figure, save_path: Optional[str], default_name: str) -> None:
    path = save_path or str(Path(FIGURES_DIR) / default_name)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
