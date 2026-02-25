# RSI Backtesting Engine

> A professional, production-grade backtesting framework for RSI mean-reversion strategies on crypto and equity markets.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Pytest](https://img.shields.io/badge/Tests-pytest-green?logo=pytest)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Code Style](https://img.shields.io/badge/style-PEP8-black)](https://peps.python.org/pep-0008/)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Execution Model](#execution-model)
- [Modules](#modules)
- [Configuration](#configuration)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Performance Metrics](#performance-metrics)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Extending the Engine](#extending-the-engine)
- [Known Limitations](#known-limitations)

---

## Overview

This project implements a **bar-by-bar backtesting engine** for a classic RSI mean-reversion strategy. It is designed from the ground up to avoid look-ahead bias, apply realistic transaction costs, and produce institutional-quality performance reports.

The strategy logic is straightforward: enter a long position when the market is oversold (RSI < 30) and exit when it becomes overbought (RSI > 70). What makes this implementation professional is the surrounding infrastructure — a modular pipeline that cleanly separates data ingestion, signal generation, order execution, portfolio accounting, and metrics computation.

**Default configuration** runs on `BTC-USD` daily data from 2020-01-01 to 2024-12-31 with the following costs:

| Parameter | Value |
|---|---|
| Initial Capital | $10,000 |
| Commission (per side) | 0.10% |
| Slippage (per fill) | 0.05% |
| RSI Period | 14 |
| Oversold Threshold | 30 |
| Overbought Threshold | 70 |
| Risk-Free Rate | 5% annual |
| Trading Days/Year | 365 (crypto) |

---

## Architecture

The engine follows a strict unidirectional data pipeline:

```
yfinance / CSV
      │
      ▼
┌─────────────┐
│ DataLoader  │  load_ohlcv() — downloads or reads from .parquet cache
└──────┬──────┘
       │  raw OHLCV DataFrame
       ▼
┌─────────────┐
│ DataCleaner │  validate_and_clean() — fills NaN, checks OHLC consistency
└──────┬──────┘
       │  clean OHLCV DataFrame
       ▼
┌──────────────────┐
│ RSI Strategy     │  generate_signals() — computes RSI, emits +1 / -1 / 0
└──────┬───────────┘
       │  signal DataFrame
       ▼
┌──────────────────────────────────────┐
│         BacktestEngine (bar loop)    │
│                                      │
│   BrokerSimulator  ◄──► Portfolio    │
│   (slippage +       (cash, trades,   │
│    commission)       equity curve)   │
└──────┬───────────────────────────────┘
       │  equity curve + closed trades
       ▼
┌──────────────────┐
│ Metrics +        │  compute_metrics(), plot_*()
│ Visualization    │
└──────────────────┘
       │
       ▼
  reports/figures/
  performance_summary.csv
  trades.csv
```

---

## Project Structure

```
.
├── main.py                         # CLI entry point
├── conftest.py                     # pytest path configuration
├── pyproject.toml                  # project metadata & pytest config
├── requirements.txt
│
├── data/                           # cached .parquet files (auto-created)
│   └── BTC-USD_2020-01-01_2024-12-31_1d.parquet
│
├── notebooks/
│   └── 01_rsi_strategy_exploration.ipynb   # exploratory analysis
│
├── reports/
│   ├── performance_summary.csv
│   ├── trades.csv
│   └── figures/                    # all output charts
│       ├── 00_price_history.png
│       ├── 01_rsi_distribution.png
│       ├── 02_rsi_signals.png
│       ├── 03_equity_drawdown.png
│       ├── 04_metrics_dashboard.png
│       ├── 05_trade_analysis.png
│       ├── 06_sensitivity_heatmap.png
│       └── 07_full_report.png
│
├── src/
│   ├── config/
│   │   └── settings.py             # all global constants
│   ├── data/
│   │   ├── data_loader.py          # yfinance download + parquet cache
│   │   └── data_cleaner.py         # OHLCV validation and cleaning
│   ├── indicators/
│   │   └── rsi.py                  # Wilder RSI implementation (no TA-Lib)
│   ├── strategy/
│   │   └── rsi_strategy.py         # signal generation logic
│   ├── execution/
│   │   ├── broker_simulator.py     # slippage + commission model
│   │   └── trade.py                # Trade dataclass
│   ├── portfolio/
│   │   └── portfolio.py            # cash & equity tracking
│   ├── metrics/
│   │   ├── performance.py          # Sharpe, CAGR, Sortino, Calmar, etc.
│   │   └── drawdown.py             # drawdown curve and max DD
│   ├── backtester/
│   │   └── engine.py               # main orchestrator
│   ├── visualization/
│   │   └── plots.py                # matplotlib chart suite
│   └── utils/
│       └── helpers.py
│
└── tests/
    ├── test_backtester.py          # integration tests (mocked data)
    ├── test_indicators.py          # unit tests for RSI logic
    └── test_metrics.py             # unit tests for performance metrics
```

---

## Execution Model

The engine is designed to be **free of look-ahead bias**. The execution sequence per bar is:

1. At **bar N close** → RSI is computed → signal is recorded (`+1`, `-1`, or `0`).
2. At **bar N+1 open** → the broker fills the order at that bar's open price.
3. Slippage is applied **adversely**: buys fill higher, sells fill lower.
4. Commission is charged **per side** as a percentage of notional.

```
Bar N       Bar N+1
─────────   ──────────────────────────────
Close →     Open (fill) → High, Low, Close
Signal=+1   BUY executed here
```

This means a signal generated on Friday's close will be filled at Monday's open — a realistic simulation that avoids the common pitfall of trading on the same bar the signal is generated.

**Force-close:** Any open position at the end of the data range is closed at the last bar's close price to ensure accurate final P&L.

---

## Modules

### `src/config/settings.py`

Single source of truth for all parameters. Modify here to change strategy behavior without touching logic code.

```python
INITIAL_CAPITAL  = 10_000.0   # USD
COMMISSION_RATE  = 0.001      # 0.10% per side
SLIPPAGE_RATE    = 0.0005     # 0.05% adverse impact
RSI_PERIOD       = 14
RSI_OVERSOLD     = 30.0
RSI_OVERBOUGHT   = 70.0
RISK_FREE_RATE   = 0.05       # 5% annual, for Sharpe
TRADING_DAYS     = 365        # use 252 for equities
```

---

### `src/data/data_loader.py`

Downloads OHLCV data via `yfinance` and caches it locally as `.parquet` to avoid redundant network calls.

```python
from src.data.data_loader import load_ohlcv

df = load_ohlcv("BTC-USD", "2020-01-01", "2024-12-31", "1d")
```

Also exposes `load_from_csv(filepath)` for loading local data files.

---

### `src/data/data_cleaner.py`

Validates and repairs OHLCV data before it enters the pipeline. Checks performed:

- All required columns present (`Open`, `High`, `Low`, `Close`, `Volume`)
- No fully-empty rows
- OHLC consistency: `High >= Low`, `High >= Close`, `Low <= Close`
- Non-positive price guard with `ffill` recovery
- Chronological sort

```python
from src.data.data_cleaner import validate_and_clean

clean_df = validate_and_clean(raw_df, fill_method="ffill")  # or "bfill" / "drop"
```

---

### `src/indicators/rsi.py`

Pure-Python, zero-dependency RSI implementation using **Wilder's exponential smoothing** (`alpha = 1/period`). Handles edge cases explicitly:

- `avg_loss == 0` → RSI = 100 (all gains, by definition)
- `avg_gain == 0` → RSI = 0 (all losses, by definition)
- Division by zero prevention via `replace(0, np.nan)`

```python
from src.indicators.rsi import compute_rsi, add_rsi_to_df

rsi_series = compute_rsi(df["Close"], period=14)
df_with_rsi = add_rsi_to_df(df, period=14)
```

> **Why no TA-Lib?** Avoiding external C-compiled dependencies makes the project portable and the math transparent for learning and auditing.

---

### `src/strategy/rsi_strategy.py`

Stateful signal generator that iterates bar by bar to correctly model the "not already in position" constraint — something a purely vectorized approach cannot handle without re-entry artifacts.

**Signal convention:**

| Value | Meaning |
|---|---|
| `+1` | BUY — enter long |
| `-1` | SELL — exit long |
| `0` | HOLD — no action |

```python
from src.strategy.rsi_strategy import generate_signals

signal_df = generate_signals(clean_df, rsi_period=14, oversold=30, overbought=70)
```

---

### `src/execution/broker_simulator.py`

Applies realistic transaction costs to simulated fills:

```
BUY  fill_price  = raw_price × (1 + slippage_rate)   # you pay more
SELL fill_price  = raw_price × (1 − slippage_rate)   # you receive less

commission = notional × commission_rate   # charged per side
```

Full capital is deployed on each buy (no partial position sizing). Shares are calculated as:

```
shares = (capital − commission) / fill_price
```

---

### `src/execution/trade.py`

A `@dataclass` representing a single round-trip trade. All fields are populated at entry; P&L fields are computed at `trade.close()`.

**Key computed fields (post-close):**

| Field | Formula |
|---|---|
| `gross_pnl` | `(exit_price − entry_price) × shares` |
| `net_pnl` | `gross_pnl − (entry_commission + exit_commission)` |
| `return_pct` | `net_pnl / (entry_price × shares)` |
| `duration_days` | `(exit_date − entry_date)` in fractional days |

---

### `src/portfolio/portfolio.py`

Tracks portfolio state across the simulation:

- `cash` — available cash (0 when in position, proceeds after close)
- `open_trade` — current `Trade` object or `None`
- `closed_trades` — list of all completed `Trade` objects
- `_equity_history` — bar-by-bar mark-to-market snapshots

Mark-to-market valuation:
- **In position:** `equity = shares × current_close`
- **Out of position:** `equity = cash`

---

### `src/metrics/performance.py`

Computes the full performance report from the equity curve and trade list:

| Category | Metrics |
|---|---|
| **Returns** | Total Return (%), CAGR (%) |
| **Risk** | Annual Volatility (%), Max Drawdown (%), Max DD Duration |
| **Risk-Adjusted** | Sharpe Ratio, Sortino Ratio, Calmar Ratio |
| **Trade Stats** | Total Trades, Win Rate (%), Avg Win ($), Avg Loss ($), Profit Factor, Best/Worst Trade, Avg Duration |

```python
from src.metrics.performance import compute_metrics, format_metrics_table

metrics = compute_metrics(equity_curve, closed_trades)
table   = format_metrics_table(metrics)
```

---

### `src/metrics/drawdown.py`

Drawdown utilities:

```python
from src.metrics.drawdown import compute_drawdown, max_drawdown, drawdown_duration

dd_df     = compute_drawdown(equity)   # DataFrame with Equity, Peak, Drawdown, Drawdown_Pct
max_dd    = max_drawdown(equity)       # float, e.g. -45.3 (%)
dd_dur    = drawdown_duration(equity)  # int, longest consecutive bars underwater
```

---

### `src/backtester/engine.py`

The main orchestrator. Chains all modules together in the correct order and exposes a clean public API:

```python
from src.backtester.engine import BacktestEngine

engine = BacktestEngine(
    ticker="BTC-USD",
    start="2020-01-01",
    end="2024-12-31",
    initial_capital=10_000,
    commission_rate=0.001,
    slippage_rate=0.0005,
    rsi_period=14,
    rsi_oversold=30,
    rsi_overbought=70,
)

engine.run()

equity = engine.get_equity_curve()   # pd.Series
trades = engine.get_trades()         # pd.DataFrame
```

Method chaining is supported: `engine.run().get_equity_curve()`.

---

### `src/visualization/plots.py`

Four chart functions, all returning `plt.Figure` and optionally saving to disk:

| Function | Output |
|---|---|
| `plot_equity_curve()` | Strategy vs. Buy & Hold benchmark |
| `plot_drawdown()` | Underwater equity curve |
| `plot_rsi_with_trades()` | Two-panel: price with markers + RSI with zones |
| `plot_full_report()` | Three-panel combined figure (equity + RSI + drawdown) |

---

## Configuration

All parameters live in `src/config/settings.py`. Key tuning knobs:

```python
# Change asset and date range
DEFAULT_TICKER = "ETH-USD"
DEFAULT_START  = "2021-01-01"
DEFAULT_END    = "2024-12-31"

# Adjust RSI sensitivity
RSI_OVERSOLD   = 25.0   # more selective entries
RSI_OVERBOUGHT = 75.0   # more selective exits

# Use equity market convention
TRADING_DAYS   = 252

# Increase costs for conservative estimates
COMMISSION_RATE = 0.002
SLIPPAGE_RATE   = 0.001
```

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/your-username/rsi-backtesting-engine.git
cd rsi-backtesting-engine

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`** (minimum):

```
yfinance>=0.2.36
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
pyarrow>=15.0          # parquet support
pytest>=7.4
```

---

## Quickstart

### Run the full backtest (default settings)

```bash
python main.py
```

### Run with custom parameters

```python
from src.backtester.engine import BacktestEngine
from src.metrics.performance import compute_metrics, format_metrics_table
from src.visualization.plots import plot_full_report

engine = BacktestEngine(
    ticker="ETH-USD",
    start="2021-01-01",
    end="2024-12-31",
    rsi_oversold=25,
    rsi_overbought=75,
    initial_capital=50_000,
).run()

equity = engine.get_equity_curve()
trades_df = engine.get_trades()

metrics = compute_metrics(equity, engine.portfolio.closed_trades)
print(format_metrics_table(metrics).to_string())

fig = plot_full_report(
    equity=equity,
    data=engine.data,
    trades=engine.portfolio.closed_trades,
    metrics=metrics,
)
```

### Load data from CSV instead of yfinance

```python
from src.data.data_loader import load_from_csv

df = load_from_csv("path/to/your/data.csv")
```

---

## Performance Metrics

The engine computes the following metrics. Definitions are given for reference:

**Sharpe Ratio**
> `(mean_excess_return / std_excess_return) × √trading_days`
> Measures return per unit of total risk. Excess return = daily return − daily risk-free rate.

**Sortino Ratio**
> Same as Sharpe but uses only the standard deviation of *negative* excess returns (downside deviation). Penalizes harmful volatility only.

**Calmar Ratio**
> `CAGR / |Max Drawdown|`
> Measures annual return relative to the worst peak-to-trough loss. Useful for comparing strategies with different drawdown profiles.

**CAGR (Compound Annual Growth Rate)**
> `(final_equity / initial_equity)^(1 / n_years) − 1`
> Annualized geometric return.

**Max Drawdown**
> The largest peak-to-trough decline in portfolio value, expressed as a percentage. A negative number (e.g., −38.5%).

**Profit Factor**
> `sum(winning_trades) / |sum(losing_trades)|`
> Values above 1.0 indicate a profitable strategy. Above 1.5 is generally considered good.

---

## Testing

The test suite covers unit and integration scenarios using synthetic data to eliminate network dependencies.

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_indicators.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

### Test coverage by module

| File | Scope |
|---|---|
| `tests/test_indicators.py` | RSI bounds, edge cases, error handling |
| `tests/test_backtester.py` | Full engine integration, equity invariants, trade schema |
| `tests/test_metrics.py` | Sharpe, drawdown, CAGR calculations |

Integration tests mock `load_ohlcv` with deterministic synthetic OHLCV data (`seed=42`), ensuring reproducibility and zero network calls in CI.

```python
# Example: how integration tests avoid network calls
with patch("src.backtester.engine.load_ohlcv", return_value=SYNTHETIC_DATA):
    engine.run()
```

---

## Design Decisions

**Why bar-by-bar iteration instead of vectorized signals?**

Pure vectorization is faster but makes it difficult to correctly model the "already in position" constraint — a vectorized approach would generate multiple consecutive buy signals without any position tracking. The stateful loop trades a small performance cost for correctness.

**Why no TA-Lib or pandas-ta?**

The RSI implementation in `src/indicators/rsi.py` uses only `numpy` and `pandas`. This eliminates a C-compiled binary dependency (`TA-Lib` requires a system library), making the project installable via `pip` on any platform without additional setup.

**Why Parquet for caching?**

Parquet preserves `datetime` index types and timezone info (`UTC`) without conversion loss. It is also significantly faster to read than CSV for large datasets and produces smaller files.

**Why is commission deducted before calculating shares?**

This mirrors real-world execution. When you place a $10,000 buy order with 0.1% commission, you effectively have $9,990 to spend on shares. Computing `shares = (capital - commission) / fill_price` ensures the portfolio never over-allocates.

**Why force-close at the last bar?**

An open position at the end of the test period represents unrealized P&L. Closing it at the last close ensures the equity curve and all metrics reflect actual realizable returns.

---

## Extending the Engine

### Add a new strategy

1. Create `src/strategy/my_strategy.py` with a `generate_signals(df) -> pd.DataFrame` function that returns a `Signal` column (`+1`, `-1`, `0`).
2. Replace the `generate_signals` call in `engine.py` with your function.
3. Add tests in `tests/test_my_strategy.py`.

### Add a new indicator

Follow the pattern in `src/indicators/rsi.py`: accept a `pd.Series`, return a `pd.Series` of the same length with NaN for the warm-up period.

### Add position sizing

Modify `BrokerSimulator.execute_buy()` to accept a `size` parameter (fraction of capital, e.g., 0.5 for 50%). Update `BacktestEngine` to pass a size from the signal DataFrame.

### Run a parameter sweep

```python
import itertools
import pandas as pd

results = []
for period, oversold, overbought in itertools.product([10, 14, 21], [25, 30], [70, 75]):
    engine = BacktestEngine(rsi_period=period, rsi_oversold=oversold, rsi_overbought=overbought).run()
    equity = engine.get_equity_curve()
    metrics = compute_metrics(equity, engine.portfolio.closed_trades)
    results.append({"period": period, "oversold": oversold, "overbought": overbought, **metrics})

df_results = pd.DataFrame(results).sort_values("Sharpe Ratio", ascending=False)
```

---

## Known Limitations

**Single long position only.** The engine does not support short selling, multiple simultaneous positions, or leverage. The portfolio is either 100% in cash or 100% in the asset.

**Full capital deployment.** Each buy signal deploys all available cash. There is no position sizing model (e.g., Kelly criterion, fixed fractional).

**No overnight/weekend gap handling.** For crypto, this is largely irrelevant. For equities on daily bars, gap risk at open is captured in the slippage model but not explicitly modeled as a separate risk factor.

**Slippage is proportional, not market-impact-based.** The current model applies a flat percentage regardless of order size. For large capital amounts relative to market liquidity, a market impact model (e.g., square-root model) would be more accurate.

**No dividends or splits.** The data loader uses `auto_adjust=True` in yfinance which handles split-adjusted prices, but dividend reinvestment is not modeled.

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.