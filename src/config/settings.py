"""
Global configuration parameters for the RSI backtesting engine.
All monetary values are in USD. Rates are expressed as decimals.
"""

# ── Capital ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL: float = 10_000.0  # Starting portfolio value in USD

# ── Execution costs ────────────────────────────────────────────────────────────
COMMISSION_RATE: float = 0.001  # 0.10% per trade (entry and exit)
SLIPPAGE_RATE: float = 0.0005  # 0.05% adverse price movement per fill

# ── RSI indicator ─────────────────────────────────────────────────────────────
RSI_PERIOD: int = 14  # Standard Wilder period
RSI_OVERSOLD: float = 30.0  # Buy signal threshold
RSI_OVERBOUGHT: float = 70.0  # Sell signal threshold

# ── Data ───────────────────────────────────────────────────────────────────────
DEFAULT_TICKER: str = "BTC-USD"
DEFAULT_START: str = "2020-01-01"
DEFAULT_END: str = "2024-12-31"
DEFAULT_INTERVAL: str = "1d"  # yfinance interval string

# ── Metrics ────────────────────────────────────────────────────────────────────
RISK_FREE_RATE: float = 0.05  # Annual risk-free rate for Sharpe calculation
TRADING_DAYS: int = 365  # Crypto trades 365 days; use 252 for equities

# ── Reporting ──────────────────────────────────────────────────────────────────
REPORTS_DIR: str = "reports"
FIGURES_DIR: str = "reports/figures"
