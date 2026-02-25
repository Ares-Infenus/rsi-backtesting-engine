"""
RSI indicator — fully vectorized implementation using Wilder's smoothing method.

This avoids any external TA library dependency, making the code portable
and easy to understand for learning purposes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import RSI_PERIOD


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) using Wilder's exponential
    smoothing (equivalent to EMA with alpha = 1/period).

    Parameters
    ----------
    close  : pd.Series  — Closing prices, indexed by datetime.
    period : int        — Look-back window (default 14).

    Returns
    -------
    pd.Series  RSI values in [0, 100], same index as *close*.
               First *period* values will be NaN (warm-up).
    """
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series.")
    if period < 2:
        raise ValueError("RSI period must be >= 2.")
    if len(close) < period + 1:
        raise ValueError(
            f"Need at least {period + 1} data points to compute RSI({period})."
        )

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder smoothing: seed with simple mean, then apply EWMA
    alpha = 1.0 / period

    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)  # avoid division by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # When avg_loss == 0, RSI is 100 by definition
    rsi[avg_loss == 0] = 100.0
    # When avg_gain == 0, RSI is 0 by definition
    rsi[avg_gain == 0] = 0.0

    rsi.name = f"RSI_{period}"
    return rsi


def add_rsi_to_df(
    df: pd.DataFrame, period: int = RSI_PERIOD, column: str = "Close"
) -> pd.DataFrame:
    """
    Convenience wrapper that appends an RSI column to an OHLCV DataFrame.

    Parameters
    ----------
    df     : DataFrame with at least a *column* price series.
    period : RSI look-back window.
    column : Name of the price column to compute RSI on.

    Returns
    -------
    pd.DataFrame  Original DataFrame with a new f'RSI_{period}' column.
    """
    df = df.copy()
    df[f"RSI_{period}"] = compute_rsi(df[column], period)
    return df
