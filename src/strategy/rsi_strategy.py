"""
RSI Strategy: generates buy/sell signals based on RSI thresholds.

Signal convention
-----------------
  +1  →  BUY  (enter long position)
  -1  →  SELL (exit long position)
   0  →  HOLD (no action)
"""

from __future__ import annotations

import pandas as pd

from src.config.settings import RSI_OVERSOLD, RSI_OVERBOUGHT, RSI_PERIOD
from src.indicators.rsi import add_rsi_to_df


def generate_signals(
    df: pd.DataFrame,
    rsi_period: int = RSI_PERIOD,
    oversold: float = RSI_OVERSOLD,
    overbought: float = RSI_OVERBOUGHT,
) -> pd.DataFrame:
    """
    Compute RSI and derive entry/exit signals.

    Logic
    -----
    - BUY  when RSI crosses **below** *oversold*  (30) — not already in position.
    - SELL when RSI crosses **above** *overbought* (70) — currently in position.

    The function avoids look-ahead bias by using only information available
    at bar close. Trades are executed on the **next** bar's open price
    (handled by the engine/broker, not here).

    Parameters
    ----------
    df         : Clean OHLCV DataFrame.
    rsi_period : Look-back window for RSI.
    oversold   : RSI level that triggers a buy signal.
    overbought : RSI level that triggers a sell signal.

    Returns
    -------
    pd.DataFrame
        Original columns + RSI_{period} + Signal column.
    """
    df = add_rsi_to_df(df, period=rsi_period)
    rsi_col = f"RSI_{rsi_period}"

    signals = pd.Series(0, index=df.index, name="Signal", dtype=int)
    in_position = False

    for i in range(1, len(df)):
        rsi_val = df[rsi_col].iloc[i]

        if pd.isna(rsi_val):
            continue

        if not in_position and rsi_val < oversold:
            signals.iloc[i] = 1  # BUY signal
            in_position = True
        elif in_position and rsi_val > overbought:
            signals.iloc[i] = -1  # SELL signal
            in_position = False

    df = df.copy()
    df["Signal"] = signals
    df[rsi_col] = df[rsi_col]  # ensure column order is clean

    return df
