"""
Drawdown analysis: calculates drawdown series and maximum drawdown.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_drawdown(equity: pd.Series) -> pd.DataFrame:
    """
    Compute the running drawdown from peak equity.

    Parameters
    ----------
    equity : pd.Series  Time-series of portfolio equity values.

    Returns
    -------
    pd.DataFrame  with columns:
        - Equity        : Original equity values.
        - Peak          : Running maximum equity up to each date.
        - Drawdown      : Absolute drawdown (Equity − Peak).
        - Drawdown_Pct  : Relative drawdown as a percentage (0 to −100).
    """
    equity = equity.copy()
    peak = equity.cummax()
    dd = equity - peak
    dd_pct = (dd / peak) * 100.0

    return pd.DataFrame(
        {
            "Equity": equity,
            "Peak": peak,
            "Drawdown": dd,
            "Drawdown_Pct": dd_pct,
        }
    )


def max_drawdown(equity: pd.Series) -> float:
    """
    Return the maximum drawdown as a negative percentage.

    Example: −45.3 means the portfolio fell 45.3% from its peak.
    """
    dd_df = compute_drawdown(equity)
    return float(dd_df["Drawdown_Pct"].min())


def drawdown_duration(equity: pd.Series) -> int:
    """
    Return the length in bars of the longest drawdown period
    (from peak to full recovery, or to end of series).
    """
    dd_df = compute_drawdown(equity)
    in_dd = (dd_df["Drawdown_Pct"] < 0).astype(int)
    # Count consecutive bars in drawdown
    groups = (in_dd != in_dd.shift()).cumsum()
    durations = in_dd.groupby(groups).sum()
    return int(durations.max()) if len(durations) else 0
