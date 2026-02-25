"""
Data cleaner: validates integrity of OHLCV DataFrames and applies standard fixes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_and_clean(df: pd.DataFrame, fill_method: str = "ffill") -> pd.DataFrame:
    """
    Validate and clean an OHLCV DataFrame.

    Checks performed
    ----------------
    - Required columns present (Open, High, Low, Close, Volume)
    - No fully-empty rows
    - OHLC consistency  (High >= Low, High >= Open/Close, Low <= Open/Close)
    - No negative prices or volumes
    - Sorted chronologically

    Parameters
    ----------
    df          : Raw OHLCV DataFrame.
    fill_method : How to impute missing values — 'ffill', 'bfill', or 'drop'.

    Returns
    -------
    pd.DataFrame  Cleaned copy.
    """
    df = df.copy()

    # 1. Required columns
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_len = len(df)

    # 2. Drop fully-NaN rows
    df.dropna(how="all", inplace=True)

    # 3. Sort by index
    df.sort_index(inplace=True)

    # 4. Handle remaining NaNs in price columns
    price_cols = ["Open", "High", "Low", "Close"]
    if fill_method == "ffill":
        df[price_cols] = df[price_cols].ffill()
    elif fill_method == "bfill":
        df[price_cols] = df[price_cols].bfill()
    elif fill_method == "drop":
        df.dropna(subset=price_cols, inplace=True)
    else:
        raise ValueError(
            f"Unknown fill_method '{fill_method}'. Use 'ffill', 'bfill', or 'drop'."
        )

    df["Volume"] = df["Volume"].fillna(0)

    # 5. Negative price / volume guard
    for col in price_cols:
        invalid = df[col] <= 0
        if invalid.any():
            logger.warning(
                "Found %d non-positive values in '%s'. Replacing with NaN → ffill.",
                invalid.sum(),
                col,
            )
            df.loc[invalid, col] = np.nan
            df[col] = df[col].ffill()

    # 6. OHLC consistency
    inconsistent = (
        (df["High"] < df["Low"])
        | (df["High"] < df["Close"])
        | (df["Low"] > df["Close"])
    )
    if inconsistent.any():
        logger.warning(
            "%d rows with OHLC inconsistencies detected. Clamping High/Low.",
            inconsistent.sum(),
        )
        df.loc[inconsistent, "High"] = df.loc[
            inconsistent, ["Open", "High", "Close"]
        ].max(axis=1)
        df.loc[inconsistent, "Low"] = df.loc[
            inconsistent, ["Open", "Low", "Close"]
        ].min(axis=1)

    removed = initial_len - len(df)
    if removed:
        logger.info(
            "Removed %d rows during cleaning (%d → %d).", removed, initial_len, len(df)
        )

    logger.info(
        "Data validation complete. Shape: %s  |  Period: %s → %s",
        df.shape,
        df.index[0].date(),
        df.index[-1].date(),
    )

    return df
